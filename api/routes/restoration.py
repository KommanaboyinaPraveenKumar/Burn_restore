"""
api/routes/restoration.py
=========================
FastAPI route handlers for document restoration endpoints.
"""

import os
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_restoration_engine
from api.schemas import (
    BatchRestoreItem,
    BatchRestoreResponse,
    HealthResponse,
    LineRestorationItem,
    ModelInfoResponse,
    RestoreImageResponse,
)
from api.services.restoration_service import RestorationService
from inference.restoration_engine import RestorationEngine

router = APIRouter(tags=["Document Restoration"])


def get_restoration_service(
    engine: RestorationEngine = Depends(get_restoration_engine),
) -> RestorationService:
    return RestorationService(engine)


@router.get("/health", response_model=HealthResponse)
def health_check(engine: RestorationEngine = Depends(get_restoration_engine)):
    """Health status, device availability, and checkpoint readiness."""
    status_str = "healthy" if engine.is_loaded else "degraded"
    ckpt_exists = bool(engine.checkpoint_path and os.path.isfile(engine.checkpoint_path))

    return HealthResponse(
        status=status_str,
        model_loaded=engine.is_loaded,
        device=str(engine.device),
        checkpoint_path=engine.checkpoint_path,
        checkpoint_available=ckpt_exists,
        timestamp=time.time(),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(engine: RestorationEngine = Depends(get_restoration_engine)):
    """Detailed model architecture, parameter counts, and active configuration."""
    status_data = engine.get_model_status()
    return ModelInfoResponse(
        model_architecture=status_data["model_architecture"],
        device=status_data["device"],
        total_parameters=status_data["total_parameters"],
        trainable_parameters=status_data["trainable_parameters"],
        load_duration_ms=status_data["load_duration_ms"],
        checkpoint=status_data["checkpoint"],
        config=status_data["config"],
        cuda_memory=status_data.get("cuda_memory"),
    )


@router.post("/restore/image", response_model=RestoreImageResponse)
@router.post("/restore/page", response_model=RestoreImageResponse, include_in_schema=False)
async def restore_document_image(
    file: UploadFile = File(..., description="Document page image file"),
    manual_bbox: Optional[str] = Form(None, description="Optional manual bbox 'x1,y1,x2,y2'"),
    max_lines: Optional[int] = Form(None, description="Maximum damaged lines to restore"),
    include_overlay: bool = Form(True, description="Whether to include Base64 visual overlay"),
    max_new_tokens: Optional[int] = Form(32, description="Maximum new tokens per line"),
    num_beams: int = Form(4, description="Beam search width"),
    page_text: Optional[str] = Form(
        None, description="Surviving text of the page, one printed line per line, top to bottom "
                          "(blank line = illegible line). Without it the model sees only the crop."),
    service: RestorationService = Depends(get_restoration_service),
):
    """Restores side-burnt text from an uploaded document image.
    
    Automatically detects damaged margin boundaries and text lines unless manual_bbox is provided.
    """
    file_bytes = await file.read()
    image = service.decode_uploaded_image(file_bytes, filename=file.filename or "upload.png")

    parsed_bbox = None
    if manual_bbox:
        try:
            parts = [int(p.strip()) for p in manual_bbox.split(",")]
            if len(parts) != 4:
                raise ValueError("Must provide 4 integers: x1,y1,x2,y2")
            parsed_bbox = tuple(parts)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid manual_bbox parameter '{manual_bbox}': {str(e)}",
            )

    # Model inference is blocking; keep it off the event loop
    return await run_in_threadpool(
        service.process_full_image,
        image=image,
        manual_bbox=parsed_bbox,
        max_lines=max_lines,
        include_overlay=include_overlay,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        page_transcripts=parse_page_text(page_text),
    )


def parse_page_text(page_text: Optional[str]) -> Optional[List[str]]:
    """One entry per printed line; surrounding blank lines dropped, inner ones kept."""
    if not page_text or not page_text.strip():
        return None
    lines = [l.rstrip() for l in page_text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


@router.post("/restore/region", response_model=LineRestorationItem)
async def restore_document_region(
    file: UploadFile = File(..., description="Document image"),
    x1: int = Form(..., description="Damage start X (left edge of burn)"),
    y1: int = Form(..., description="Line top Y"),
    x2: int = Form(..., description="Damage end X (right margin)"),
    y2: int = Form(..., description="Line bottom Y"),
    prefix_text: Optional[str] = Form(None, description="Surviving visible prefix text"),
    page_context: Optional[str] = Form(None, description="Surrounding context text"),
    max_new_tokens: Optional[int] = Form(32, description="Max generated tokens"),
    num_beams: int = Form(4, description="Beam search width"),
    service: RestorationService = Depends(get_restoration_service),
):
    """Restores text for a specific user-defined line bounding box."""
    file_bytes = await file.read()
    image = service.decode_uploaded_image(file_bytes, filename=file.filename or "region_upload.png")

    return await run_in_threadpool(
        service.process_manual_region,
        image=image,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        prefix_text=prefix_text,
        page_context=page_context,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )


@router.post("/restore/batch", response_model=BatchRestoreResponse)
async def restore_document_batch(
    files: List[UploadFile] = File(..., description="Batch of document images (max 10)"),
    max_lines_per_doc: Optional[int] = Form(5, description="Max lines to restore per document in batch"),
    service: RestorationService = Depends(get_restoration_service),
):
    """Processes a batch of up to 10 document images sequentially."""
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum limit of 10 images (received {len(files)}).",
        )

    results: List[BatchRestoreItem] = []
    success_count = 0
    fail_count = 0

    for f in files:
        try:
            file_bytes = await f.read()
            img = service.decode_uploaded_image(file_bytes, filename=f.filename or "batch_item.png")
            res = await run_in_threadpool(
                service.process_full_image,
                image=img,
                max_lines=max_lines_per_doc,
                include_overlay=False,  # Omit overlay in batch for performance
            )
            results.append(BatchRestoreItem(filename=f.filename or "doc.png", success=True, result=res))
            success_count += 1
        except Exception as e:
            results.append(BatchRestoreItem(filename=f.filename or "doc.png", success=False, error=str(e)))
            fail_count += 1

    return BatchRestoreResponse(
        total_documents_processed=len(files),
        successful_count=success_count,
        failed_count=fail_count,
        documents=results,
    )

