"""
api/services/restoration_service.py
==================================
Business logic service connecting FastAPI endpoints to the RestorationEngine.
Handles image byte decoding, validation, Base64 visualization rendering, and error handling.
"""

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from api.schemas import (
    BurnDetectionSchema,
    LineRestorationItem,
    PhysicalBudgetSchema,
    RestoreImageResponse,
)
from inference.burn_detector import SeveredLineRegion
from inference.restoration_engine import RestorationEngine


class RestorationService:
    """Service providing document restoration operations for the API."""

    MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB max upload
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    def __init__(self, engine: RestorationEngine):
        self.engine = engine

    def decode_uploaded_image(self, file_bytes: bytes, filename: str = "upload.png") -> np.ndarray:
        """Validates and decodes image bytes into a BGR numpy array."""
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        if len(file_bytes) > self.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds 25 MB limit ({len(file_bytes) / (1024*1024):.1f} MB).",
            )

        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode image from '{filename}'. File may be corrupted or unsupported.",
            )

        h, w = img.shape[:2]
        if h < 32 or w < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image dimensions ({w}x{h}) are too small for document restoration (minimum 32x32).",
            )

        return img

    def encode_image_base64(self, img: np.ndarray) -> str:
        """Encodes a BGR image array to a Base64 PNG data URL."""
        success, encoded = cv2.imencode(".png", img)
        if not success:
            return ""
        b64_str = base64.b64encode(encoded).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def process_full_image(
        self,
        image: np.ndarray,
        manual_bbox: Optional[Tuple[int, int, int, int]] = None,
        manual_boundary_x: Optional[int] = None,
        max_lines: Optional[int] = None,
        include_overlay: bool = True,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 4,
        page_transcripts: Optional[List[str]] = None,
    ) -> RestoreImageResponse:
        """Executes full-page restoration and formats the response schema."""
        raw_res = self.engine.restore_page(
            image=image,
            page_transcripts=page_transcripts,
            manual_bbox=manual_bbox,
            manual_boundary_x=manual_boundary_x,
            max_lines=max_lines,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        det = raw_res["burn_detection"]
        burn_det_schema = BurnDetectionSchema(
            image_height=det["image_height"],
            image_width=det["image_width"],
            burn_bbox=det["burn_bbox"],
            is_manual=det["is_manual"],
            detection_confidence=det["detection_confidence"],
            confidence_method=det["confidence_method"],
            mean_burn_depth_px=det["mean_burn_depth_px"],
            max_burn_depth_px=det["max_burn_depth_px"],
            severed_lines_count=det["severed_lines_count"],
            warnings=det["warnings"],
        )

        restorations_list: List[LineRestorationItem] = []
        for r in raw_res["restorations"]:
            pb = r["physical_budget"]
            budget_schema = PhysicalBudgetSchema(
                estimated_budget=pb["estimated_budget"],
                budget_margin_buffer=pb["budget_margin_buffer"],
                max_allowed_characters=pb["max_allowed_characters"],
                generated_character_count=pb["generated_character_count"],
                raw_infill_text=pb["raw_infill_text"],
                cleaned_text=pb["cleaned_text"],
                excess_characters=pb["excess_characters"],
                budget_violation=pb["budget_violation"],
                word_snapped_text=pb.get("word_snapped_text"),
                was_snapped=pb.get("was_snapped", False),
                is_heuristic=pb.get("is_heuristic", True),
                estimation_method=pb.get("estimation_method", "geometric_pixel_ratio"),
            )
            item = LineRestorationItem(
                line_idx=r["line_idx"],
                b_x=r["b_x"],
                y0=r["y0"],
                y1=r["y1"],
                crop_bbox=r["crop_bbox"],
                damaged_width_px=r["damaged_width_px"],
                surviving_prefix=r["surviving_prefix"],
                predicted_restoration=r["predicted_restoration"],
                word_snapped_restoration=r.get("word_snapped_restoration"),
                was_word_snapped=r.get("was_word_snapped", False),
                raw_generated_tokens=r["raw_generated_tokens"],
                candidates=r.get("candidates") or [],
                visual_reading=r.get("visual_reading") or "",
                physical_budget=budget_schema,
                estimated_missing_chars=r.get("estimated_missing_chars"),
                upper_char_budget=r.get("upper_char_budget"),
                inference_time_ms=r["inference_time_ms"],
                warnings=r.get("warnings", []),
            )
            restorations_list.append(item)

        # Generate Base64 diagnostic overlay
        overlay_b64 = None
        if include_overlay:
            processed_img = self.engine.preprocess_image(image)
            det_obj = self.engine.detect_burn_region(
                processed_img,
                manual_bbox=manual_bbox,
                manual_boundary_x=manual_boundary_x,
            )
            vis = self.engine.burn_detector.generate_diagnostic_visualization(processed_img, det_obj)
            for r in raw_res["restorations"]:
                txt = f"-> {r['predicted_restoration'][:18]}"
                cv2.putText(
                    vis,
                    txt,
                    (r["b_x"] + 6, (r["y0"] + r["y1"]) // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 70, 220),
                    1,
                )
            overlay_b64 = self.encode_image_base64(vis)

        return RestoreImageResponse(
            document_shape=raw_res["document_shape"],
            burn_detection=burn_det_schema,
            total_lines_severed=raw_res["total_lines_severed"],
            restored_lines_count=raw_res["restored_lines_count"],
            restorations=restorations_list,
            diagnostic_overlay_base64=overlay_b64,
            total_page_inference_ms=raw_res["total_page_inference_ms"],
            avg_latency_per_line_ms=raw_res["avg_latency_per_line_ms"],
            checkpoint_id=raw_res["checkpoint_id"],
            model_architecture=raw_res["model_architecture"],
            limitations_notice=raw_res["limitations_notice"],
        )

    def process_manual_region(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        prefix_text: Optional[str] = None,
        page_context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 4,
        use_physical_grounding: bool = True,
    ) -> LineRestorationItem:
        """Restores a single user-specified bounding box region."""
        processed_img = self.engine.preprocess_image(image)
        h, w = processed_img.shape[:2]

        # Clamp coordinates
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))

        damaged_w = w - x1
        intact_w = x1
        crop_x0 = max(0, x1 - self.engine.config.crop_context_px)
        crop_bbox = (crop_x0, y1, w, y2)

        region = SeveredLineRegion(
            line_idx=0,
            y0=y1,
            y1=y2,
            b_x=x1,
            crop_bbox=crop_bbox,
            damaged_width_px=damaged_w,
            intact_width_px=intact_w,
            line_height_px=y2 - y1,
        )

        r = self.engine.restore_region(
            image=processed_img,
            region=region,
            prefix_text=prefix_text,
            page_context=page_context,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_physical_grounding=use_physical_grounding,
        )

        pb = r["physical_budget"]
        budget_schema = PhysicalBudgetSchema(
            estimated_budget=pb["estimated_budget"],
            budget_margin_buffer=pb["budget_margin_buffer"],
            max_allowed_characters=pb["max_allowed_characters"],
            generated_character_count=pb["generated_character_count"],
            raw_infill_text=pb["raw_infill_text"],
            cleaned_text=pb["cleaned_text"],
            excess_characters=pb["excess_characters"],
            budget_violation=pb["budget_violation"],
            word_snapped_text=pb.get("word_snapped_text"),
            was_snapped=pb.get("was_snapped", False),
            is_heuristic=pb.get("is_heuristic", True),
            estimation_method=pb.get("estimation_method", "geometric_pixel_ratio"),
        )

        return LineRestorationItem(
            line_idx=0,
            b_x=r["b_x"],
            y0=r["y0"],
            y1=r["y1"],
            crop_bbox=r["crop_bbox"],
            damaged_width_px=r["damaged_width_px"],
            surviving_prefix=r["surviving_prefix"],
            predicted_restoration=r["predicted_restoration"],
            word_snapped_restoration=r.get("word_snapped_restoration"),
            was_word_snapped=r.get("was_word_snapped", False),
            raw_generated_tokens=r["raw_generated_tokens"],
            physical_budget=budget_schema,
            inference_time_ms=r["inference_time_ms"],
            warnings=r.get("warnings", []),
        )

