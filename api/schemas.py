"""
api/schemas.py
==============
Pydantic schemas for the Grounded Document Restoration REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status: healthy, ready, degraded")
    model_loaded: bool = Field(..., description="Whether restoration engine is loaded in memory")
    device: str = Field(..., description="Compute device: cpu or cuda")
    checkpoint_path: Optional[str] = Field(None, description="Path to active checkpoint file")
    checkpoint_available: bool = Field(..., description="Whether checkpoint was verified on disk")
    timestamp: float = Field(..., description="Server timestamp")


class ModelInfoResponse(BaseModel):
    model_architecture: str = Field(..., description="Core neural architecture identifier")
    device: str = Field(..., description="Compute device")
    total_parameters: int = Field(..., description="Total model parameter count")
    trainable_parameters: int = Field(..., description="Trainable model parameter count")
    load_duration_ms: float = Field(..., description="Time taken to load model into memory")
    checkpoint: Dict[str, Any] = Field(..., description="Checkpoint metadata")
    config: Dict[str, Any] = Field(..., description="Active inference configuration")
    cuda_memory: Optional[Dict[str, Any]] = Field(None, description="CUDA memory usage if applicable")


class PhysicalBudgetSchema(BaseModel):
    estimated_budget: float = Field(..., description="Estimated geometric character capacity")
    budget_margin_buffer: float = Field(..., description="Allowed character tolerance buffer")
    max_allowed_characters: float = Field(..., description="Budget + margin buffer")
    generated_character_count: int = Field(..., description="Length of decoded infill text")
    raw_infill_text: str = Field(..., description="Raw infill text directly from decoder")
    cleaned_text: str = Field(..., description="Cleaned text stripped of special tokens")
    excess_characters: int = Field(..., description="Characters exceeding max allowed limit")
    budget_violation: bool = Field(..., description="Whether generation exceeded physical bounds")
    word_snapped_text: Optional[str] = Field(None, description="Safely snapped text at nearest word boundary")
    was_snapped: bool = Field(..., description="Whether word boundary snapping was applied")
    is_heuristic: bool = Field(True, description="Indicates budget is a geometric estimate")
    estimation_method: str = Field(..., description="Method used to calculate budget")


class LineRestorationItem(BaseModel):
    line_idx: int = Field(..., description="Vertical text-line index")
    b_x: int = Field(..., description="Horizontal severance point (px)")
    y0: int = Field(..., description="Line top coordinate (px)")
    y1: int = Field(..., description="Line bottom coordinate (px)")
    crop_bbox: List[int] = Field(..., description="Line crop coordinates [x0, y0, x1, y1]")
    damaged_width_px: int = Field(..., description="Missing horizontal pixel span")
    surviving_prefix: str = Field(..., description="Surviving visible text prompt")
    predicted_restoration: str = Field(..., description="Model-generated missing text infill")
    word_snapped_restoration: Optional[str] = Field(None, description="Restoration snapped to word boundary")
    was_word_snapped: bool = Field(False, description="Whether word snapping modified output")
    raw_generated_tokens: str = Field(..., description="Raw decoded SentencePiece tokens")
    candidates: List[str] = Field(default_factory=list,
                                  description="Ranked alternative endings (suggestions, not recovered text)")
    visual_reading: str = Field("", description="Vision-only transcription of the text visible in the crop")
    physical_budget: PhysicalBudgetSchema = Field(..., description="Budget compliance evaluation")
    estimated_missing_chars: Optional[float] = Field(None, description="Point estimate of missing characters")
    upper_char_budget: Optional[float] = Field(None, description="Calibrated upper bound used for decoding")
    inference_time_ms: float = Field(..., description="Latency in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Quality warnings (e.g. runaway text)")


class BurnDetectionSchema(BaseModel):
    image_height: int = Field(..., description="Height in pixels")
    image_width: int = Field(..., description="Width in pixels")
    burn_bbox: List[int] = Field(..., description="Overall damage bounding box [x1, y1, x2, y2]")
    is_manual: bool = Field(..., description="Whether detection was manually specified")
    detection_confidence: float = Field(..., description="Heuristic detection confidence in [0, 1]")
    confidence_method: str = Field(..., description="Confidence calculation method")
    mean_burn_depth_px: float = Field(..., description="Average horizontal burn depth (px)")
    max_burn_depth_px: float = Field(..., description="Maximum horizontal burn depth (px)")
    severed_lines_count: int = Field(..., description="Number of text lines severed by damage")
    warnings: List[str] = Field(default_factory=list, description="Detection warnings")


class RestoreImageResponse(BaseModel):
    document_shape: List[int] = Field(..., description="Preprocessed document dimensions [H, W, C]")
    burn_detection: BurnDetectionSchema = Field(..., description="Burn boundary detection metadata")
    total_lines_severed: int = Field(..., description="Total damaged lines detected")
    restored_lines_count: int = Field(..., description="Total lines successfully restored")
    restorations: List[LineRestorationItem] = Field(..., description="List of per-line restorations")
    diagnostic_overlay_base64: Optional[str] = Field(None, description="Base64-encoded visual overlay image")
    total_page_inference_ms: float = Field(..., description="Total inference latency for page (ms)")
    avg_latency_per_line_ms: float = Field(..., description="Average latency per restored line (ms)")
    checkpoint_id: str = Field(..., description="Active checkpoint identifier")
    model_architecture: str = Field(..., description="Model architecture name")
    limitations_notice: str = Field(..., description="Scientific limitations and prototype status notice")


class RestoreRegionRequest(BaseModel):
    x1: int = Field(..., description="Left coordinate of damaged line region")
    y1: int = Field(..., description="Top coordinate of damaged line region")
    x2: int = Field(..., description="Right coordinate of damaged line region")
    y2: int = Field(..., description="Bottom coordinate of damaged line region")
    prefix_text: Optional[str] = Field(None, description="Surviving visible text to the left")
    page_context: Optional[str] = Field(None, description="Surrounding document text context")
    max_new_tokens: Optional[int] = Field(32, description="Max tokens generated")
    num_beams: Optional[int] = Field(4, description="Beam search width")
    use_physical_grounding: Optional[bool] = Field(True, description="Enable PhysicalBudgetLogitsProcessor")


class BatchRestoreItem(BaseModel):
    filename: str
    success: bool
    result: Optional[RestoreImageResponse] = None
    error: Optional[str] = None


class BatchRestoreResponse(BaseModel):
    total_documents_processed: int
    successful_count: int
    failed_count: int
    documents: List[BatchRestoreItem]

