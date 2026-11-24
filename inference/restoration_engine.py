"""
inference/restoration_engine.py
===============================
Central Multimodal Document Restoration Engine.

Encapsulates:
  - Model and tokenizer lifecycle management (loads once, stays in memory).
  - Checkpoint integrity validation and graceful fallback handling.
  - Automatic and manual burn-damage detection via BurnDetector.
  - Geometric character budget estimation via BudgetEstimator.
  - PyTorch inference under torch.inference_mode().
  - Single-region, multi-line, and full-page restoration workflows.
  - Structured, uninflated result reporting with latency and compliance metadata.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Add Burn_restore and repo root to sys.path
_repo_root = Path(__file__).resolve().parent.parent
_burn_dir = _repo_root / "Burn_restore"
# Best evaluated model (README §5); override with the BURN_CKPT environment variable.
DEFAULT_CHECKPOINT = os.environ.get(
    "BURN_CKPT", str(_repo_root / "checkpoints" / "v3_base_full" / "best_checkpoint.pt"))
for p in [str(_burn_dir), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import Config
    from model import ForensicT5Restorer, apply_saved_architecture, load_cut_estimator, upper_budget
    from line_data import build_prompt, crop_line, crop_to_tensor, estimate_budget
except ImportError:
    from Burn_restore.config import Config
    from Burn_restore.model import ForensicT5Restorer, apply_saved_architecture, load_cut_estimator, upper_budget
    from Burn_restore.line_data import build_prompt, crop_line, crop_to_tensor, estimate_budget

from inference.burn_detector import BurnDetectionResult, BurnDetector, SeveredLineRegion
from inference.budget_estimator import BudgetComplianceReport, BudgetEstimator


class RestorationEngine:
    """Production and research inference engine for Grounded Side-Burnt Document Restoration."""

    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        checkpoint_path: Optional[str] = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
        config: Optional[Config] = None,
        lazy_load: bool = False,
        cut_estimator_path: Optional[str] = str(_repo_root / "checkpoints" / "cut_estimator.pt"),
    ):
        self.config = config if config is not None else Config()
        self.cut_estimator_path = cut_estimator_path
        self.cut_estimator, self.cut_meta = None, None
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[ForensicT5Restorer] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.burn_detector = BurnDetector(
            charcoal_threshold=70,
            edge_search_ratio=0.50,
            min_ink_row_density=self.config.min_ink_row_density,
            line_gap_merge_px=self.config.line_gap_merge_px,
            crop_context_px=self.config.crop_context_px,
        )
        self.budget_estimator = BudgetEstimator(
            default_char_width_px=self.config.avg_char_width_px,
            margin_buffer=self.config.budget_margin_buffer,
        )

        self.is_loaded = False
        self.load_duration_ms: float = 0.0
        self.checkpoint_metadata: Dict[str, Any] = {}

        if not lazy_load:
            self.load_model()

    def validate_checkpoint(
        self, path: Optional[str], ckpt: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Verifies checkpoint existence, file size, and dictionary structure.
        Pass an already-loaded ``ckpt`` to avoid reading the file twice."""
        if not path or not os.path.isfile(path):
            return False, f"Checkpoint file '{path}' does not exist on disk.", {}

        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if ckpt is None:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                return False, f"Checkpoint at '{path}' is not a valid dictionary (got {type(ckpt)}).", {}

            has_weights = "model_state_dict" in ckpt or "t5.shared.weight" in ckpt
            if not has_weights:
                return False, f"Checkpoint at '{path}' missing model weights or state dictionary.", {}

            meta = {
                "file_size_mb": round(size_mb, 2),
                "epoch": ckpt.get("epoch", "unknown"),
                "global_step": ckpt.get("global_step", "unknown"),
                "has_config": "config_dict" in ckpt,
            }
            return True, "Checkpoint validated successfully.", meta
        except Exception as e:
            return False, f"Failed to load or parse checkpoint: {str(e)}", {}

    def load_model(self) -> None:
        """Loads model weights and tokenizer onto the designated device."""
        start_t = time.time()
        print(f"[RestorationEngine] Initializing tokenizer ({self.config.text_model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_name)

        ckpt = None
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            # Respect architecture switches the checkpoint was trained with (e.g. text-only)
            saved_cfg = ckpt.get("config_dict", {}) if isinstance(ckpt, dict) else {}
            apply_saved_architecture(self.config, saved_cfg)
            # Weights come from the checkpoint; skip the ImageNet download
            self.config.pretrained_vision = False

        print(f"[RestorationEngine] Instantiating ForensicT5Restorer on {self.device}...")
        self.model = ForensicT5Restorer(self.config).to(self.device)

        if ckpt is not None:
            is_valid, msg, meta = self.validate_checkpoint(self.checkpoint_path, ckpt=ckpt)
            if is_valid:
                state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                # strict=True: a silently half-loaded model is worse than a clear error
                self.model.load_state_dict(state, strict=True)
                self.checkpoint_metadata = {
                    "checkpoint_path": self.checkpoint_path,
                    **meta,
                    "missing_keys_count": 0,
                    "unexpected_keys_count": 0,
                }
                print(f"[RestorationEngine] Loaded checkpoint weights from {self.checkpoint_path}")
            else:
                print(f"[RestorationEngine] Warning: Checkpoint validation failed ({msg}). Using base initialization.")
                self.checkpoint_metadata = {"checkpoint_path": self.checkpoint_path, "status": "unloaded", "error": msg}
        else:
            print(f"[RestorationEngine] No checkpoint provided or file not found. Running initialized base model.")
            self.checkpoint_metadata = {"checkpoint_path": None, "status": "initialized_base_model"}

        self.model.eval()
        # Calibrated upper character budget (optional; falls back to the point estimate)
        self.cut_estimator, self.cut_meta = load_cut_estimator(self.cut_estimator_path, self.device)
        if self.cut_meta:
            print(f"[RestorationEngine] Upper budget: x{self.cut_meta['ub_ratio']:.3f} + {self.cut_meta['ub_add']:.0f} "
                  f"(cut estimator {self.cut_estimator_path})")
        self.is_loaded = True
        self.load_duration_ms = (time.time() - start_t) * 1000.0
        print(f"[RestorationEngine] Model ready in {self.load_duration_ms:.1f} ms.")

    def get_model_status(self) -> Dict[str, Any]:
        """Returns comprehensive health, device, and parameter status."""
        total_p = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        trainable_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0

        cuda_mem = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_mem = {
                "allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024 * 1024), 2),
            }

        return {
            "status": "ready" if self.is_loaded else "uninitialized",
            "model_architecture": "ForensicT5Restorer (T5-small + ResNet-34)",
            "device": str(self.device),
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "load_duration_ms": round(self.load_duration_ms, 2),
            "checkpoint": self.checkpoint_metadata,
            "cuda_memory": cuda_mem,
            "config": {
                "line_crop_size": list(self.config.line_crop_size),
                "avg_char_width_px": self.config.avg_char_width_px,
                "use_physical_budget_processor": getattr(self.config, "use_physical_budget_processor", True),
                "budget_margin_buffer": getattr(self.config, "budget_margin_buffer", 2.0),
            },
        }

    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Loads and resizes document image to standard dimensions."""
        if isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Expected image path (str) or numpy.ndarray, got {type(image)}")

        # Resize to standard page dimensions (W, H)
        target_w, target_h = self.config.page_resize
        if (img.shape[1], img.shape[0]) != (target_w, target_h):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return img

    def detect_burn_region(
        self,
        image: np.ndarray,
        manual_bbox: Optional[Tuple[int, int, int, int]] = None,
        manual_boundary_x: Optional[int] = None,
    ) -> BurnDetectionResult:
        """Detects burn boundary profile, binary damage mask, and severed text lines."""
        return self.burn_detector.detect(image, manual_bbox=manual_bbox, manual_boundary_x=manual_boundary_x)

    def extract_forensic_crop(
        self,
        image: np.ndarray,
        crop_bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Crops local forensic region, resizes to line_crop_size, and normalizes."""
        h, w = image.shape[:2]
        x0, y0, x1, y1 = crop_bbox
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))

        crop = image[y0:y1, x0:x1]
        target_h, target_w = self.config.line_crop_size
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # BGR -> RGB before ImageNet (RGB) normalisation, as in training
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (resized.astype(np.float32) / 255.0 - self._IMAGENET_MEAN) / self._IMAGENET_STD
        tensor = torch.from_numpy(norm).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        return resized, tensor

    def restore_region(
        self,
        image: np.ndarray,
        region: SeveredLineRegion,
        prefix_text: Optional[str] = None,
        page_context: Optional[str] = None,
        budget: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 4,
        use_physical_grounding: bool = True,
        prev_prefix: Optional[str] = None,
        next_prefix: Optional[str] = None,
        prev2_prefix: Optional[str] = None,
        next2_prefix: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Restores missing text for a single severed text-line region under inference_mode.

        ``prev_prefix`` / ``next_prefix`` are the surviving text of the neighbouring
        lines.  ``page_context`` is accepted for backward compatibility and used as
        ``prev_prefix`` when the latter is not given.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("RestorationEngine model is not loaded. Call load_model() first.")
        if prev_prefix is None and page_context:
            prev_prefix = page_context[-200:]

        start_t = time.time()
        h, w = image.shape[:2]
        warnings_list: List[str] = []

        # 1. Forensic visual crop -- identical geometry/normalisation to training
        raw_crop = crop_line(image, region.y0, region.y1, region.b_x)
        crop_tensor = crop_to_tensor(raw_crop).unsqueeze(0).to(self.device)

        # 2. Surviving prefix.  There is no OCR stage: without a transcript the
        #    model has no linguistic evidence, so we say so instead of inventing one.
        clean_prefix = (prefix_text or "").rstrip("\n")
        if not clean_prefix.strip():
            warnings_list.append(
                "No surviving-prefix transcript supplied for this line; prediction is "
                "conditioned on the image only and is unreliable."
            )

        # 3. Geometric character budget -- same formula as training
        if budget is None:
            est_budget = estimate_budget(
                region.b_x, getattr(region, "ink_x0", 0), w, len(clean_prefix.strip())
            )
        else:
            est_budget = float(budget)
        # Decoding enforces a calibrated *upper* bound: the estimate above is a median-type
        # guess (unknown glyph widths), so ~30% of true endings are longer than it.
        budget_mode = None
        decode_budget = est_budget
        style = getattr(self.config, "prompt_style", "v2")
        prompt_budget = est_budget
        if self.cut_estimator is not None and budget is None:
            with torch.inference_mode():
                cut_off = float(self.cut_estimator(crop_tensor).clamp(0, 120).item())
            est_pred = estimate_budget(region.b_x, getattr(region, "ink_x0", 0), w,
                                       len(clean_prefix.strip()), ghost_cut_px=cut_off)
            decode_budget = upper_budget(est_pred, self.cut_meta)
            budget_mode = "hard"
            if style == "v3e":
                prompt_budget = est_pred  # checkpoint was trained with the predicted-cut length hint
        elif style == "v3e" and budget is None:
            warnings_list.append("v3e checkpoint but no cut estimator: length hint uses the fixed-offset estimate.")
        budget_tensor = torch.tensor([decode_budget], dtype=torch.float32).to(self.device)

        # 4. Prompt -- same template as the checkpoint was trained with
        prompt = build_prompt(
            clean_prefix, prev_prefix or "", next_prefix or "", budget=prompt_budget,
            prev2_prefix=prev2_prefix or "", next2_prefix=next2_prefix or "",
            style=style,
        )
        enc = self.tokenizer(
            prompt,
            max_length=self.config.max_context_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # 4. Multimodal Generation
        max_tokens = max_new_tokens or self.config.max_target_len
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                line_burn_crop=crop_tensor,
                line_budget=budget_tensor,
                max_len=max_tokens,
                num_beams=num_beams,
                use_physical_grounding=use_physical_grounding,
                budget_mode=budget_mode,
            )

            # Ranked alternatives (the fully burnt part has no single answer)
            candidates: List[str] = []
            if top_k and top_k > 1:
                # different token paths can decode to the same string: over-generate, keep distinct
                topk = self.model.generate_topk(input_ids, attention_mask, crop_tensor,
                                                line_budget=budget_tensor, top_k=2 * top_k, max_len=max_tokens,
                                                budget_mode=budget_mode)
                candidates = [self.budget_estimator.evaluate_compliance(
                    generated_text=self.tokenizer.decode(c, skip_special_tokens=False),
                    budget=decode_budget, enable_word_snapping=False).cleaned_text for c in topk[0]]
                candidates = [c for c in dict.fromkeys(candidates) if c.strip()][:top_k]
            # Vision-only reading of the text visible in the crop (CTC head)
            visual_reading = self.model.read_crop(crop_tensor)[0] if getattr(self.model, "reading_head", None) else ""

        latency_ms = (time.time() - start_t) * 1000.0
        raw_decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        # 5. Extract infill text & budget compliance evaluation
        compliance = self.budget_estimator.evaluate_compliance(
            generated_text=raw_decoded,
            budget=decode_budget,
            enable_word_snapping=True,
        )

        return {
            "line_idx": region.line_idx,
            "b_x": region.b_x,
            "y0": region.y0,
            "y1": region.y1,
            "crop_bbox": list(region.crop_bbox),
            "damaged_width_px": region.damaged_width_px,
            "surviving_prefix": clean_prefix,
            "predicted_restoration": compliance.cleaned_text,
            "word_snapped_restoration": compliance.word_snapped_text,
            "was_word_snapped": compliance.was_snapped,
            "raw_generated_tokens": raw_decoded,
            "candidates": candidates,
            "visual_reading": visual_reading,
            "physical_budget": compliance.to_dict(),
            "estimated_missing_chars": round(prompt_budget, 2),
            "upper_char_budget": round(decode_budget, 2),
            "inference_time_ms": round(latency_ms, 2),
            "warnings": warnings_list + (
                ["Budget violation detected: generated text exceeded physical margin bound."]
                if compliance.budget_violation else []
            ),
        }

    def restore_page(
        self,
        image: Union[str, np.ndarray],
        manual_bbox: Optional[Tuple[int, int, int, int]] = None,
        manual_boundary_x: Optional[int] = None,
        page_transcripts: Optional[List[str]] = None,
        max_lines: Optional[int] = None,
        save_overlay_path: Optional[str] = None,
        **inference_kwargs,
    ) -> Dict[str, Any]:
        """Full-page restoration workflow: detection -> line segmentation -> restoration."""
        page_start_t = time.time()
        processed_img = self.preprocess_image(image)
        h, w = processed_img.shape[:2]

        # 1. Detection
        det_result = self.detect_burn_region(
            processed_img,
            manual_bbox=manual_bbox,
            manual_boundary_x=manual_boundary_x,
        )

        lines = det_result.severed_lines
        if max_lines is not None and max_lines > 0:
            lines = lines[:max_lines]

        # 2. Sequential line restoration
        restored_lines: List[Dict[str, Any]] = []
        # page_transcripts[i] is the surviving text of the i-th *detected text line*
        # on the page (all lines, not only severed ones).  Pair by region.line_idx.
        transcripts = page_transcripts or []
        if not transcripts:
            det_result.warnings.append(
                "No surviving text supplied: the model sees only the burn-edge crop, so suggestions "
                "will be poor. Paste the page's surviving text, one printed line per line.")
        elif det_result.text_lines_count and len(transcripts) != det_result.text_lines_count:
            det_result.warnings.append(
                f"Pasted {len(transcripts)} lines but detected {det_result.text_lines_count} text lines; "
                "lines are paired top to bottom, so context may be shifted. Check the line count.")

        def _t(i: int) -> str:
            return transcripts[i] if 0 <= i < len(transcripts) else ""

        for line_reg in lines:
            li = line_reg.line_idx
            res = self.restore_region(
                image=processed_img,
                region=line_reg,
                prefix_text=_t(li) or None,
                prev_prefix=_t(li - 1),
                next_prefix=_t(li + 1),
                prev2_prefix=_t(li - 2),
                next2_prefix=_t(li + 2),
                **inference_kwargs,
            )
            restored_lines.append(res)

        total_latency_ms = (time.time() - page_start_t) * 1000.0

        # 3. Optional diagnostic overlay generation
        overlay_file = None
        if save_overlay_path:
            vis = self.burn_detector.generate_diagnostic_visualization(processed_img, det_result)
            # Add restoration text labels
            for r in restored_lines:
                text_lbl = f"-> {r['predicted_restoration'][:20]}"
                cv2.putText(
                    vis,
                    text_lbl,
                    (r["b_x"] + 8, (r["y0"] + r["y1"]) // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 80, 220),
                    1,
                )
            os.makedirs(os.path.dirname(os.path.abspath(save_overlay_path)), exist_ok=True)
            cv2.imwrite(save_overlay_path, vis)
            overlay_file = save_overlay_path

        return {
            "document_shape": [h, w, 3],
            "burn_detection": det_result.to_dict(),
            "total_lines_severed": len(det_result.severed_lines),
            "restored_lines_count": len(restored_lines),
            "restorations": restored_lines,
            "diagnostic_overlay_path": overlay_file,
            "total_page_inference_ms": round(total_latency_ms, 2),
            "avg_latency_per_line_ms": round(total_latency_ms / max(1, len(restored_lines)), 2),
            "checkpoint_id": self.checkpoint_metadata.get("checkpoint_path") or "none (untrained base model)",
            "model_architecture": f"ForensicT5Restorer ({self.config.text_model_name} + ResNet-34)",
            "limitations_notice": (
                "Research prototype trained on synthetic burns of rendered public-domain text. "
                "Restorations are plausible guesses, not recovered evidence; surviving-prefix "
                "transcripts must be supplied (no OCR stage); budgets are geometric estimates. "
                "See results/line_eval.json for held-out accuracy."
            ),
        }

