"""
inference/burn_detector.py
==========================
Automatic page-level burn boundary detection and damaged text-line segmentation.

Provides:
  - Computer-vision pipeline using OpenCV for identifying side-burnt regions (right edge).
  - Jagged boundary curve estimation b_x(y) across all vertical scanlines.
  - Binary damage mask generation (0 = intact, 255 = damaged/scorched).
  - Text-line detection on surviving document body using horizontal projection profiling.
  - Per-line severed damage bounding-box calculation.
  - Manual bounding-box / boundary-override fallback when automatic detection is uncertain.
  - Diagnostic visual overlay rendering showing boundaries, masks, and line boxes.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SeveredLineRegion:
    """Represents a single text line severed by burn damage."""
    line_idx: int
    y0: int
    y1: int
    b_x: int  # x-coordinate where the burn boundary severs this line
    crop_bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) for forensic model crop
    damaged_width_px: int
    intact_width_px: int
    line_height_px: int
    ink_x0: int = 0  # leftmost ink column of the line (left text margin)


@dataclass
class BurnDetectionResult:
    """Structured container for document burn detection outputs."""
    image_shape: Tuple[int, int, int]  # (H, W, C)
    burn_bbox: Tuple[int, int, int, int]  # Overall (x1, y1, x2, y2)
    boundary_profile: List[int]  # Array of length H with b_x at each y
    damage_mask: np.ndarray  # (H, W) uint8 binary mask (255 = damaged)
    severed_lines: List[SeveredLineRegion] = field(default_factory=list)
    is_manual: bool = False
    detection_confidence: float = 0.0  # Heuristic confidence in [0, 1]
    confidence_method: str = "heuristic_scorch_density"
    mean_burn_depth_px: float = 0.0
    max_burn_depth_px: float = 0.0
    warnings: List[str] = field(default_factory=list)
    text_lines_count: int = 0  # all detected text lines (severed or not), top to bottom

    def to_dict(self) -> Dict[str, Any]:
        """Serializes detection result to JSON-friendly dictionary."""
        return {
            "image_height": self.image_shape[0],
            "image_width": self.image_shape[1],
            "burn_bbox": list(self.burn_bbox),
            "is_manual": self.is_manual,
            "detection_confidence": round(self.detection_confidence, 4),
            "confidence_method": self.confidence_method,
            "mean_burn_depth_px": round(self.mean_burn_depth_px, 2),
            "max_burn_depth_px": round(self.max_burn_depth_px, 2),
            "severed_lines_count": len(self.severed_lines),
            "text_lines_count": self.text_lines_count,
            "severed_lines": [
                {
                    "line_idx": l.line_idx,
                    "y0": l.y0,
                    "y1": l.y1,
                    "b_x": l.b_x,
                    "crop_bbox": list(l.crop_bbox),
                    "damaged_width_px": l.damaged_width_px,
                    "intact_width_px": l.intact_width_px,
                    "line_height_px": l.line_height_px,
                }
                for l in self.severed_lines
            ],
            "warnings": self.warnings,
        }


class BurnDetector:
    """OpenCV-based automatic detector for document margin fire damage.
    
    Identifies dark charred edges, scorched transition zones, and page cut-outs
    along the right edge of document pages. Supports manual bounding box fallback.
    """

    def __init__(
        self,
        charcoal_threshold: int = 70,
        edge_search_ratio: float = 0.50,
        min_ink_row_density: float = 0.015,
        line_gap_merge_px: int = 6,
        crop_context_px: int = 180,
        char_level: float = 0.6,
        smooth_w: int = 31,
        smooth_h: int = 9,
    ):
        self.char_level = char_level
        self.smooth_w = smooth_w
        self.smooth_h = smooth_h
        self.charcoal_threshold = charcoal_threshold
        self.edge_search_ratio = edge_search_ratio
        self.min_ink_row_density = min_ink_row_density
        self.line_gap_merge_px = line_gap_merge_px
        self.crop_context_px = crop_context_px

    def detect(
        self,
        image: np.ndarray,
        manual_bbox: Optional[Tuple[int, int, int, int]] = None,
        manual_boundary_x: Optional[int] = None,
    ) -> BurnDetectionResult:
        """Detects burn boundary profile, binary damage mask, and severed text lines.
        
        Args:
            image: BGR or grayscale document image (H, W, C) or (H, W).
            manual_bbox: Optional user-provided (x1, y1, x2, y2) to override auto detection.
            manual_boundary_x: Optional vertical cut coordinate overriding auto detection.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty.")

        if len(image.shape) == 2:
            h, w = image.shape
            gray = image.copy()
            color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            h, w = image.shape[:2]
            color = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        warnings: List[str] = []

        # ------------------------------------------------------------------ #
        # Case A: User Manual Override
        # ------------------------------------------------------------------ #
        if manual_bbox is not None or manual_boundary_x is not None:
            if manual_bbox is not None:
                x1, y1, x2, y2 = manual_bbox
                x1 = max(0, min(w - 1, int(x1)))
                y1 = max(0, min(h - 1, int(y1)))
                x2 = max(x1 + 1, min(w, int(x2)))
                y2 = max(y1 + 1, min(h, int(y2)))
                burn_bbox = (x1, y1, x2, y2)
                boundary_profile = [x1 if y1 <= y < y2 else w for y in range(h)]
            else:
                bx = max(0, min(w - 1, int(manual_boundary_x)))
                burn_bbox = (bx, 0, w, h)
                boundary_profile = [bx] * h

            damage_mask = np.zeros((h, w), dtype=np.uint8)
            for y in range(h):
                bx = boundary_profile[y]
                if bx < w:
                    damage_mask[y, bx:w] = 255

            severed_lines = self._detect_text_lines_and_intersect(
                gray, boundary_profile, h, w
            )
            depths = [w - boundary_profile[y] for y in range(h) if boundary_profile[y] < w]
            mean_depth = float(np.mean(depths)) if depths else 0.0
            max_depth = float(np.max(depths)) if depths else 0.0

            return BurnDetectionResult(
                image_shape=(h, w, 3),
                burn_bbox=burn_bbox,
                boundary_profile=boundary_profile,
                damage_mask=damage_mask,
                severed_lines=severed_lines,
                text_lines_count=self.last_text_lines_count,
                is_manual=True,
                detection_confidence=1.0,
                confidence_method="manual_user_specification",
                mean_burn_depth_px=mean_depth,
                max_burn_depth_px=max_depth,
                warnings=["Manual bounding box override applied by user."],
            )

        # ------------------------------------------------------------------ #
        # Case B: Automatic Computer Vision Detection (Right-Edge Burn)
        # ------------------------------------------------------------------ #
        # 1-5. Burn front = how far the *continuous* char layer reaches in from the
        # right edge. A plain "darker than X" test also fires on black text ink, which
        # put the front at the first letter past mid-page; instead darkness is
        # normalised between paper and char colour and smoothed over a window wider
        # than a glyph stroke, so isolated text strokes stay well below the level.
        search_start_x = int(w * (1.0 - self.edge_search_ratio))
        g = gray.astype(np.float32)
        paper = float(np.percentile(g[:, : max(1, int(w * 0.4))], 80))
        char = float(np.median(g[:, max(0, w - max(4, w // 50)):]))
        boundary_profile = [w] * h
        detected_burn_rows = 0
        fade_offsets: List[int] = []   # b_x - (mid-fade column), per detected row
        total_charred_px = 0
        if paper - char >= 40:
            raw = np.clip((paper - g) / (paper - char), 0.0, 1.0)
            dark = cv2.blur(raw, (self.smooth_w, self.smooth_h))
            fine = cv2.blur(raw, (5, self.smooth_h))
            charred = dark >= self.char_level
            charred[:, :search_start_x] = False
            total_charred_px = int(charred.sum())
            # length of the all-charred run that ends at the right edge, per row
            not_charred = ~charred[:, ::-1]
            run = np.where(not_charred.any(axis=1), not_charred.argmax(axis=1), w)
            for y in range(h):
                if run[y] > 0:
                    b0 = int(w - run[y])
                    # The run starts mid-fade; b_x (as in training) is where the char
                    # layer becomes opaque: first column reaching 95% of the row's plateau.
                    seg = fine[y, b0:]
                    plateau = float(np.median(fine[y, min(w - 1, b0 + 40):min(w, b0 + 140)])) \
                        if b0 + 60 < w else float(np.median(seg))
                    hit = np.where(seg >= 0.95 * plateau)[0]
                    off = int(hit[0]) if len(hit) else 0
                    boundary_profile[y] = b0 + off
                    fade_offsets.append(off)
                    detected_burn_rows += 1

        # 6. Smooth the boundary curve to eliminate single-pixel noise spikes
        boundary_arr = np.array(boundary_profile, dtype=np.float32)
        valid_mask = boundary_arr < w
        if np.sum(valid_mask) > (h * 0.05):
            # Median-filter valid boundary points
            smoothed = boundary_arr.copy()
            win = 15
            for i in range(h):
                i0 = max(0, i - win // 2)
                i1 = min(h, i + win // 2 + 1)
                sub = boundary_arr[i0:i1]
                sub_v = sub[sub < w]
                if len(sub_v) > 0:
                    smoothed[i] = np.median(sub_v)
            boundary_profile = [int(np.clip(smoothed[y], 0, w)) for y in range(h)]
        else:
            # Fallback: No prominent dark charcoal detected.
            # Document may be clipped without dark residue or uniform cut.
            warnings.append(
                "Low charcoal density detected on right margin; falling back to conservative margin estimate."
            )
            fallback_cut_x = int(w * 0.82)
            boundary_profile = [fallback_cut_x] * h

        # 7. Construct 2D binary damage mask (255 = damaged, 0 = intact)
        damage_mask = np.zeros((h, w), dtype=np.uint8)
        min_bx = w
        max_bx = 0
        min_by = h
        max_by = 0

        for y in range(h):
            bx = boundary_profile[y]
            if bx < w:
                damage_mask[y, bx:w] = 255
                min_bx = min(min_bx, bx)
                max_bx = max(max_bx, w)
                min_by = min(min_by, y)
                max_by = max(max_by, y)

        if min_bx >= w:
            min_bx = int(w * 0.80)
            max_bx = w
            min_by = 0
            max_by = h

        burn_bbox = (min_bx, min_by, max_bx, max_by)

        # 8. Detect text lines and compute per-line severance
        # Line segmentation must ignore the faded fringe left of b_x too, or the
        # ghost strokes join all rows into one band: mask from before the fade.
        fade = int(np.median(fade_offsets)) if fade_offsets else 0
        severed_lines = self._detect_text_lines_and_intersect(
            gray, boundary_profile, h, w, ink_mask_shift=fade + 10
        )

        # 9. Compute heuristic confidence score
        damage_area = int(np.sum(damage_mask > 0))
        total_pixels = h * w
        burn_fraction = damage_area / max(1, total_pixels)

        # Heuristic confidence based on:
        # 1. Reasonable burn fraction (3% to 40% of page)
        # 2. Proportion of detected rows
        # 3. Presence of charred dark pixels
        row_coverage = detected_burn_rows / max(1, h)
        charred_ratio = total_charred_px / max(1, damage_area) if damage_area > 0 else 0.0

        confidence = 0.5 * min(1.0, row_coverage * 1.5) + 0.3 * min(1.0, charred_ratio * 4.0) + 0.2 * (1.0 if (0.02 <= burn_fraction <= 0.45) else 0.4)
        confidence = float(np.clip(confidence, 0.05, 0.95))

        depths = [w - boundary_profile[y] for y in range(h) if boundary_profile[y] < w]
        mean_depth = float(np.mean(depths)) if depths else 0.0
        max_depth = float(np.max(depths)) if depths else 0.0

        if len(severed_lines) == 0:
            warnings.append("No severed text lines identified; check image text density.")

        return BurnDetectionResult(
            image_shape=(h, w, 3),
            burn_bbox=burn_bbox,
            boundary_profile=boundary_profile,
            damage_mask=damage_mask,
            severed_lines=severed_lines,
            text_lines_count=self.last_text_lines_count,
            is_manual=False,
            detection_confidence=round(confidence, 4),
            confidence_method="heuristic_scorch_density",
            mean_burn_depth_px=mean_depth,
            max_burn_depth_px=max_depth,
            warnings=warnings,
        )

    @staticmethod
    def _clean_bands(bands: List[Tuple[int, int]], ink: np.ndarray, density: np.ndarray,
                     w: int) -> List[Tuple[int, int]]:
        """Makes bands correspond to printed lines (so pasted transcripts line up):
        drops bands that do not start near the left text margin (burn texture at the
        page edge) and splits bands about twice the usual height at the emptiest row
        (two lines joined by ascenders/descenders)."""
        if len(bands) < 3:
            return bands
        starts = []
        for y0, y1 in bands:
            cols = np.where(ink[y0:y1].any(axis=0))[0]
            starts.append(int(cols[0]) if len(cols) else w)
        margin = float(np.median(starts))
        bands = [b for b, x in zip(bands, starts) if x <= margin + 0.2 * w]
        if len(bands) < 3:
            return bands
        centres = np.array([(y0 + y1) / 2 for y0, y1 in bands])
        pitch = float(np.median(np.diff(centres)))   # baseline-to-baseline distance
        med_h = float(np.median([y1 - y0 for y0, y1 in bands]))
        out: List[Tuple[int, int]] = []
        for y0, y1 in bands:
            k = 1 + int(round((y1 - y0 - med_h) / pitch))
            if k >= 2 and (y1 - y0) >= 1.3 * pitch:
                cuts = [y0]
                for j in range(1, k):
                    c = y0 + int(j * (y1 - y0) / k)
                    lo, hi = max(y0 + 1, c - int(med_h // 2)), min(y1 - 1, c + int(med_h // 2))
                    cuts.append(lo + int(np.argmin(density[lo:hi])) if hi > lo else c)
                cuts.append(y1)
                out.extend((a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 2)
            else:
                out.append((y0, y1))
        # Fill gaps of whole line pitches (very short lines, e.g. a one-word paragraph
        # end, can fall below the ink-density threshold) with placeholder bands so the
        # band index still equals the printed line index.
        filled: List[Tuple[int, int]] = [out[0]]
        for y0, y1 in out[1:]:
            p0, p1 = filled[-1]
            k = int(round(((y0 + y1) / 2 - (p0 + p1) / 2) / pitch))
            for j in range(1, k):
                c = int((p0 + p1) / 2 + j * pitch)
                filled.append((c - int(med_h // 2), c + int(med_h // 2)))
            filled.append((y0, y1))
        return filled

    def _detect_text_lines_and_intersect(
        self,
        gray: np.ndarray,
        boundary_profile: List[int],
        h: int,
        w: int,
        ink_mask_shift: int = 0,
    ) -> List[SeveredLineRegion]:
        """Detects horizontal text-line row bands and computes severed burn intersection."""
        # Binarize intact page region to find ink pixels
        _, binary_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Mask out the damaged right region when computing ink density
        intact_ink = binary_ink.copy()
        for y in range(h):
            bx = boundary_profile[y]
            if bx < w:
                intact_ink[y, max(0, bx - ink_mask_shift):w] = 0

        # Horizontal projection profile
        row_ink_density = np.sum(intact_ink > 0, axis=1) / float(w)

        # Find continuous row bands with ink
        in_line = False
        start_y = 0
        raw_bands: List[Tuple[int, int]] = []

        for y in range(h):
            if row_ink_density[y] >= self.min_ink_row_density:
                if not in_line:
                    in_line = True
                    start_y = y
            else:
                if in_line:
                    in_line = False
                    if (y - start_y) >= 2:  # Minimum text line height (2 px)
                        raw_bands.append((start_y, y))

        if in_line and (h - start_y) >= 2:
            raw_bands.append((start_y, h))

        # Merge bands separated by less than line_gap_merge_px
        merged_bands: List[Tuple[int, int]] = []
        for band in raw_bands:
            if not merged_bands:
                merged_bands.append(band)
            else:
                prev_y0, prev_y1 = merged_bands[-1]
                curr_y0, curr_y1 = band
                if (curr_y0 - prev_y1) <= self.line_gap_merge_px:
                    merged_bands[-1] = (prev_y0, curr_y1)
                else:
                    merged_bands.append(band)

        merged_bands = self._clean_bands(merged_bands, intact_ink, row_ink_density, w)
        self.last_text_lines_count = len(merged_bands)

        # Intersect text lines with the burn boundary profile
        severed_lines: List[SeveredLineRegion] = []
        for idx, (y0, y1) in enumerate(merged_bands):
            line_profile = boundary_profile[y0:y1]
            if not line_profile:
                continue
            # Severance point is median boundary along the line height
            line_bx = int(np.median(line_profile))
            line_bx = max(0, min(w, line_bx))
            ink_cols = np.where(intact_ink[y0:y1].any(axis=0))[0]
            ink_x0 = int(ink_cols[0]) if len(ink_cols) else 0

            # Only consider a line severed if it is cut before the right margin
            # A line whose ink stops well before the fade (e.g. a short paragraph end)
            # was not cut by the burn.
            reaches_burn = len(ink_cols) > 0 and int(ink_cols[-1]) >= line_bx - ink_mask_shift - 40
            if line_bx < w and reaches_burn:
                # Compute forensic crop bounding box:
                # Include crop_context_px of intact text to the left
                crop_x0 = max(0, line_bx - self.crop_context_px)
                crop_x1 = w
                # Add vertical padding of 20%
                pad_y = max(2, int((y1 - y0) * 0.20))
                crop_y0 = max(0, y0 - pad_y)
                crop_y1 = min(h, y1 + pad_y)

                severed_lines.append(
                    SeveredLineRegion(
                        line_idx=idx,
                        y0=y0,
                        y1=y1,
                        b_x=line_bx,
                        crop_bbox=(crop_x0, crop_y0, crop_x1, crop_y1),
                        damaged_width_px=w - line_bx,
                        intact_width_px=line_bx,
                        line_height_px=y1 - y0,
                        ink_x0=ink_x0,
                    )
                )

        return severed_lines

    def generate_diagnostic_visualization(
        self,
        image: np.ndarray,
        detection_result: BurnDetectionResult,
    ) -> np.ndarray:
        """Renders an annotated visualization showing boundaries, masks, and severed lines.
        
        Returns an RGB image (H, W, 3).
        """
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        h, w = annotated.shape[:2]

        # 1. Tint damaged region with translucent blue overlay
        overlay = annotated.copy()
        mask = detection_result.damage_mask > 0
        overlay[mask] = (overlay[mask] * 0.5 + np.array([220, 100, 50]) * 0.5).astype(np.uint8)
        annotated = cv2.addWeighted(annotated, 0.65, overlay, 0.35, 0)

        # 2. Draw jagged burn boundary curve in vibrant red
        for y in range(h - 1):
            x1 = detection_result.boundary_profile[y]
            x2 = detection_result.boundary_profile[y + 1]
            if x1 < w and x2 < w:
                cv2.line(annotated, (x1, y), (x2, y + 1), (0, 0, 255), 2)

        # 3. Draw detected severed text line boxes in green
        for s in detection_result.severed_lines:
            cv2.rectangle(
                annotated,
                (0, s.y0),
                (s.b_x, s.y1),
                (0, 200, 0),
                1,
            )
            # Mark the severance cut point
            cv2.circle(annotated, (s.b_x, (s.y0 + s.y1) // 2), 3, (0, 0, 255), -1)

        # 4. Header status banner
        cv2.rectangle(annotated, (0, 0), (w, 36), (30, 30, 30), -1)
        mode_str = "MANUAL" if detection_result.is_manual else f"AUTO (Conf: {detection_result.detection_confidence:.2f})"
        info_str = f"Burn Detection [{mode_str}] | Severed Lines: {len(detection_result.severed_lines)} | Mean Cut Depth: {detection_result.mean_burn_depth_px:.1f}px"
        cv2.putText(annotated, info_str, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return annotated

