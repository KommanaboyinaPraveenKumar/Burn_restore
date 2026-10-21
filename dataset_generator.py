"""
dataset_generator.py
=====================
Synthetic Side-Burn Dataset Generator.

Takes a folder of clean, high-resolution document images (with matching
``.txt`` transcription files, one line of text per physical text line) and
produces training pairs for the restoration model:

    * a corrupted image, burned along the right-hand edge with a jagged,
      non-linear boundary and a "ghost stroke" fade rather than a clean cut;
    * a binary mask / bounding box describing the burned region;
    * the ground-truth text that was clipped off the right side of each
      affected line.

Design notes / simplifying assumptions (documented rather than hidden):
    * We do not have per-character bounding boxes for the source pages, so
      the clipped/kept text split is estimated *proportionally*: the
      fraction of a text line's on-page width that survives the jagged
      boundary is applied to the same fraction of that line's character
      count. This is a standard approximation for synthetic corruption
      datasets when OCR-level box annotations aren't available.
    * Text lines are auto-segmented from the image via a horizontal ink
      projection profile. If the number of detected bands does not match
      the number of transcribed lines, we fall back to a uniform vertical
      split of the page height across the transcribed lines.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config


class SideBurnGenerator:
    """Synthesizes realistic, jagged side-burn document corruption."""

    def __init__(self, config: Config):
        self.config = config

    # ------------------------------------------------------------------ #
    # Line segmentation
    # ------------------------------------------------------------------ #
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Returns a binary image where ink pixels are 255, background is 0."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        # Otsu automatically picks a global threshold; INV so dark ink -> 255.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _segment_lines(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """Segments text rows via a horizontal ink-density projection profile."""
        h, w = binary.shape
        row_density = binary.sum(axis=1) / 255.0 / max(w, 1)
        is_text_row = row_density > self.config.min_ink_row_density

        bands: List[Tuple[int, int]] = []
        start = None
        gap = 0
        for y in range(h):
            if is_text_row[y]:
                if start is None:
                    start = y
                gap = 0
            else:
                if start is not None:
                    gap += 1
                    if gap > self.config.line_gap_merge_px:
                        bands.append((start, y - gap))
                        start = None
                        gap = 0
        if start is not None:
            bands.append((start, h - 1))
        return bands

    def _resolve_line_bands(self, binary: np.ndarray, num_lines: int) -> List[Tuple[int, int]]:
        """Matches detected ink bands to the transcribed line count, with a
        uniform-split fallback if detection disagrees with the transcript."""
        h = binary.shape[0]
        bands = self._segment_lines(binary)
        if len(bands) == num_lines and num_lines > 0:
            return bands
        # Fallback: evenly divide the page height across the known line count.
        if num_lines <= 0:
            return []
        step = h / float(num_lines)
        return [(int(i * step), int((i + 1) * step) - 1) for i in range(num_lines)]

    # ------------------------------------------------------------------ #
    # Jagged burn boundary
    # ------------------------------------------------------------------ #
    def _generate_jagged_boundary(self, height: int, base_x: int, amplitude_px: int) -> np.ndarray:
        """Generates a per-row x-coordinate describing where full char-out
        (0,0,0) begins, using a smoothed random walk for a scorched, organic
        (non-linear) edge instead of a straight vertical cut."""
        amplitude_px = max(amplitude_px, 1)
        steps = np.random.normal(0.0, amplitude_px * 0.12, size=height).astype(np.float32)
        walk = np.cumsum(steps)
        walk -= walk.mean()  # center around zero so base_x stays the anchor

        # Smooth the walk so the jaggedness looks like scorch damage, not noise.
        kernel_size = max(int(height * 0.01) | 1, 5)  # odd kernel
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smoothed = np.convolve(walk, kernel, mode="same")

        boundary = base_x + np.clip(smoothed, -amplitude_px, amplitude_px)
        boundary = np.clip(boundary, 0, None)
        return boundary.astype(np.int32)

    # ------------------------------------------------------------------ #
    # Ghost-stroke fade rendering
    # ------------------------------------------------------------------ #
    def _apply_ghost_fade(self, image: np.ndarray, boundary: np.ndarray) -> np.ndarray:
        """Blends pixels approaching the jagged boundary into a low-contrast
        charcoal "ghost stroke" residue, then to full black beyond it."""
        h, w = image.shape[:2]
        cfg = self.config
        out = image.astype(np.float32)
        charcoal = np.array(cfg.charcoal_color, dtype=np.float32)

        x_idx = np.tile(np.arange(w, dtype=np.float32), (h, 1))          # (H, W)
        boundary_col = boundary.astype(np.float32).reshape(-1, 1)        # (H, 1)
        fade_start = boundary_col - cfg.transition_zone_px               # (H, 1)

        # alpha: 0 before the fade starts, ramps 0->1 across the transition
        # zone (ghost residue), saturates at 1 beyond the boundary (charred).
        alpha = (x_idx - fade_start) / np.maximum(boundary_col - fade_start, 1e-3)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]                       # (H, W, 1)

        # Organic scorch texture noise, confined to the damaged region only.
        noise = np.random.normal(0.0, cfg.scorch_noise_std, size=(h, w, 1)).astype(np.float32)
        in_damage_zone = (x_idx >= fade_start)[..., None]
        charred = out * (1.0 - alpha) + charcoal * alpha
        charred = charred + noise * in_damage_zone

        out = np.where(in_damage_zone, charred, out)
        return np.clip(out, 0, 255).astype(np.uint8)



    # ------------------------------------------------------------------ #
    # Proportional text splitting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _proportional_split(text: str, kept_ratio: float) -> Tuple[str, str]:
        """Splits ``text`` into (surviving, clipped) using the fraction of
        the line's pixel width that remains visible before the burn edge.
        Snaps to the nearest whitespace boundary when possible so the
        clipped ground truth remains word-aligned."""
        kept_ratio = float(np.clip(kept_ratio, 0.0, 1.0))
        n = len(text)
        if n == 0:
            return "", ""
        cut = int(round(n * kept_ratio))

        # Search a small window for a nearby space to avoid mid-word cuts.
        window = 6
        best_cut = cut
        for offset in range(window + 1):
            for candidate in (cut - offset, cut + offset):
                if 0 <= candidate <= n and (candidate == n or text[candidate - 1:candidate] == " "):
                    best_cut = candidate
                    break
            else:
                continue
            break

        kept = text[:best_cut].rstrip()
        clipped = text[best_cut:].lstrip()
        return kept, clipped

    @staticmethod
    def _split_line_calibrated(
        text: str,
        binary_row: np.ndarray,
        burn_x: float,
        page_width: int,
    ) -> Tuple[str, str, float]:
        """Calculates kept_ratio using actual line text ink bounds (x_start, x_end)
        rather than naive whole-page width, eliminating margin offset errors."""
        n = len(text)
        if n == 0:
            return "", "", 0.0

        ink_cols = np.where(binary_row > 0)[0]
        if len(ink_cols) >= 10:
            x_start = float(ink_cols[0])
            x_end = float(ink_cols[-1])
            line_width = max(x_end - x_start, 20.0)
            if burn_x <= x_start:
                kept_ratio = 0.0
            elif burn_x >= x_end:
                kept_ratio = 1.0
            else:
                kept_ratio = (burn_x - x_start) / line_width
        else:
            # Fallback using page width with estimated 12% margins
            margin = 0.12 * page_width
            eff_width = max(page_width - 2 * margin, 1.0)
            kept_ratio = np.clip((burn_x - margin) / eff_width, 0.0, 1.0)

        kept, clipped = SideBurnGenerator._proportional_split(text, kept_ratio)
        return kept, clipped, float(kept_ratio)


    # ------------------------------------------------------------------ #
    # Main entry points
    # ------------------------------------------------------------------ #
    def apply_burn(self, image: np.ndarray, text_lines: List[str]) -> Dict:
        """Applies the full side-burn corruption pipeline to a single page.

        Returns a dict with:
            corrupted_image, burn_mask, burn_bbox, boundary,
            burned_width_profile, healthy_context, target_text
        """
        cfg = self.config
        h, w = image.shape[:2]

        burn_ratio = random.uniform(cfg.burn_min_ratio, cfg.burn_max_ratio)
        base_x = int(w * (1.0 - burn_ratio))
        amplitude_px = int(w * cfg.jagged_amplitude_ratio)
        boundary = self._generate_jagged_boundary(h, base_x, amplitude_px)
        boundary = np.clip(boundary, 0, w - 1)

        binary = self._binarize(image)
        bands = self._resolve_line_bands(binary, len(text_lines))

        healthy_lines: List[str] = []
        clipped_lines: List[str] = []
        for i, text in enumerate(text_lines):
            if i >= len(bands):
                healthy_lines.append(text)
                continue
            y0, y1 = bands[i]
            y1 = max(y1, y0 + 1)
            row_boundary_avg = float(np.mean(boundary[y0:y1])) if y1 > y0 else float(boundary[y0])
            kept_ratio = row_boundary_avg / float(w)
            kept, clipped = self._proportional_split(text, kept_ratio)
            healthy_lines.append(kept)
            if clipped:
                clipped_lines.append(clipped)

        corrupted = self._apply_ghost_fade(image, boundary)

        fade_start = np.clip(boundary - cfg.transition_zone_px, 0, w - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            mask[y, fade_start[y]:] = 255

        min_x = int(np.min(fade_start))
        burn_bbox = (min_x, 0, w - min_x, h)  # (x, y, width, height)
        burned_width_profile = (w - boundary).astype(np.float32)  # px missing per row

        return {
            "corrupted_image": corrupted,
            "burn_mask": mask,
            "burn_bbox": burn_bbox,
            "boundary": boundary,
            "burned_width_profile": burned_width_profile,
            "healthy_context": "\n".join(l for l in healthy_lines if l),
            "target_text": " ".join(clipped_lines),
        }

    def generate_pair(
        self,
        image_path: str,
        text_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Loads a source page + transcript, applies the burn, and optionally
        writes the corrupted image / mask / target text to ``output_dir``."""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.resize(image, self.config.page_resize, interpolation=cv2.INTER_AREA)

        with open(text_path, "r", encoding="utf-8") as f:
            text_lines = [line.rstrip("\n") for line in f if line.strip()]

        result = self.apply_burn(image, text_lines)
        result["image_path"] = image_path
        result["text_path"] = text_path

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_dir, f"{stem}_corrupted.png"), result["corrupted_image"])
            cv2.imwrite(os.path.join(output_dir, f"{stem}_mask.png"), result["burn_mask"])
            with open(os.path.join(output_dir, f"{stem}_target.txt"), "w", encoding="utf-8") as f:
                f.write(result["target_text"])
            meta = {
                "burn_bbox": result["burn_bbox"],
                "healthy_context": result["healthy_context"],
                "target_text": result["target_text"],
            }
            with open(os.path.join(output_dir, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return result

    def generate_dataset(
        self,
        images_dir: str,
        texts_dir: str,
        output_dir: str,
        num_samples: Optional[int] = None,
    ) -> List[Dict]:
        """Batch-generates a full synthetic dataset from a folder of clean
        images paired by filename stem with ``.txt`` transcripts."""
        image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )
        manifest = []
        count = 0
        for image_file in image_files:
            if num_samples is not None and count >= num_samples:
                break
            stem = os.path.splitext(image_file)[0]
            text_path = os.path.join(texts_dir, f"{stem}.txt")
            if not os.path.exists(text_path):
                continue
            image_path = os.path.join(images_dir, image_file)
            try:
                result = self.generate_pair(image_path, text_path, output_dir=output_dir)
            except Exception as exc:  # keep batch generation resilient to bad files
                print(f"[SideBurnGenerator] Skipping {image_file}: {exc}")
                continue
            manifest.append(
                {
                    "stem": stem,
                    "image_path": image_path,
                    "target_text": result["target_text"],
                    "burn_bbox": result["burn_bbox"],
                }
            )
            count += 1

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[SideBurnGenerator] Generated {len(manifest)} samples -> {output_dir}")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic side-burnt document dataset.")
    parser.add_argument("--images", required=True, help="Folder of clean source page images.")
    parser.add_argument("--texts", required=True, help="Folder of matching .txt transcripts.")
    parser.add_argument("--out", required=True, help="Output folder for corrupted samples.")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    generator = SideBurnGenerator(Config())
    generator.generate_dataset(args.images, args.texts, args.out, num_samples=args.num_samples)
