"""
dataset_generator.py
=====================
Advanced Synthetic Side-Burn Dataset Generator & Ground-Truth Alignment Engine.

Transforms clean high-resolution document images (with matching transcripts and
optional character/word bounding-box annotations) into realistic, physically grounded
side-burnt training and evaluation pairs.

Key Enhancements (Phase 1):
  1. Priority-Based Text Alignment:
     - Priority 1: Exact character-level bounding boxes (when available).
     - Priority 2: Word-level bounding boxes with calibrated subdivision.
     - Priority 3: Ink-projection column bounds (_split_line_calibrated),
       eliminating margin-induced label contamination.
     - Priority 4: Graceful uncertain alignment fallback.
  2. Physically Plausible Fire Damage:
     - Organic smoothed random-walk burn boundary with configurable irregularity.
     - Charcoal ghost-stroke fade with localized scorch noise.
     - Configurable corruption profiles (mild, moderate, severe, irregular).
  3. Strict Reproducibility:
     - Deterministic seeded corruption for validation/test splits.
     - Controlled dynamic augmentation for training.
  4. Built-in Synthetic Page & Mini-Dataset Fixture Generator:
     - Allows instant CPU/CI test execution with zero external downloads.
"""

from dataclasses import dataclass, asdict
import argparse
import json
import os
import random
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
from pathlib import Path

_burn_dir = Path(__file__).resolve().parent
if str(_burn_dir) not in sys.path:
    sys.path.insert(0, str(_burn_dir))

try:
    from config import Config
except ImportError:
    from Burn_restore.config import Config


@dataclass
class TextSpanAnnotation:
    """Documented schema for character and word level spatial annotations."""
    text: str
    bbox: Tuple[float, float, float, float]  # [x_min, y_min, x_max, y_max] in image pixel coords
    span_type: str = "word"                  # "word" or "char"
    char_index: Optional[int] = None
    visibility_status: str = "unknown"       # "surviving", "severed", "charred", "uncertain"
    damage_ratio: float = 0.0                # 0.0 = completely intact, 1.0 = completely charred
    confidence: float = 1.0                  # Alignment reliability (1.0 = exact box, 0.7 = ink projection, 0.4 = uniform)
    line_idx: int = 0


class SideBurnGenerator:
    """Synthesizes realistic, jagged side-burn document corruption with calibrated
    ground-truth text-to-image alignment."""

    def __init__(self, config: Config):
        self.config = config

    # ------------------------------------------------------------------ #
    # Line segmentation via ink density
    # ------------------------------------------------------------------ #
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Returns binary image where ink pixels are 255 and background is 0."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
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
        if num_lines <= 0:
            return []
        step = h / float(num_lines)
        return [(int(i * step), int((i + 1) * step) - 1) for i in range(num_lines)]

    # ------------------------------------------------------------------ #
    # Jagged burn boundary with seed control
    # ------------------------------------------------------------------ #
    def _generate_jagged_boundary(
        self,
        height: int,
        base_x: int,
        amplitude_px: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Generates a per-row x-coordinate describing where full char-out (0,0,0) begins,
        using a smoothed random walk for a scorched organic non-linear edge."""
        amplitude_px = max(amplitude_px, 1)
        r = rng if rng is not None else np.random
        intensity = getattr(self.config, "random_walk_intensity", 0.12)
        steps = r.normal(0.0, amplitude_px * intensity, size=height).astype(np.float32)
        walk = np.cumsum(steps)
        walk -= walk.mean()

        kernel_factor = getattr(self.config, "boundary_smoothness", 0.01)
        kernel_size = max(int(height * kernel_factor) | 1, 5)
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smoothed = np.convolve(walk, kernel, mode="same")

        boundary = base_x + np.clip(smoothed, -amplitude_px, amplitude_px)
        boundary = np.clip(boundary, 0, None)
        return boundary.astype(np.int32)

    # ------------------------------------------------------------------ #
    # Ghost-stroke fade rendering
    # ------------------------------------------------------------------ #
    def _apply_ghost_fade(
        self,
        image: np.ndarray,
        boundary: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Blends pixels approaching the jagged boundary into low-contrast
        charcoal ghost residue, then to full charred color beyond it."""
        h, w = image.shape[:2]
        cfg = self.config
        out = image.astype(np.float32)
        charcoal = np.array(cfg.charcoal_color, dtype=np.float32)

        x_idx = np.tile(np.arange(w, dtype=np.float32), (h, 1))          # (H, W)
        boundary_col = boundary.astype(np.float32).reshape(-1, 1)        # (H, 1)
        fade_start = boundary_col - cfg.transition_zone_px               # (H, 1)

        alpha = (x_idx - fade_start) / np.maximum(boundary_col - fade_start, 1e-3)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]                       # (H, W, 1)

        r = rng if rng is not None else np.random
        noise = r.normal(0.0, cfg.scorch_noise_std, size=(h, w, 1)).astype(np.float32)
        in_damage_zone = (x_idx >= fade_start)[..., None]
        charred = out * (1.0 - alpha) + charcoal * alpha
        charred = charred + noise * in_damage_zone

        out = np.where(in_damage_zone, charred, out)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Priority-based ground truth alignment
    # ------------------------------------------------------------------ #
    @staticmethod
    def _proportional_split(text: str, kept_ratio: float) -> Tuple[str, str]:
        """Splits text into (surviving, clipped) snapping to whitespace boundary."""
        kept_ratio = float(np.clip(kept_ratio, 0.0, 1.0))
        n = len(text)
        if n == 0:
            return "", ""
        cut = int(round(n * kept_ratio))

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
    ) -> Tuple[str, str, float, float]:
        """Priority 3 Fallback: Calculates kept_ratio using actual line text ink bounds
        (x_start, x_end) rather than whole-page width, eliminating margin offset errors.
        Returns:
            kept_text, clipped_text, kept_ratio, alignment_confidence
        """
        n = len(text)
        if n == 0:
            return "", "", 0.0, 1.0

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
            confidence = 0.85
        else:
            # Priority 4: Fallback with estimated margins
            margin = 0.12 * page_width
            eff_width = max(page_width - 2 * margin, 1.0)
            kept_ratio = np.clip((burn_x - margin) / eff_width, 0.0, 1.0)
            confidence = 0.40

        kept, clipped = SideBurnGenerator._proportional_split(text, kept_ratio)
        return kept, clipped, float(kept_ratio), confidence

    @staticmethod
    def _split_line_by_word_boxes(
        words_with_boxes: List[Tuple[str, float, float, float, float]],
        burn_x: float,
    ) -> Tuple[str, str, float, float]:
        """Priority 2: Splits text line using word-level bounding boxes [word, x1, y1, x2, y2].
        Returns:
            kept_text, clipped_text, kept_ratio, alignment_confidence
        """
        if not words_with_boxes:
            return "", "", 0.0, 0.5

        kept_words = []
        clipped_words = []
        for word, x1, y1, x2, y2 in words_with_boxes:
            w_mid = (x1 + x2) / 2.0
            if x2 <= burn_x:
                kept_words.append(word)
            elif x1 >= burn_x:
                clipped_words.append(word)
            else:
                # Word severed at burn boundary
                if w_mid <= burn_x:
                    kept_words.append(word)
                else:
                    clipped_words.append(word)

        kept_str = " ".join(kept_words).strip()
        clipped_str = " ".join(clipped_words).strip()
        total_len = len(kept_str) + len(clipped_str)
        kept_ratio = len(kept_str) / max(total_len, 1) if total_len > 0 else 0.0
        return kept_str, clipped_str, float(kept_ratio), 0.95

    @staticmethod
    def _split_line_by_char_boxes(
        chars_with_boxes: List[Tuple[str, float, float, float, float]],
        burn_x: float,
    ) -> Tuple[str, str, float, float]:
        """Priority 1: Splits text line using exact character-level bounding boxes.
        Returns:
            kept_text, clipped_text, kept_ratio, alignment_confidence
        """
        if not chars_with_boxes:
            return "", "", 0.0, 0.5

        kept_chars = []
        clipped_chars = []
        for ch, x1, y1, x2, y2 in chars_with_boxes:
            ch_mid = (x1 + x2) / 2.0
            if x2 <= burn_x:
                kept_chars.append(ch)
            elif x1 >= burn_x:
                clipped_chars.append(ch)
            else:
                if ch_mid <= burn_x:
                    kept_chars.append(ch)
                else:
                    clipped_chars.append(ch)

        kept_str = "".join(kept_chars).rstrip()
        clipped_str = "".join(clipped_chars).lstrip()
        total_len = len(kept_str) + len(clipped_str)
        kept_ratio = len(kept_str) / max(total_len, 1) if total_len > 0 else 0.0
        return kept_str, clipped_str, float(kept_ratio), 1.0

    # ------------------------------------------------------------------ #
    # Main corruption pipeline
    # ------------------------------------------------------------------ #
    def apply_burn(
        self,
        image: np.ndarray,
        text_lines: List[str],
        seed: Optional[int] = None,
        line_word_boxes: Optional[List[List[Tuple[str, float, float, float, float]]]] = None,
        line_char_boxes: Optional[List[List[Tuple[str, float, float, float, float]]]] = None,
    ) -> Dict:
        """Applies physically calibrated side-burn corruption to a single document page.

        Args:
            image: (H, W, 3) BGR image.
            text_lines: List of clean transcript strings (one per line).
            seed: Optional random seed for 100% deterministic evaluation.
            line_word_boxes: Optional list of word bounding boxes per line.
            line_char_boxes: Optional list of char bounding boxes per line.

        Returns:
            Dict containing corrupted_image, burn_mask, burn_bbox, boundary,
            burned_width_profile, healthy_context, target_text, severed_lines, all_line_splits.
        """
        cfg = self.config
        h, w = image.shape[:2]

        rng = np.random.RandomState(seed) if seed is not None else None
        py_rng = random.Random(seed) if seed is not None else random

        burn_ratio = py_rng.uniform(cfg.burn_min_ratio, cfg.burn_max_ratio)
        base_x = int(w * (1.0 - burn_ratio))
        amplitude_px = int(w * cfg.jagged_amplitude_ratio)
        boundary = self._generate_jagged_boundary(h, base_x, amplitude_px, rng=rng)
        boundary = np.clip(boundary, 0, w - 1)

        binary = self._binarize(image)
        bands = self._resolve_line_bands(binary, len(text_lines))

        healthy_lines: List[str] = []
        clipped_lines: List[str] = []
        severed_line_details: List[Dict] = []
        all_line_splits: List[Dict] = []

        for i, text in enumerate(text_lines):
            # Unicode normalization for consistent character representations
            clean_text = unicodedata.normalize("NFKD", text).strip()
            if i >= len(bands):
                healthy_lines.append(clean_text)
                continue

            y0, y1 = bands[i]
            y1 = max(y1, y0 + 1)
            row_boundary_avg = float(np.mean(boundary[y0:y1])) if y1 > y0 else float(boundary[y0])
            binary_row = binary[y0:y1, :].sum(axis=0) if y1 > y0 else binary[y0, :]

            # Alignment strategy priority
            if line_char_boxes and i < len(line_char_boxes) and line_char_boxes[i]:
                kept, clipped, kept_ratio, conf = self._split_line_by_char_boxes(line_char_boxes[i], row_boundary_avg)
                strategy = "priority_1_char_box"
            elif line_word_boxes and i < len(line_word_boxes) and line_word_boxes[i]:
                kept, clipped, kept_ratio, conf = self._split_line_by_word_boxes(line_word_boxes[i], row_boundary_avg)
                strategy = "priority_2_word_box"
            else:
                kept, clipped, kept_ratio, conf = self._split_line_calibrated(clean_text, binary_row, row_boundary_avg, w)
                strategy = "priority_3_ink_calibrated" if conf > 0.5 else "priority_4_uncertain_fallback"

            # Categorize line damage status
            if not clipped.strip():
                damage_status = "intact"
            elif not kept.strip():
                damage_status = "destroyed"
            else:
                damage_status = "severed"

            split_info = {
                "line_idx": i,
                "prefix": kept.strip(),
                "target": clipped.strip(),
                "y0": y0,
                "y1": y1,
                "b_x": int(row_boundary_avg),
                "kept_ratio": kept_ratio,
                "damage_status": damage_status,
                "alignment_strategy": strategy,
                "confidence": conf,
            }
            all_line_splits.append(split_info)

            if kept.strip():
                healthy_lines.append(kept.strip())
            if clipped.strip():
                clipped_lines.append(clipped.strip())
                if damage_status == "severed":
                    severed_line_details.append(split_info)

        corrupted = self._apply_ghost_fade(image, boundary, rng=rng)

        fade_start = np.clip(boundary - cfg.transition_zone_px, 0, w - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            mask[y, fade_start[y]:] = 255

        min_x = int(np.min(fade_start))
        burn_bbox = (min_x, 0, w - min_x, h)
        burned_width_profile = (w - boundary).astype(np.float32)

        return {
            "corrupted_image": corrupted,
            "burn_mask": mask,
            "burn_bbox": burn_bbox,
            "boundary": boundary,
            "burned_width_profile": burned_width_profile,
            "healthy_context": "\n".join(l for l in healthy_lines if l),
            "target_text": " ".join(clipped_lines),
            "severed_lines": severed_line_details,
            "all_line_splits": all_line_splits,
        }

    def generate_pair(
        self,
        image_path: str,
        text_path: str,
        output_dir: Optional[str] = None,
        annotation_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """Loads source page, transcript, and optional annotations, applies burn, and saves output."""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.resize(image, self.config.page_resize, interpolation=cv2.INTER_AREA)

        with open(text_path, "r", encoding="utf-8") as f:
            text_lines = [line.rstrip("\n") for line in f if line.strip()]

        line_word_boxes = None
        if annotation_path and os.path.isfile(annotation_path):
            with open(annotation_path, "r", encoding="utf-8") as f:
                anno_data = json.load(f)
                line_word_boxes = anno_data.get("line_word_boxes")

        result = self.apply_burn(image, text_lines, seed=seed, line_word_boxes=line_word_boxes)
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
                "severed_lines": result["severed_lines"],
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
        seed: Optional[int] = 42,
    ) -> List[Dict]:
        """Batch-generates a full synthetic dataset with reproducible seeded corruption."""
        image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )
        manifest = []
        count = 0
        for idx, image_file in enumerate(image_files):
            if num_samples is not None and count >= num_samples:
                break
            stem = os.path.splitext(image_file)[0]
            text_path = os.path.join(texts_dir, f"{stem}.txt")
            if not os.path.exists(text_path):
                continue
            image_path = os.path.join(images_dir, image_file)
            sample_seed = (seed + idx * 7919) if seed is not None else None
            try:
                result = self.generate_pair(image_path, text_path, output_dir=output_dir, seed=sample_seed)
            except Exception as exc:
                print(f"[SideBurnGenerator] Skipping {image_file}: {exc}")
                continue
            manifest.append(
                {
                    "stem": stem,
                    "image_path": image_path,
                    "target_text": result["target_text"],
                    "burn_bbox": result["burn_bbox"],
                    "num_severed_lines": len(result["severed_lines"]),
                }
            )
            count += 1

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[SideBurnGenerator] Generated {len(manifest)} samples -> {output_dir}")
        return manifest


# ---------------------------------------------------------------------- #
# Built-in synthetic document generation for testing & CI fixtures
# ---------------------------------------------------------------------- #
def generate_synthetic_clean_page(
    lines: List[str],
    output_image_path: str,
    output_text_path: str,
    output_annotation_path: Optional[str] = None,
    page_size: Tuple[int, int] = (1024, 1024),
    margin_x: int = 120,
    margin_top: int = 100,
    line_spacing: int = 36,
    font_size: int = 22,
) -> Dict:
    """Renders a clean, high-resolution document page with known typography
    and generates exact word bounding-box annotations."""
    w, h = page_size
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("times.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    curr_y = margin_top
    line_word_boxes: List[List[Tuple[str, float, float, float, float]]] = []

    for line_idx, line_str in enumerate(lines):
        words = line_str.split()
        curr_x = margin_x
        row_boxes = []
        for word in words:
            # Measure word bounding box using draw.textbbox
            try:
                bbox = draw.textbbox((curr_x, curr_y), word, font=font)
                w_width = bbox[2] - bbox[0]
                w_height = bbox[3] - bbox[1]
                word_box = (word, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except Exception:
                # Fallback estimation if textbbox unavailable
                w_width = len(word) * (font_size * 0.55)
                w_height = font_size
                word_box = (word, float(curr_x), float(curr_y), float(curr_x + w_width), float(curr_y + w_height))

            draw.text((curr_x, curr_y), word, fill=(15, 15, 15), font=font)
            row_boxes.append(word_box)
            curr_x += w_width + (font_size * 0.35)

        line_word_boxes.append(row_boxes)
        curr_y += line_spacing
        if curr_y > h - 80:
            break

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    img.save(output_image_path)

    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:len(line_word_boxes)]))

    annotations = {
        "page_size": page_size,
        "num_lines": len(line_word_boxes),
        "line_word_boxes": line_word_boxes,
    }

    if output_annotation_path:
        os.makedirs(os.path.dirname(output_annotation_path), exist_ok=True)
        with open(output_annotation_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

    return annotations


def generate_mini_dataset_fixture(
    fixture_dir: str = "tests/fixtures/mini_dataset",
    num_pages: int = 6,
    seed: int = 42,
) -> Dict[str, str]:
    """Creates a self-contained synthetic document test fixture with clean pages,
    matching transcripts, exact word annotations, and a split manifest."""
    random.seed(seed)
    clean_img_dir = os.path.join(fixture_dir, "clean_images")
    clean_txt_dir = os.path.join(fixture_dir, "clean_texts")
    clean_anno_dir = os.path.join(fixture_dir, "annotations")
    os.makedirs(clean_img_dir, exist_ok=True)
    os.makedirs(clean_txt_dir, exist_ok=True)
    os.makedirs(clean_anno_dir, exist_ok=True)

    sample_documents = [
        # Doc 1: Game theory academic paper
        [
            "Value of Information in Noncooperative Games and Strategic Networks",
            "Nils Bertschinger, David H. Wolpert, and Eckehard Olbrich",
            "Abstract. We consider the value of information in finite noncooperative games.",
            "In single-agent decision problems, additional information can never harm the agent.",
            "In multi-agent strategic settings, however, more information can alter equilibrium strategies.",
            "This counter-intuitive phenomenon is closely related to the famous Braess paradox.",
            "We analyze Cournot oligopoly models and demonstrate when information proves harmful.",
            "Let theta denote the private payoff state distributed according to prior distribution P.",
            "Each player observes a noisy signal regarding theta before choosing an action vector.",
            "Finally, these standard deviations quantify the severity of coordination failures.",
        ],
        # Doc 2: Mathematical Proof & Lemma
        [
            "Theorem 4.2 (Asymptotic Convergence of Grounded Projections)",
            "Let H be a separable Hilbert space and let T be a bounded linear operator.",
            "Lemma 5.1. Suppose that the sequence of partial projections converges strongly to unity.",
            "Then the corresponding sequence of spectral approximations satisfies inequality (18).",
            "Proof. By direct application of the Cauchy-Schwarz inequality, we obtain:",
            "|| T x - T_n x || <= || T || * || x - P_n x || + epsilon_n.",
            "Taking the infimum over all compact perturbations yields the desired bound.",
            "Hence the operator residue vanishes asymptotically as n approaches infinity.",
            "Q.E.D. This completes the demonstration of global geometric consistency.",
        ],
        # Doc 3: Chemical & Biomedical Analysis
        [
            "Pharmacokinetic Profiling of Novel Pyrimidine Receptor Antagonists",
            "Department of Pharmaceutical Sciences and Structural Biology",
            "Compound 4b demonstrated exceptional in vitro potency against the kinase domain.",
            "Half-maximal inhibitory concentration (IC50) was determined to be 4.2 nanomolar.",
            "Liquid chromatography mass spectrometry confirmed plasma stability exceeding four hours.",
            "Metabolic degradation yielded two primary hydroxylated metabolites in microsomal assays.",
            "No significant inhibition of cytochrome P450 isoforms was observed below ten micromolar.",
            "These findings establish compound 4b as a viable candidate for preclinical evaluation.",
        ],
        # Doc 4: Legal / Contractual text with large margins
        [
            "NON-DISCLOSURE AND RESTORATION TECHNOLOGY LICENSE AGREEMENT",
            "This Agreement is entered into by and between the Research Institute and Licensee.",
            "1. Confidential Information. All proprietary models, weights, and dataset manifests",
            "shall be preserved with strict confidentiality and protected against unauthorized disclosure.",
            "2. Permitted Use. The framework shall be utilized solely for academic verification.",
            "3. Term and Termination. This license remains in full effect until formally terminated.",
            "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date below.",
        ],
        # Doc 5: Physics & Differential Equations
        [
            "On the Solvability of Non-Linear Boundary Value Problems in Fluid Dynamics",
            "Consider the stationary Navier-Stokes equations governing incompressible viscous flow:",
            "-nu * Delta u + (u . grad) u + grad p = f in Omega",
            "div u = 0 in Omega, with no-slip boundary condition u = 0 on dOmega.",
            "We establish the existence of weak solutions in Sobolev space H1_0(Omega).",
            "The regularity of pressure p follows from the classical de Rham decomposition theorem.",
            "Numerical simulations were performed using quadratic Lagrange finite elements.",
        ],
        # Doc 6: Historical Document Specimen
        [
            "Historical Manuscript Transcription: Folio 14 recto (Ancient Archives)",
            "The restoration of damaged papyri presents profound epigraphic challenges.",
            "Where mechanical abrasion or scorch damage has obliterated the margin glyphs,",
            "linguistic context must be synthesized with residual stroke morphology.",
            "Careful microscopic examination reveals surviving ink filaments under infrared lighting.",
            "Reconstruction proceeds by matching letter spacing against known scribal hands.",
        ],
    ]

    for idx, lines in enumerate(sample_documents[:num_pages]):
        stem = f"sample_page_{idx+1:03d}"
        img_p = os.path.join(clean_img_dir, f"{stem}.png")
        txt_p = os.path.join(clean_txt_dir, f"{stem}.txt")
        anno_p = os.path.join(clean_anno_dir, f"{stem}_anno.json")
        margin = 120 if idx % 2 == 0 else 180
        generate_synthetic_clean_page(
            lines, img_p, txt_p, anno_p, page_size=(1024, 1024), margin_x=margin, font_size=21
        )

    return {
        "clean_images": clean_img_dir,
        "clean_texts": clean_txt_dir,
        "annotations": clean_anno_dir,
        "fixture_dir": fixture_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic side-burnt document dataset.")
    parser.add_argument("--images", default=None, help="Folder of clean source images.")
    parser.add_argument("--texts", default=None, help="Folder of clean transcripts.")
    parser.add_argument("--out", default="data/burnt", help="Output directory.")
    parser.add_argument("--profile", default="moderate", choices=["mild", "moderate", "severe", "irregular"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--create_fixture", action="store_true", help="Generate mini synthetic test fixture.")
    args = parser.parse_args()

    cfg = Config()
    cfg.set_burn_profile(args.profile)
    gen = SideBurnGenerator(cfg)

    if args.create_fixture:
        paths = generate_mini_dataset_fixture()
        print(f"Generated mini dataset fixture: {paths['fixture_dir']}")
    elif args.images and args.texts:
        gen.generate_dataset(args.images, args.texts, args.out, num_samples=args.num_samples)
    else:
        print("Please provide --images and --texts, or run with --create_fixture.")
