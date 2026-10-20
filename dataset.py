"""
dataset.py
==========
Refactored Line-Grounded Dataset with Full-Page Context and High-Res Forensic Crops.
Pairs each severed line ending with:
  1. Full surviving document context (topic, domain, vocabulary).
  2. Surviving line prefix (immediate sentence context).
  3. High-resolution local burn crop (where character glyphs are 35px tall
     and ink ghost strokes are physically resolved).
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import Config
from dataset_generator import SideBurnGenerator

# ImageNet normalization stats
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LineGroundedDocumentDataset(Dataset):
    """Extracts line-level text restoration instances paired with high-resolution
    forensic burn-edge crops and full surviving page context."""

    def __init__(
        self,
        config: Config,
        image_dir: str,
        text_dir: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        is_eval: bool = False,
    ):
        self.config = config
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.generator = SideBurnGenerator(config)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.text_model_name)
        self.is_eval = is_eval

        self.pairs: List[Tuple[str, str]] = []
        text_files_map = {}
        for root, _, files in os.walk(text_dir):
            for fname in files:
                if fname.lower().endswith(".txt"):
                    stem = os.path.splitext(fname)[0]
                    text_files_map[stem] = os.path.join(root, fname)

        for root, _, files in os.walk(image_dir):
            for fname in files:
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    continue
                stem = os.path.splitext(fname)[0]
                text_path = text_files_map.get(stem)
                if text_path:
                    self.pairs.append((os.path.join(root, fname), text_path))

        if not self.pairs:
            raise ValueError(f"No matching pairs found between '{image_dir}' and '{text_dir}'.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _prepare_line_crop(
        self,
        corrupted_image: np.ndarray,
        y0: int,
        y1: int,
        b_x: int,
    ) -> torch.Tensor:
        """Extracts a focused high-resolution crop around the severed line's burn edge."""
        h, w = corrupted_image.shape[:2]
        pad_y = max(12, (y1 - y0) // 2)
        cy0 = max(0, y0 - pad_y)
        cy1 = min(h, y1 + pad_y)

        # 120px to the left of burn boundary (surviving context), 120px into burn
        cx0 = max(0, b_x - 120)
        cx1 = min(w, b_x + 120)

        crop = corrupted_image[cy0:cy1, cx0:cx1]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((128, 256, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (256, 128), interpolation=cv2.INTER_AREA)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop_rgb = (crop_rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).contiguous().float()
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, text_path = self.pairs[idx]

        with open(text_path, "r", encoding="utf-8") as f:
            text_lines = [line.rstrip("\n") for line in f if line.strip()]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        else:
            image = cv2.resize(image, self.config.page_resize, interpolation=cv2.INTER_AREA)

        # Dynamically apply the side burn
        res = self.generator.apply_burn(image, text_lines)
        boundary = res["boundary"]
        corrupted = res["corrupted_image"]
        h, w = corrupted.shape[:2]

        binary = self.generator._binarize(image)
        bands = self.generator._resolve_line_bands(binary, len(text_lines))

        # Collect all lines that were severed along the jagged burn border
        severed_candidates = []
        healthy_lines = []
        for i, text in enumerate(text_lines):
            if i >= len(bands):
                healthy_lines.append(text)
                continue
            y0, y1 = bands[i]
            row_boundary_avg = float(np.mean(boundary[y0:y1])) if y1 > y0 else float(boundary[y0])
            binary_row = binary[y0:y1, :].sum(axis=0) if y1 > y0 else binary[y0, :]
            kept, clipped, kept_ratio = self.generator._split_line_calibrated(text, binary_row, row_boundary_avg, w)

            if kept.strip():
                healthy_lines.append(kept)
            if clipped.strip():
                severed_candidates.append({
                    "line_idx": i,
                    "prefix": kept.strip(),
                    "target": clipped.strip(),
                    "y0": y0,
                    "y1": y1,
                    "b_x": int(row_boundary_avg),
                })

        # If no severed candidate found, fallback to first line
        if not severed_candidates:
            target_cand = {
                "line_idx": 0,
                "prefix": text_lines[0][:20] if text_lines else "the",
                "target": text_lines[0][20:] if (text_lines and len(text_lines[0]) > 20) else "text",
                "y0": 0,
                "y1": 40,
                "b_x": int(w * 0.75),
            }
        else:
            # For training, pick a random severed line; for eval, pick the first
            target_cand = severed_candidates[0] if self.is_eval else random.choice(severed_candidates)

        # Prepare global page context (excluding the active line)
        page_context_text = " ".join(healthy_lines[:15])[:300]
        line_prefix = target_cand["prefix"]
        target_text = target_cand["target"]

        # Structured prompt for T5: provides both full page context and immediate line prefix
        prompt_text = f"restore line | page: {page_context_text} | prefix: {line_prefix}"

        # Tokenize prompt and target
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.config.max_context_len,
            padding="max_length",
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.config.max_target_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract focused high-resolution line crop (128x256)
        line_crop = self._prepare_line_crop(
            corrupted, target_cand["y0"], target_cand["y1"], target_cand["b_x"]
        )

        # Missing width budget for this specific line
        missing_px = float(w - target_cand["b_x"])
        line_budget = torch.tensor(missing_px / self.config.avg_char_width_px, dtype=torch.float32)

        target_ids = target_enc["input_ids"].squeeze(0)
        # In T5, replace pad token IDs with -100 so CrossEntropy ignores them
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": prompt_enc["input_ids"].squeeze(0),
            "attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "target_input_ids": target_ids,
            "line_burn_crop": line_crop,
            "line_budget": line_budget,
            "raw_prefix": line_prefix,
            "raw_target_text": target_text,
            # Backward-compatibility aliases
            "context_input_ids": prompt_enc["input_ids"].squeeze(0),
            "context_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "burn_crop_image": line_crop,
            "burn_mask_profile": torch.zeros(self.config.profile_bins, dtype=torch.float32),
        }
        return item

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        collated: Dict = {}
        tensor_keys = [k for k in batch[0].keys() if isinstance(batch[0][k], torch.Tensor)]
        for k in tensor_keys:
            collated[k] = torch.stack([b[k] for b in batch], dim=0)
        for k in batch[0].keys():
            if k not in tensor_keys:
                collated[k] = [b[k] for b in batch]
        return collated


# Backward-compatible alias
SideBurnDocumentDataset = LineGroundedDocumentDataset
