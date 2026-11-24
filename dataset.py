"""
dataset.py
==========
Refactored Line-Grounded Dataset with Calibrated Text-to-Image Alignment,
Deterministic Evaluation, and Zero-Data-Leakage Splitting.

Key Enhancements (Phase 1):
  1. Priority-Based Alignment Integration:
     - Leverages SideBurnGenerator's calibrated ink-bound and bounding-box splitting.
     - Eliminates margin offset errors and false target text contamination.
  2. Strict Determinism for Evaluation:
     - In 'val' and 'test' modes, corruption is fully deterministic and seeded per sample.
     - In 'train' mode, benefits from dynamic augmentation with controlled epoch seeding.
  3. Page-Level Splitting & Zero-Leakage:
     - Supports loading directly from a split manifest ('data/splits.json').
     - Guarantees zero document page overlap across train/val/test splits.
  4. Real Missing-Width Geometric Profile:
     - Computes the true 16-bin missing width profile across the line height.
  5. Backward Compatibility:
     - Maintains all expected tensor keys, shapes, and class aliases.
"""

import hashlib
import json
import os
import random
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_burn_dir = Path(__file__).resolve().parent
if str(_burn_dir) not in sys.path:
    sys.path.insert(0, str(_burn_dir))

try:
    from config import Config
    from dataset_generator import SideBurnGenerator
except ImportError:
    from Burn_restore.config import Config
    from Burn_restore.dataset_generator import SideBurnGenerator

# ImageNet normalization stats
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LineGroundedDocumentDataset(Dataset):
    """Extracts line-level text restoration instances paired with high-resolution
    forensic burn-edge crops, surviving page context, and physical character budgets."""

    def __init__(
        self,
        config: Config,
        image_dir: Optional[str] = None,
        text_dir: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        is_eval: bool = False,
        split: Optional[str] = None,
        splits_manifest: Optional[str] = None,
        annotation_dir: Optional[str] = None,
    ):
        self.config = config
        self.generator = SideBurnGenerator(config)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.text_model_name)
        self.is_eval = is_eval
        self.split = split
        self.annotation_dir = annotation_dir
        self.current_epoch = 0

        self.pairs: List[Tuple[str, str, str]] = []  # (image_path, text_path, stem)

        # 1. Load from splits manifest if specified
        if splits_manifest and os.path.isfile(splits_manifest):
            with open(splits_manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            split_key = split or ("val" if is_eval else "train")
            items = manifest_data.get(split_key, [])
            for item in items:
                self.pairs.append((item["image_path"], item["text_path"], item["stem"]))
        elif split and os.path.isfile("data/splits.json"):
            with open("data/splits.json", "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            items = manifest_data.get(split, [])
            for item in items:
                self.pairs.append((item["image_path"], item["text_path"], item["stem"]))
        else:
            # 2. Backward-compatible directory scanning
            if not image_dir or not text_dir:
                raise ValueError("Must provide either a valid splits_manifest or (image_dir, text_dir).")

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
                    if text_path and os.path.isfile(text_path):
                        self.pairs.append((os.path.join(root, fname), text_path, stem))

        if not self.pairs:
            raise ValueError(
                f"No matching pairs found. image_dir='{image_dir}', text_dir='{text_dir}', split='{split}'"
            )

        # Sort pairs deterministically by stem
        self.pairs.sort(key=lambda x: x[2])

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        """Sets the current training epoch for reproducible dynamic augmentation."""
        self.current_epoch = epoch

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

    def _compute_binned_profile(self, boundary: np.ndarray, y0: int, y1: int, page_width: int) -> torch.Tensor:
        """Computes true binned missing-width profile across line height."""
        bins = self.config.profile_bins
        if y1 <= y0:
            return torch.zeros(bins, dtype=torch.float32)
        missing_px = (page_width - boundary[y0:y1]).astype(np.float32)
        if len(missing_px) == 0:
            return torch.zeros(bins, dtype=torch.float32)
        # Resample into 'bins' uniform bins
        indices = np.linspace(0, len(missing_px) - 1, bins).astype(int)
        profile = missing_px[indices]
        return torch.from_numpy(profile).float()

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        image_path, text_path, stem = self.pairs[idx]

        with open(text_path, "r", encoding="utf-8") as f:
            text_lines = [unicodedata.normalize("NFKD", line.rstrip("\n")).strip() for line in f if line.strip()]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        else:
            image = cv2.resize(image, self.config.page_resize, interpolation=cv2.INTER_AREA)

        h, w = image.shape[:2]

        # Optional word annotation loading for Priority 2 alignment
        line_word_boxes = None
        if self.annotation_dir:
            anno_p = os.path.join(self.annotation_dir, f"{stem}_anno.json")
            if os.path.isfile(anno_p):
                try:
                    with open(anno_p, "r", encoding="utf-8") as af:
                        anno_data = json.load(af)
                        line_word_boxes = anno_data.get("line_word_boxes")
                except Exception:
                    line_word_boxes = None

        # Deterministic seeding for evaluation vs controlled dynamic seeding for training
        if self.is_eval or (self.split in ["val", "test"]):
            stem_hash = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)
            sample_seed = (self.config.seed + idx * 7919 + stem_hash % 10007) & 0x7FFFFFFF
        else:
            # Training: dynamic per epoch
            sample_seed = (self.config.seed + idx * 7919 + self.current_epoch * 104729) & 0x7FFFFFFF

        # Apply burn corruption with calibrated alignment
        res = self.generator.apply_burn(
            image,
            text_lines,
            seed=sample_seed,
            line_word_boxes=line_word_boxes,
        )

        boundary = res["boundary"]
        corrupted = res["corrupted_image"]
        severed_candidates = res["severed_lines"]

        # Selection of active restoration target line
        if not severed_candidates:
            target_cand = {
                "line_idx": 0,
                "prefix": text_lines[0][:20] if text_lines else "the",
                "target": text_lines[0][20:] if (text_lines and len(text_lines[0]) > 20) else "text",
                "y0": 0,
                "y1": 40,
                "b_x": int(w * 0.75),
                "damage_status": "severed",
                "confidence": 0.50,
            }
        else:
            # Deterministic selection for evaluation (first severed line), random choice for training
            if self.is_eval or (self.split in ["val", "test"]):
                target_cand = severed_candidates[0]
            else:
                py_rng = random.Random(sample_seed)
                target_cand = py_rng.choice(severed_candidates)

        # Linguistic context: surviving lines (excluding active line)
        healthy_lines = [s["prefix"] for s in res["all_line_splits"] if s["prefix"] and s["line_idx"] != target_cand["line_idx"]]
        page_context_text = " ".join(healthy_lines[:15])[:300]
        line_prefix = target_cand["prefix"]
        target_text = target_cand["target"]

        # T5 Native Infilling syntax
        prompt_text = f"Context: {page_context_text} | Incomplete line: {line_prefix} <extra_id_0>"
        infill_target_text = f"<extra_id_0> {target_text} <extra_id_1>"

        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.config.max_context_len,
            padding="max_length",
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            infill_target_text,
            truncation=True,
            max_length=self.config.max_target_len,
            padding="max_length",
            return_tensors="pt",
        )

        line_crop = self._prepare_line_crop(
            corrupted, target_cand["y0"], target_cand["y1"], target_cand["b_x"]
        )

        missing_px = float(w - target_cand["b_x"])
        line_budget = torch.tensor(missing_px / self.config.avg_char_width_px, dtype=torch.float32)

        target_ids = target_enc["input_ids"].squeeze(0)
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        binned_profile = self._compute_binned_profile(boundary, target_cand["y0"], target_cand["y1"], w)

        item = {
            "input_ids": prompt_enc["input_ids"].squeeze(0),
            "attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "target_input_ids": target_ids,
            "line_burn_crop": line_crop,
            "line_budget": line_budget,
            "raw_prefix": line_prefix,
            "raw_target_text": target_text,
            "stem": stem,
            "line_idx": target_cand["line_idx"],
            "damage_status": target_cand.get("damage_status", "severed"),
            "alignment_confidence": torch.tensor(float(target_cand.get("confidence", 0.85)), dtype=torch.float32),
            # Backward-compatibility aliases
            "context_input_ids": prompt_enc["input_ids"].squeeze(0),
            "context_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "burn_crop_image": line_crop,
            "burn_mask_profile": binned_profile,
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


# --------------------------------------------------------------------------- #
# Line-level cached dataset (one example per severed line)
# --------------------------------------------------------------------------- #
class LineCacheDataset(Dataset):
    """Reads the cache produced by ``scripts/build_line_dataset.py``.

    Every severed line is its own example.  Prompt, crop and budget come from
    ``line_data`` so they are identical to evaluation and inference.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_context_len: int = 160,
        max_target_len: int = 32,
        augment: bool = False,
        limit: Optional[int] = None,
        seed: int = 42,
        prompt_style: str = "v2",
        context_noise: float = 0.0,
    ):
        """
        prompt_style   "v2" (legacy) or "v3" (2-line context + length hint)
        context_noise  OCR-style error rate applied to the context strings.
                       With ``augment`` it is the *maximum* rate, sampled per
                       example (half the examples stay clean); without
                       ``augment`` it is a fixed, deterministic rate (robustness eval).
        """
        try:
            from line_data import build_target, crop_to_tensor, ocr_noise, prompt_from_record, split_visible
            from model import encode_ctc_text
        except ImportError:
            from Burn_restore.line_data import (build_target, crop_to_tensor, ocr_noise, prompt_from_record,
                                                split_visible)
            from Burn_restore.model import encode_ctc_text
        self._prompt_from_record, self._build_target, self._to_tensor = prompt_from_record, build_target, crop_to_tensor
        self._split_visible, self._ocr_noise, self._encode_ctc = split_visible, ocr_noise, encode_ctc_text
        self.prompt_style = prompt_style
        self.context_noise = context_noise

        split_dir = os.path.join(data_dir, split)
        with open(os.path.join(split_dir, "records.jsonl"), "r", encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f]
        self._crops_path = os.path.join(split_dir, "crops.npy")
        self._crops = None  # opened lazily so DataLoader workers don't pickle the memmap
        assert len(self.records) == len(self.crops), "records / crops length mismatch"
        self.indices = list(range(len(self.records)))
        if limit is not None and limit < len(self.indices):
            rng = random.Random(seed)
            self.indices = sorted(rng.sample(self.indices, limit))
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_target_len = max_target_len
        self.augment = augment

    @property
    def crops(self) -> np.ndarray:
        if self._crops is None:
            self._crops = np.load(self._crops_path, mmap_mode="r")
        return self._crops

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_crops"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        x = crop.astype(np.float32)
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
        x = x + rng.normal(0, rng.uniform(0, 6), x.shape)
        return np.clip(x, 0, 255).astype(np.uint8)

    def __getitem__(self, i: int) -> Dict:
        idx = self.indices[i]
        r = self.records[idx]
        crop = np.asarray(self.crops[idx])
        if self.augment:
            crop = self._augment(crop)
        noise = None
        if self.context_noise > 0:
            if self.augment:
                rng = random.Random()
                if rng.random() < 0.5:
                    p = rng.uniform(0, self.context_noise)
                    noise = lambda t, _r=rng, _p=p: self._ocr_noise(t, _r, _p)  # noqa: E731
            else:
                rng = random.Random(idx)
                noise = lambda t, _r=rng: self._ocr_noise(t, _r, self.context_noise)  # noqa: E731
        prompt = self._prompt_from_record(r, style=self.prompt_style, noise=noise)
        enc = self.tokenizer(prompt, truncation=True, max_length=self.max_context_len)
        ctc = self._encode_ctc(r.get("crop_text", ""))
        tgt = self.tokenizer(self._build_target(r["target"]), truncation=True, max_length=self.max_target_len)
        # Visible ("read") part of the target, used only by the contrastive loss.
        # Older caches without the annotation simply yield no visible ids.
        read_text, _ = self._split_visible(r["target"], r.get("visible_chars", 0))
        vis_ids = (self.tokenizer(read_text, add_special_tokens=False, truncation=True, max_length=8)["input_ids"]
                   if read_text else [])
        return {
            "ctc_targets": torch.tensor(ctc, dtype=torch.long),
            "visible_ids": torch.tensor(vis_ids, dtype=torch.long),
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "labels": torch.tensor(tgt["input_ids"], dtype=torch.long),
            "line_burn_crop": self._to_tensor(crop),
            "line_budget": torch.tensor(float(r["budget"]), dtype=torch.float32),
            "raw_prefix": r["prefix"],
            "raw_target_text": r["target"].strip(),
            "record_idx": idx,
        }

    @staticmethod
    def make_collate(pad_id: int) -> "LineCollator":
        return LineCollator(pad_id)


class LineCollator:
    """Picklable collate function (Windows DataLoader workers use spawn)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict:
        pad_id = self.pad_id
        L = max(len(b["input_ids"]) for b in batch)
        T = max(len(b["labels"]) for b in batch)
        V = max([len(b["visible_ids"]) for b in batch if "visible_ids" in b] + [1])
        C = max([len(b["ctc_targets"]) for b in batch if "ctc_targets" in b] + [1])
        n = len(batch)
        visible_ids = torch.full((n, V), pad_id, dtype=torch.long)
        ctc_targets = torch.zeros((n, C), dtype=torch.long)
        ctc_lengths = torch.zeros((n,), dtype=torch.long)
        input_ids = torch.full((n, L), pad_id, dtype=torch.long)
        attn = torch.zeros((n, L), dtype=torch.long)
        labels = torch.full((n, T), -100, dtype=torch.long)
        for k, b in enumerate(batch):
            li, lt = len(b["input_ids"]), len(b["labels"])
            input_ids[k, :li] = b["input_ids"]
            attn[k, :li] = 1
            labels[k, :lt] = b["labels"]
            if "visible_ids" in b and len(b["visible_ids"]) > 0:
                visible_ids[k, :len(b["visible_ids"])] = b["visible_ids"]
            if "ctc_targets" in b and len(b["ctc_targets"]) > 0:
                ctc_targets[k, :len(b["ctc_targets"])] = b["ctc_targets"]
                ctc_lengths[k] = len(b["ctc_targets"])
        return {
            "ctc_targets": ctc_targets,
            "ctc_lengths": ctc_lengths,
            "visible_ids": visible_ids,
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "line_burn_crop": torch.stack([b["line_burn_crop"] for b in batch]),
            "line_budget": torch.stack([b["line_budget"] for b in batch]),
            "raw_prefix": [b["raw_prefix"] for b in batch],
            "raw_target_text": [b["raw_target_text"] for b in batch],
            "record_idx": [b["record_idx"] for b in batch],
        }
