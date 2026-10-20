"""
config.py
=========
Central configuration for the Grounded Side-Burnt Document Text Restoration
framework. A single dataclass is shared across the dataset generator, the
PyTorch ``Dataset``, the model, and the training loop so every component
agrees on image sizes, burn-synthesis parameters, and model dimensions.
"""

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Burn synthesis parameters
    # ------------------------------------------------------------------ #
    burn_min_ratio: float = 0.15            # min burned width, fraction of page width
    burn_max_ratio: float = 0.30            # max burned width, fraction of page width
    jagged_amplitude_ratio: float = 0.04    # jaggedness of the burn edge, fraction of page width
    transition_zone_px: int = 55            # width (px) of the ghost-stroke fade band
    charcoal_color: Tuple[int, int, int] = (18, 14, 12)   # BGR dark charcoal residue color
    scorch_noise_std: float = 6.0           # gaussian noise injected into the charred region
    min_ink_row_density: float = 0.015      # fraction of black pixels in a row-band to call it "text"
    line_gap_merge_px: int = 6              # merge row-bands separated by less than this (px)

    # ------------------------------------------------------------------ #
    # Cropping / vision input
    # ------------------------------------------------------------------ #
    page_resize: Tuple[int, int] = (1024, 1024)   # (W, H) source page is resized to before burning
    crop_size: Tuple[int, int] = (256, 256)       # (W, H) fed into legacy vision modules
    line_crop_size: Tuple[int, int] = (128, 256)  # (H, W) high-resolution local line crop
    crop_context_px: int = 180                    # healthy context (px) kept to the left of the burn bbox

    # ------------------------------------------------------------------ #
    # Text model (Global Context Engine & Seq2Seq)
    # ------------------------------------------------------------------ #
    text_model_name: str = "t5-small"
    max_context_len: int = 384
    max_target_len: int = 32
    avg_char_width_px: float = 14.0    # heuristic average glyph width used for the alignment budget

    # ------------------------------------------------------------------ #
    # Vision model (Local Forensic Eye)
    # ------------------------------------------------------------------ #
    vision_backbone: str = "resnet34"
    pretrained_vision: bool = True

    # ------------------------------------------------------------------ #
    # Fusion / decoder
    # ------------------------------------------------------------------ #
    embed_dim: int = 512               # shared fusion dimension (matches T5-small d_model)
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_ff_dim: int = 2048
    alignment_penalty_weight: float = 4.0
    profile_bins: int = 16             # number of vertical bins used to summarize the burn-width profile

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    batch_size: int = 8
    learning_rate: float = 1e-4

    weight_decay: float = 1e-4
    epochs: int = 3
    grad_clip_norm: float = 1.0
    log_every: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
