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
    burn_profile: str = "moderate"          # "mild", "moderate", "severe", "irregular"
    random_walk_intensity: float = 0.12     # Gaussian step variance factor
    boundary_smoothness: float = 0.01       # 1D kernel size factor relative to height

    # ------------------------------------------------------------------ #
    # Dataset splitting parameters
    # ------------------------------------------------------------------ #
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def set_burn_profile(self, profile: str) -> None:
        """Configures preset physical corruption profiles."""
        self.burn_profile = profile.lower()
        if self.burn_profile == "mild":
            self.burn_min_ratio = 0.10
            self.burn_max_ratio = 0.18
            self.jagged_amplitude_ratio = 0.025
            self.transition_zone_px = 40
            self.scorch_noise_std = 4.0
            self.random_walk_intensity = 0.08
        elif self.burn_profile == "moderate":
            self.burn_min_ratio = 0.15
            self.burn_max_ratio = 0.30
            self.jagged_amplitude_ratio = 0.040
            self.transition_zone_px = 55
            self.scorch_noise_std = 6.0
            self.random_walk_intensity = 0.12
        elif self.burn_profile == "severe":
            self.burn_min_ratio = 0.25
            self.burn_max_ratio = 0.45
            self.jagged_amplitude_ratio = 0.060
            self.transition_zone_px = 75
            self.scorch_noise_std = 8.0
            self.random_walk_intensity = 0.16
        elif self.burn_profile == "irregular":
            self.burn_min_ratio = 0.15
            self.burn_max_ratio = 0.35
            self.jagged_amplitude_ratio = 0.080
            self.transition_zone_px = 65
            self.scorch_noise_std = 7.5
            self.random_walk_intensity = 0.20
        else:
            raise ValueError(f"Unknown burn profile '{profile}'. Choose from: mild, moderate, severe, irregular.")

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
    max_context_len: int = 256             # v3 prompts carry 2 lines of context each side
    max_target_len: int = 32
    avg_char_width_px: float = 14.0    # heuristic average glyph width used for the alignment budget

    # ------------------------------------------------------------------ #
    # Vision model (Local Forensic Eye)
    # ------------------------------------------------------------------ #
    use_vision: bool = True            # False = text-only baseline (no vision tokens / fusion)
    vision_backbone: str = "resnet34"
    pretrained_vision: bool = True

    # ------------------------------------------------------------------ #
    # Fusion / decoder
    # ------------------------------------------------------------------ #
    embed_dim: int = 512               # shared fusion dimension (matches T5-small d_model)
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    use_cross_modal_fusion: bool = True     # Gated multi-head cross-attention between text and vision
    fusion_style: str = "flamingo"          # "flamingo" = residual zero-init gates (default); "legacy" = pre-fix LayerNorm fusion
    use_spatial_pos_embed: bool = True      # 2D spatial positional embeddings (row/col)
    use_multi_scale_vision: bool = True     # Multi-scale feature fusion (ResNet layer3 + layer4)
    vision_grid: str = "fine"               # "fine" = 8x16 tokens (16 px each); "coarse" = 4x8 (legacy)
    use_reading_head: bool = True           # CTC head that reads the visible text in the crop
    lambda_read: float = 0.3                # weight of the CTC reading loss
    prompt_style: str = "v3"                # "v3" = 2 lines of context each side + length hint; "v2" = legacy;
                                            # "v3e" = v3 with the length hint from the predicted cut (cut estimator)
    context_noise: float = 0.04             # max OCR-style error rate injected into context during training
    gradient_checkpointing: bool = False    # T5 activation checkpointing (needed for t5-base on 6 GB)
    lambda_contrastive: float = 0.1        # Weight for contrastive stroke-text alignment loss
    contrastive_temp: float = 0.07          # InfoNCE temperature
    use_physical_budget_processor: bool = True  # Autoregressive physical character budget processor
    alignment_penalty_weight: float = 4.0   # Penalty weight for exceeding physical character budget
    budget_margin_buffer: float = 2.0       # Allowed character tolerance buffer before penalty
    budget_mode: str = "soft"               # "soft" (legacy EOS bias) | "hard" (upper-bound look-ahead mask)
    profile_bins: int = 16             # number of vertical bins used to summarize the burn-width profile

    # ------------------------------------------------------------------ #
    # Training & Optimization (Phase 3 Staged Training Engine)
    # ------------------------------------------------------------------ #
    batch_size: int = 8
    learning_rate: float = 1e-4             # fallback single-LR
    weight_decay: float = 1e-4
    epochs: int = 5
    grad_clip_norm: float = 1.0
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1

    # Staged fine-tuning
    warmup_epochs: int = 1                  # Stage 1 adapter-only warmup epochs
    freeze_t5_warmup: bool = True           # Freeze T5 backbone during Stage 1
    lr_t5_backbone: float = 1e-5            # Stage 2 T5 backbone LR
    lr_vision_backbone: float = 1e-5        # Stage 2 ResNet backbone LR
    lr_adapters: float = 5e-5               # Stage 2 projection, fusion, aligner LR
    lr_warmup_adapters: float = 1e-4        # Stage 1 adapter warmup LR

    # Learning rate scheduler
    scheduler_type: str = "cosine"          # "cosine", "linear", "constant", "none"
    scheduler_warmup_ratio: float = 0.1

    # Dynamic contrastive loss scheduling
    contrastive_sched_start: float = 0.2    # Initial lambda_contrastive
    contrastive_sched_end: float = 0.05     # Final lambda_contrastive after decay
    contrastive_sched_type: str = "linear"  # "linear", "fixed", "cosine"

    # Mixed precision & hardware
    use_amp: bool = True                    # Enabled for CUDA; automatically falls back on CPU
    amp_dtype: str = "bfloat16"             # T5 overflows in fp16; bf16 is stable on Ampere/Ada GPUs

    # Evaluation, early stopping & checkpointing
    val_every_epochs: int = 1
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.005
    early_stopping_metric: str = "val_cer"

    # Paths and manifests
    splits_manifest: str = "data/splits.json"
    checkpoint_dir: str = "checkpoints"
    log_history_path: str = "checkpoints/training_history.jsonl"

    log_every: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
