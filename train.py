"""
train.py
========
Advanced Multimodal Staged Training and Optimization Engine for Grounded
Side-Burnt Document Text Restoration (Phase 3).

Key Capabilities:
  1. Staged Fine-Tuning:
     - Stage 1 (Adapter Warmup): Freezes T5 backbone; warms up visual projection,
       gated cross-modal fusion, and contrastive aligner.
     - Stage 2 (Joint Fine-Tuning): Unfreezes T5 and optimizes with three disjoint
       differential learning rate parameter groups (T5, ResNet, Adapters).
  2. Loss Scheduling:
     - Dynamic annealing of contrastive stroke loss lambda_contrast(t).
  3. Optimization & Precision:
     - Supports mixed precision (AMP GradScaler on CUDA, FP32 on CPU).
     - Gradient accumulation over arbitrary micro-batch steps.
     - Cosine / linear annealing with warmup steps.
     - NaN / Inf loss guards.
  4. Rigorous Validation & Early Stopping:
     - Deterministic validation tracking CER, WER, Character Accuracy, Word Accuracy,
       Line Exact Match, and physical budget violation rate.
     - Automatic best checkpoint selection on lowest validation CER.
     - Early stopping with configurable patience and min_delta.
  5. Checkpoint Management & Telemetry:
     - Saves latest_checkpoint.pt, best_checkpoint.pt, and final_checkpoint.pt.
     - Complete resumption support (--resume).
     - Structured JSONL logging to training_history.jsonl.
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# Ensure Burn_restore directory and workspace root are on sys.path
_burn_dir = Path(__file__).resolve().parent
_root_dir = _burn_dir.parent
for p in [str(_burn_dir), str(_root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import Config
    from dataset import LineCacheDataset, LineGroundedDocumentDataset, SideBurnDocumentDataset
    from line_data import extract_infill
    from model import ForensicT5Restorer
    from metrics import (
        compute_cer,
        compute_wer,
        compute_character_accuracy,
        compute_word_accuracy,
        compute_exact_match_line_accuracy,
        extract_infill_text,
    )
except ImportError:
    from Burn_restore.config import Config
    from Burn_restore.dataset import LineCacheDataset, LineGroundedDocumentDataset, SideBurnDocumentDataset
    from Burn_restore.line_data import extract_infill
    from Burn_restore.model import ForensicT5Restorer
    from Burn_restore.metrics import (
        compute_cer,
        compute_wer,
        compute_character_accuracy,
        compute_word_accuracy,
        compute_exact_match_line_accuracy,
        extract_infill_text,
    )


def set_seed(seed: int) -> None:
    """Enforces deterministic behavior across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Moves all tensor elements in a batch dictionary to target device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_optimizer_and_scheduler(
    model: ForensicT5Restorer,
    config: Config,
    stage: int,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Constructs optimizer parameter groups and learning rate schedulers for Stage 1 or Stage 2.
    
    Stage 1:
      - T5 backbone is frozen (model.t5.requires_grad_(False)).
      - Adapters & fusion train with config.lr_warmup_adapters.
    
    Stage 2:
      - T5 is unfrozen (model.t5.requires_grad_(True)).
      - 3 Disjoint parameter groups with differential learning rates:
          1. T5 backbone: config.lr_t5_backbone
          2. Vision backbone: config.lr_vision_backbone
          3. Adapters & fusion: config.lr_adapters
    """
    total_steps = max(1, total_steps)
    warmup_steps = int(total_steps * config.scheduler_warmup_ratio)

    if stage == 1:
        # Stage 1: Freeze T5 backbone
        if config.freeze_t5_warmup:
            model.t5.requires_grad_(False)
        else:
            model.t5.requires_grad_(True)

        # Trainable adapter parameters (fusion gates get their own, faster group)
        gate_ids = _gate_param_ids(model)
        adapter_params = [p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids]
        gate_params = [p for p in model.parameters() if p.requires_grad and id(p) in gate_ids]
        groups = [{"params": adapter_params, "lr": config.lr_warmup_adapters, "name": "adapters"}]
        if gate_params:
            groups.append({"params": gate_params, "lr": config.lr_warmup_adapters * GATE_LR_MULT,
                           "weight_decay": 0.0, "name": "fusion_gates"})
        optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    else:
        # Stage 2: Unfreeze full model
        model.t5.requires_grad_(True)
        model.visual_encoder.requires_grad_(True)

        t5_param_ids = {id(p) for p in model.t5.parameters()}
        if hasattr(model.visual_encoder, "resnet_parameters"):
            vision_backbone_params = list(model.visual_encoder.resnet_parameters)
        elif hasattr(model.visual_encoder, "resnet"):
            vision_backbone_params = list(model.visual_encoder.resnet.parameters())
        else:
            vision_backbone_params = (
                list(model.visual_encoder.stem.parameters())
                + list(model.visual_encoder.layer1.parameters())
                + list(model.visual_encoder.layer2.parameters())
                + list(model.visual_encoder.layer3.parameters())
                + list(model.visual_encoder.layer4.parameters())
            )
        resnet_param_ids = {id(p) for p in vision_backbone_params}

        group_t5 = [p for p in model.t5.parameters() if p.requires_grad]
        group_vision = [p for p in vision_backbone_params if p.requires_grad]
        gate_ids = _gate_param_ids(model)
        group_gates = [p for p in model.parameters() if p.requires_grad and id(p) in gate_ids]
        group_adapters = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in t5_param_ids and id(p) not in resnet_param_ids
            and id(p) not in gate_ids
        ]

        # Verify disjointness
        s_t5 = set(id(p) for p in group_t5)
        s_vis = set(id(p) for p in group_vision)
        s_adp = set(id(p) for p in group_adapters)
        assert len(s_t5 & s_vis) == 0, "T5 and Vision parameter groups overlap!"
        assert len(s_t5 & s_adp) == 0, "T5 and Adapter parameter groups overlap!"
        assert len(s_vis & s_adp) == 0, "Vision and Adapter parameter groups overlap!"

        param_groups = [
            {"params": group_t5, "lr": config.lr_t5_backbone, "name": "t5_backbone"},
            {"params": group_vision, "lr": config.lr_vision_backbone, "name": "vision_backbone"},
            {"params": group_adapters, "lr": config.lr_adapters, "name": "adapters_and_fusion"},
        ]
        if group_gates:
            param_groups.append({"params": group_gates, "lr": config.lr_adapters * GATE_LR_MULT,
                                 "weight_decay": 0.0, "name": "fusion_gates"})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    # Configure LR Scheduler
    sched_type = getattr(config, "scheduler_type", "cosine").lower()
    if sched_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif sched_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = None

    return optimizer, scheduler


GATE_LR_MULT = 10.0  # zero-initialised fusion gates need a larger step to open


def _gate_param_ids(model) -> set:
    """ids of the scalar tanh gates of the residual fusion (if present)."""
    fusion = getattr(model, "cross_modal_fusion", None)
    if fusion is None:
        return set()
    return {id(p) for n, p in fusion.named_parameters() if n.endswith("_gate")}


def compute_scheduled_contrastive_weight(epoch: int, total_epochs: int, config: Config) -> float:
    """Computes dynamic contrastive loss weight lambda_contrast(t)."""
    if config.lambda_contrastive <= 0.0:
        return 0.0

    sched_type = getattr(config, "contrastive_sched_type", "linear").lower()
    if sched_type == "fixed":
        return float(config.lambda_contrastive)

    start = getattr(config, "contrastive_sched_start", 0.2)
    end = getattr(config, "contrastive_sched_end", 0.05)

    if total_epochs <= 1:
        return start

    progress = float(epoch) / float(max(1, total_epochs - 1))
    progress = min(max(progress, 0.0), 1.0)

    if sched_type == "cosine":
        decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(end + (start - end) * decay)
    else:
        # Linear decay
        return float(start + (end - start) * progress)


@torch.no_grad()
def evaluate_model(
    model: ForensicT5Restorer,
    dataloader: DataLoader,
    tokenizer: Any,
    config: Config,
    device: torch.device,
    lambda_contrastive: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Executes deterministic evaluation across the validation split."""
    model.eval()
    total_edits = 0
    total_chars = 0
    total_loss = 0.0
    total_gen_loss = 0.0
    total_contrast_loss = 0.0
    num_batches = 0

    all_cer: List[float] = []
    all_wer: List[float] = []
    all_char_acc: List[float] = []
    all_word_acc: List[float] = []
    all_exact_match: List[float] = []
    budget_violations = 0
    total_samples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        b_size = batch["input_ids"].size(0)
        total_samples += b_size

        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                line_burn_crop=batch["line_burn_crop"],
                labels=batch["labels"],
                alignment_confidence=batch.get("alignment_confidence"),
                lambda_contrastive=lambda_contrastive,
            )

        total_loss += float(loss.item())
        total_gen_loss += getattr(model, "last_gen_loss", float(loss.item()))
        total_contrast_loss += getattr(model, "last_contrast_loss", 0.0)
        num_batches += 1

        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                line_burn_crop=batch["line_burn_crop"],
                line_budget=batch.get("line_budget"),
                max_len=config.max_target_len,
                use_physical_grounding=getattr(config, "use_physical_budget_processor", True),
            )

        raw_targets = batch["raw_target_text"]
        budgets = batch.get("line_budget")

        for b_idx in range(b_size):
            pred_raw = tokenizer.decode(generated[b_idx], skip_special_tokens=False)
            pred_text = extract_infill(pred_raw)
            target_text = raw_targets[b_idx] if isinstance(raw_targets, list) else raw_targets

            cer = compute_cer(pred_text, target_text)
            if cer is not None:
                total_edits += cer * max(len(target_text), 1)
                total_chars += max(len(target_text), 1)
            wer = compute_wer(pred_text, target_text)
            c_acc = compute_character_accuracy(pred_text, target_text)
            w_acc = compute_word_accuracy(pred_text, target_text)
            line_em = compute_exact_match_line_accuracy(pred_text, target_text)

            if cer is not None:
                all_cer.append(cer)
            if wer is not None:
                all_wer.append(wer)
            all_char_acc.append(c_acc)
            all_word_acc.append(w_acc)
            all_exact_match.append(line_em)

            if budgets is not None:
                b_val = float(budgets[b_idx].item() if isinstance(budgets[b_idx], torch.Tensor) else budgets[b_idx])
                if len(pred_text) > (b_val + config.budget_margin_buffer):
                    budget_violations += 1

    return {
        "val_loss": round(total_loss / max(1, num_batches), 4),
        "val_gen_loss": round(total_gen_loss / max(1, num_batches), 4),
        "val_contrast_loss": round(total_contrast_loss / max(1, num_batches), 4),
        "val_cer": round(total_edits / total_chars, 4) if total_chars else 1.0,  # corpus CER
        "val_macro_cer": round(float(np.mean(all_cer)) if all_cer else 1.0, 4),
        "val_wer": round(float(np.mean(all_wer)) if all_wer else 1.0, 4),
        "val_char_acc": round(float(np.mean(all_char_acc)) if all_char_acc else 0.0, 4),
        "val_word_acc": round(float(np.mean(all_word_acc)) if all_word_acc else 0.0, 4),
        "val_line_em": round(float(np.mean(all_exact_match)) if all_exact_match else 0.0, 4),
        "budget_violation_rate": round(budget_violations / max(1, total_samples), 4),
        "total_eval_samples": total_samples,
    }


def save_checkpoint(
    path: str,
    model: ForensicT5Restorer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    metrics: Dict[str, Any],
    config: Config,
    weights_only: bool = False,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves a training checkpoint.  ``weights_only`` omits optimizer/scheduler
    state (3x smaller) and is used for best/final checkpoints."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    full = not weights_only
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if (full and optimizer is not None) else None,
        "scheduler_state_dict": scheduler.state_dict() if (full and scheduler is not None) else None,
        "scaler_state_dict": scaler.state_dict() if (full and scaler is not None) else None,
        "trainer_state": extra_state or {},
        "metrics": metrics,
        "config_dict": {
            k: getattr(config, k)
            for k in dir(config)
            if not k.startswith("_") and not callable(getattr(config, k))
        },
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: ForensicT5Restorer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads checkpoint weights and optionally restores optimizer and scheduler state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                warnings.warn(f"[train] Could not load optimizer state: {e}")
        if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                warnings.warn(f"[train] Could not load scheduler state: {e}")
        if scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                warnings.warn(f"[train] Could not load scaler state: {e}")
        return {
            "epoch": ckpt.get("epoch", 0),
            "global_step": ckpt.get("global_step", 0),
            "metrics": ckpt.get("metrics", {}),
            "trainer_state": ckpt.get("trainer_state", {}),
        }
    else:
        model.load_state_dict(ckpt)
        return {"epoch": 0, "global_step": 0, "metrics": {}}


def run_qualitative_eval(
    model: ForensicT5Restorer,
    tokenizer: Any,
    batch: Dict[str, Any],
    config: Config,
    device: torch.device,
) -> None:
    """Generates a qualitative restoration sample and logs comparison."""
    try:
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"][:1],
                attention_mask=batch["attention_mask"][:1],
                line_burn_crop=batch["line_burn_crop"][:1],
                line_budget=batch["line_budget"][:1] if "line_budget" in batch else None,
                max_len=config.max_target_len,
            )
        pred_text = extract_infill(tokenizer.decode(generated[0], skip_special_tokens=False))
        target_text = batch["raw_target_text"][0]
        prefix_text = batch["raw_prefix"][0]
        cer = compute_cer(pred_text, target_text)
        wer = compute_wer(pred_text, target_text)

        safe_prefix = prefix_text.encode("ascii", "replace").decode("ascii")
        safe_pred = pred_text.encode("ascii", "replace").decode("ascii")
        safe_target = target_text.encode("ascii", "replace").decode("ascii")
        print(f"    [eval-sample] prefix:    {safe_prefix!r}")
        print(f"    [eval-sample] predicted: {safe_pred!r}")
        print(f"    [eval-sample] target:    {safe_target!r}")
        print(f"    [eval-sample] CER: {cer:.4f} | WER: {wer:.4f}")
    except Exception as exc:
        warnings.warn(f"[train] Qualitative eval skipped due to error: {exc}")
    finally:
        model.train()


def train_restorer(
    config: Config,
    splits_manifest: Optional[str] = None,
    image_dir: Optional[str] = None,
    text_dir: Optional[str] = None,
    resume_path: Optional[str] = None,
    max_steps_per_epoch: Optional[int] = None,
    data_dir: Optional[str] = None,
    val_limit: Optional[int] = 600,
    train_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """High-level training entry point executing Staged Fine-Tuning."""
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.bfloat16 if getattr(config, "amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    print(f"[train] Initializing Training Pipeline on {device} (AMP: {amp_dtype})")

    ckpt_dir = os.path.abspath(config.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    history_file = os.path.abspath(config.log_history_path)
    os.makedirs(os.path.dirname(history_file), exist_ok=True)

    # Initialize Tokenizer and Datasets
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

    manifest_p = splits_manifest or getattr(config, "splits_manifest", "data/splits.json")
    collate = LineGroundedDocumentDataset.collate_fn
    if data_dir:
        print(f"[train] Loading line cache from: {data_dir}")
        style = getattr(config, "prompt_style", "v2")
        train_dataset = LineCacheDataset(data_dir, "train", tokenizer, config.max_context_len,
                                         config.max_target_len, augment=True, limit=train_limit,
                                         seed=config.seed, prompt_style=style,
                                         context_noise=getattr(config, "context_noise", 0.0))
        val_dataset = LineCacheDataset(data_dir, "val", tokenizer, config.max_context_len,
                                       config.max_target_len, augment=False, limit=val_limit,
                                       seed=config.seed, prompt_style=style)
        print(f"[train] prompt_style={style} context_noise={getattr(config, 'context_noise', 0.0)} "
              f"text_model={config.text_model_name} vision_grid={getattr(config, 'vision_grid', '-')} "
              f"reading_head={getattr(config, 'use_reading_head', False)}")
        collate = LineCacheDataset.make_collate(tokenizer.pad_token_id)
    elif os.path.isfile(manifest_p):
        print(f"[train] Loading splits from manifest: {manifest_p}")
        train_dataset = LineGroundedDocumentDataset(
            config=config,
            tokenizer=tokenizer,
            split="train",
            splits_manifest=manifest_p,
            is_eval=False,
        )
        val_dataset = LineGroundedDocumentDataset(
            config=config,
            tokenizer=tokenizer,
            split="val",
            splits_manifest=manifest_p,
            is_eval=True,
        )
    elif image_dir and text_dir:
        print(f"[train] Loading dataset from directories: {image_dir}, {text_dir}")
        train_dataset = SideBurnDocumentDataset(
            config=config,
            image_dir=image_dir,
            text_dir=text_dir,
            tokenizer=tokenizer,
            is_eval=False,
        )
        val_dataset = SideBurnDocumentDataset(
            config=config,
            image_dir=image_dir,
            text_dir=text_dir,
            tokenizer=tokenizer,
            is_eval=True,
        )
    else:
        # Fallback check inside Burn_restore/data
        alt_manifest = _burn_dir / "data" / "splits.json"
        if alt_manifest.is_file():
            manifest_p = str(alt_manifest)
            train_dataset = LineGroundedDocumentDataset(
                config=config, tokenizer=tokenizer, split="train", splits_manifest=manifest_p, is_eval=False
            )
            val_dataset = LineGroundedDocumentDataset(
                config=config, tokenizer=tokenizer, split="val", splits_manifest=manifest_p, is_eval=True
            )
        else:
            raise FileNotFoundError(f"Could not locate splits manifest at {manifest_p} or source directories.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    print(f"[train] Train instances: {len(train_dataset)} | Val instances: {len(val_dataset)}")
    print(f"[train] Batches per epoch: {len(train_dataloader)}")

    # Initialize Multimodal Model
    model = ForensicT5Restorer(config).to(device)

    # GradScaler is only needed for fp16; bf16 has fp32 range
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Resume if requested (weights first; optimizer/scheduler after they are built)
    start_epoch = 0
    global_step = 0
    best_val_cer = float("inf")
    patience_counter = 0
    ckpt_meta: Dict[str, Any] = {}

    if resume_path and os.path.isfile(resume_path):
        print(f"[train] Resuming state from checkpoint: {resume_path}")
        ckpt_meta = load_checkpoint(resume_path, model, device=str(device))
        start_epoch = ckpt_meta.get("epoch", 0)
        global_step = ckpt_meta.get("global_step", 0)
        tstate = ckpt_meta.get("trainer_state", {})
        best_val_cer = tstate.get("best_val_cer", ckpt_meta.get("metrics", {}).get("val_cer", float("inf")))
        patience_counter = tstate.get("patience_counter", 0)

    # Determine initial stage
    current_stage = 1 if (start_epoch < config.warmup_epochs) else 2
    steps_per_epoch = len(train_dataloader)
    if max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)

    stage_total_steps = (
        (config.warmup_epochs if current_stage == 1 else (config.epochs - config.warmup_epochs))
        * steps_per_epoch
        // max(1, config.grad_accum_steps)
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        config=config,
        stage=current_stage,
        total_steps=stage_total_steps,
    )
    if resume_path and os.path.isfile(resume_path):
        # Restore optimizer / scheduler / scaler now that they match the current stage
        load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, device=str(device))

    print(f"[train] Starting Training at Stage {current_stage} (Epoch {start_epoch + 1}/{config.epochs})")

    training_history = []

    # Main Training Loop
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()

        # Handle Stage transition
        expected_stage = 1 if (epoch < config.warmup_epochs) else 2
        if expected_stage != current_stage:
            print(f"\n[train] >>> Transitioning from Stage {current_stage} to Stage {expected_stage} (Joint Fine-Tuning) <<<")
            current_stage = expected_stage
            remaining_steps = (
                (config.epochs - epoch) * steps_per_epoch // max(1, config.grad_accum_steps)
            )
            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                config=config,
                stage=current_stage,
                total_steps=remaining_steps,
            )

        # Dynamic epoch augmentation seeding
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        # Dynamic contrastive weight
        current_lambda_c = compute_scheduled_contrastive_weight(epoch, config.epochs, config)

        model.train()
        train_loss_accum = 0.0
        train_gen_accum = 0.0
        train_cont_accum = 0.0
        train_token_correct = 0
        train_token_total = 0
        epoch_steps = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                loss, logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    line_burn_crop=batch["line_burn_crop"],
                    labels=batch["labels"],
                    alignment_confidence=batch.get("alignment_confidence"),
                    lambda_contrastive=current_lambda_c,
                    visible_ids=batch.get("visible_ids"),
                    ctc_targets=batch.get("ctc_targets"),
                    ctc_lengths=batch.get("ctc_lengths"),
                )
                scaled_loss = loss / config.grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn(f"[train] NaN or Inf loss encountered at step {global_step}. Skipping step.")
                optimizer.zero_grad()
                continue

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Accumulate metrics
            train_loss_accum += float(loss.detach().item())
            train_gen_accum += getattr(model, "last_gen_loss", float(loss.detach().item()))
            train_cont_accum += getattr(model, "last_contrast_loss", 0.0)

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid_mask = batch["labels"] != -100
                if valid_mask.any():
                    train_token_correct += int((preds[valid_mask] == batch["labels"][valid_mask]).sum().item())
                    train_token_total += int(valid_mask.sum().item())

            # Gradient step at accumulation boundary
            if (step + 1) % config.grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()
                global_step += 1

            epoch_steps += 1
            if step % config.log_every == 0 or (step + 1) == len(train_dataloader):
                cur_acc = (train_token_correct / max(1, train_token_total))
                print(
                    f"[ep {epoch + 1}/{config.epochs} | Stage {current_stage}] step {step + 1}/{len(train_dataloader)} "
                    f"| loss: {loss.item():.4f} (gen: {getattr(model, 'last_gen_loss', 0.0):.4f}, cont: {getattr(model, 'last_contrast_loss', 0.0):.4f}, read: {getattr(model, 'last_read_loss', 0.0):.4f}) "
                    f"| lambda_c: {current_lambda_c:.4f} | token_acc: {cur_acc:.4f}"
                )

            if max_steps_per_epoch is not None and (step + 1) >= max_steps_per_epoch:
                print(f"[train] Reached max_steps_per_epoch ({max_steps_per_epoch}). Finishing epoch early.")
                break

        # Compute epoch training metrics
        mean_train_loss = train_loss_accum / max(1, epoch_steps)
        mean_train_gen = train_gen_accum / max(1, epoch_steps)
        mean_train_cont = train_cont_accum / max(1, epoch_steps)
        train_token_acc = train_token_correct / max(1, train_token_total)

        # Validation Loop
        should_val = ((epoch + 1) % config.val_every_epochs == 0) or ((epoch + 1) == config.epochs)
        val_metrics: Dict[str, Any] = {}
        if should_val and len(val_dataset) > 0:
            print(f"[train] Running validation for epoch {epoch + 1}...")
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_dataloader,
                tokenizer=tokenizer,
                config=config,
                device=device,
                lambda_contrastive=current_lambda_c,
                amp_dtype=amp_dtype,
            )
            print(
                f"[val {epoch + 1}/{config.epochs}] val_loss: {val_metrics['val_loss']:.4f} | "
                f"val_cer: {val_metrics['val_cer']:.4f} | val_wer: {val_metrics['val_wer']:.4f} | "
                f"char_acc: {val_metrics['val_char_acc']:.4f} | word_acc: {val_metrics['val_word_acc']:.4f} | "
                f"line_em: {val_metrics['val_line_em']:.4f} | budget_violation: {val_metrics['budget_violation_rate']:.2%}"
            )

            # Qualitative visual sample log
            try:
                sample_batch = next(iter(val_dataloader))
                sample_batch = move_batch_to_device(sample_batch, device)
                run_qualitative_eval(model, tokenizer, sample_batch, config, device)
            except Exception as e:
                warnings.warn(f"[train] Qualitative inspection skipped: {e}")

        epoch_duration = round(time.time() - epoch_start_time, 2)

        # Record Telemetry
        epoch_record = {
            "epoch": epoch + 1,
            "stage": current_stage,
            "global_step": global_step,
            "train_loss": round(mean_train_loss, 4),
            "train_gen_loss": round(mean_train_gen, 4),
            "train_contrast_loss": round(mean_train_cont, 4),
            "train_token_acc": round(train_token_acc, 4),
            "lambda_contrastive": round(current_lambda_c, 4),
            "duration_sec": epoch_duration,
            **val_metrics,
        }
        training_history.append(epoch_record)

        # Append to JSONL history file
        with open(history_file, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(epoch_record) + "\n")

        # Best checkpoint tracking first, so trainer_state in latest is up to date
        cur_val_cer = val_metrics.get("val_cer", None)
        improved = cur_val_cer is not None and cur_val_cer < (best_val_cer - config.early_stopping_min_delta)
        if improved:
            best_val_cer = cur_val_cer
            patience_counter = 0
        elif cur_val_cer is not None:
            patience_counter += 1
        trainer_state = {"best_val_cer": best_val_cer, "patience_counter": patience_counter}

        # Save Latest Checkpoint (full state, for --resume)
        latest_ckpt_p = os.path.join(ckpt_dir, "latest_checkpoint.pt")
        save_checkpoint(
            path=latest_ckpt_p,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            metrics=epoch_record,
            config=config,
            extra_state=trainer_state,
        )

        if improved:
            best_ckpt_p = os.path.join(ckpt_dir, "best_checkpoint.pt")
            save_checkpoint(
                path=best_ckpt_p,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                metrics=epoch_record,
                config=config,
                weights_only=True,
            )
            print(f"[train] >>> Best checkpoint updated! Lowest val_cer: {best_val_cer:.4f} <<<")
        else:
            if patience_counter >= config.early_stopping_patience and epoch >= config.warmup_epochs:
                print(
                    f"\n[train] Early stopping triggered! Validation CER failed to improve "
                    f"for {config.early_stopping_patience} consecutive epochs. Terminating training."
                )
                break

    # Save Final Checkpoint
    final_ckpt_p = os.path.join(ckpt_dir, "final_checkpoint.pt")
    save_checkpoint(
        path=final_ckpt_p,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch + 1,
        global_step=global_step,
        metrics=training_history[-1] if training_history else {},
        config=config,
        weights_only=True,
    )
    print(f"\n[train] Training completed. Final checkpoint saved to {final_ckpt_p}")

    return {
        "best_val_cer": best_val_cer,
        "epochs_completed": epoch + 1,
        "global_steps": global_step,
        "history": training_history,
        "final_checkpoint": final_ckpt_p,
        "best_checkpoint": os.path.join(ckpt_dir, "best_checkpoint.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced Multimodal Document Restoration Training Engine (Phase 3).")
    parser.add_argument("--splits_manifest", type=str, default="data/splits.json", help="Path to splits.json.")
    parser.add_argument("--images", type=str, default=None, help="Folder of clean source page images.")
    parser.add_argument("--texts", type=str, default=None, help="Folder of matching transcripts.")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs.")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Stage 1 adapter warmup epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--lr_t5", type=float, default=None, help="Stage 2 T5 backbone LR.")
    parser.add_argument("--lr_vision", type=float, default=None, help="Stage 2 ResNet backbone LR.")
    parser.add_argument("--lr_adapters", type=float, default=None, help="Stage 2 adapter & fusion LR.")
    parser.add_argument("--lr_warmup", type=float, default=None, help="Stage 1 adapter warmup LR.")
    parser.add_argument("--lambda_contrast", type=float, default=None, help="Weight for contrastive stroke loss.")
    parser.add_argument("--contrastive_sched", type=str, default=None, choices=["linear", "fixed", "cosine"])
    parser.add_argument("--max_steps_per_epoch", type=int, default=None, help="Optional max batches per epoch.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to store checkpoints.")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision AMP.")
    parser.add_argument("--data_dir", type=str, default=None, help="Line cache from scripts/build_line_dataset.py.")
    parser.add_argument("--val_limit", type=int, default=600, help="Validation lines evaluated per epoch.")
    parser.add_argument("--train_limit", type=int, default=None, help="Optional cap on training lines.")
    parser.add_argument("--text_only", action="store_true", help="Train the text-only baseline (no vision).")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs).")
    parser.add_argument("--text_model", default=None, help="HF T5 model name, e.g. t5-small / t5-base.")
    parser.add_argument("--prompt_style", choices=["v2", "v3", "v3e"], default=None)
    parser.add_argument("--context_noise", type=float, default=None, help="Max OCR-noise rate on context (train).")
    parser.add_argument("--vision_grid", choices=["coarse", "fine"], default=None)
    parser.add_argument("--no_reading_head", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Trade compute for T5 memory.")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.warmup_epochs is not None:
        config.warmup_epochs = args.warmup_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = args.grad_accum_steps
    if args.lr_t5 is not None:
        config.lr_t5_backbone = args.lr_t5
    if args.lr_vision is not None:
        config.lr_vision_backbone = args.lr_vision
    if args.lr_adapters is not None:
        config.lr_adapters = args.lr_adapters
    if args.lr_warmup is not None:
        config.lr_warmup_adapters = args.lr_warmup
    if args.lambda_contrast is not None:
        config.lambda_contrastive = args.lambda_contrast
    if args.contrastive_sched is not None:
        config.contrastive_sched_type = args.contrastive_sched
    if args.device is not None:
        config.device = args.device
    if args.ckpt_dir is not None:
        config.checkpoint_dir = args.ckpt_dir
    if args.no_amp:
        config.use_amp = False
    if args.text_model is not None:
        config.text_model_name = args.text_model
    if args.prompt_style is not None:
        config.prompt_style = args.prompt_style
    if args.context_noise is not None:
        config.context_noise = args.context_noise
    if args.vision_grid is not None:
        config.vision_grid = args.vision_grid
    if args.no_reading_head:
        config.use_reading_head = False
    config.gradient_checkpointing = bool(args.gradient_checkpointing)
    if args.text_only:
        config.use_reading_head = False
        config.use_vision = False
        config.use_cross_modal_fusion = False
        config.lambda_contrastive = 0.0
        config.warmup_epochs = 0  # stage 1 would freeze T5 and leave nothing trainable
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.log_every is not None:
        config.log_every = args.log_every
    if args.patience is not None:
        config.early_stopping_patience = args.patience
    config.log_history_path = os.path.join(config.checkpoint_dir, "training_history.jsonl")

    train_restorer(
        config=config,
        splits_manifest=args.splits_manifest,
        image_dir=args.images,
        text_dir=args.texts,
        resume_path=args.resume,
        max_steps_per_epoch=args.max_steps_per_epoch,
        data_dir=args.data_dir,
        val_limit=args.val_limit,
        train_limit=args.train_limit,
    )


if __name__ == "__main__":
    main()
