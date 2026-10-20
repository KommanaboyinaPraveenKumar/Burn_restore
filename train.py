"""
train.py
========
Minimal, executable training loop wiring the pieces together:

    SideBurnGenerator -> SideBurnDocumentDataset -> DataLoader
        -> GroundedDocumentRestorer (dual-stream + fusion + alignment gate)
        -> Cross-Entropy loss on the clipped-text tokens
        -> AdamW optimization with progress logging

Run:
    python train.py --images path/to/images --texts path/to/texts
"""

import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config
from dataset import SideBurnDocumentDataset
from model import ForensicT5Restorer, GroundedDocumentRestorer
from metrics import compute_cer, compute_token_accuracy, extract_infill_text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def run_qualitative_eval(model, tokenizer, batch, config, device) -> None:
    """Generates a restoration for the first sample of a batch and reports
    CER against the ground truth, with every failure handled gracefully so
    a bad sample never stops training."""
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
        pred_text = extract_infill_text(tokenizer.decode(generated[0], skip_special_tokens=False))
        target_text = batch["raw_target_text"][0]
        prefix_text = batch["raw_prefix"][0]
        cer = compute_cer(pred_text, target_text)

        safe_prefix = prefix_text.encode("ascii", "replace").decode("ascii")
        safe_pred = pred_text.encode("ascii", "replace").decode("ascii")
        safe_target = target_text.encode("ascii", "replace").decode("ascii")
        print(f"    [eval] prefix:    {safe_prefix!r}")
        print(f"    [eval] predicted: {safe_pred!r}")
        print(f"    [eval] target:    {safe_target!r}")
        print(f"    [eval] Line CER:  {cer:.4f}")
    except Exception as exc:  # graceful degradation, never crash the run
        warnings.warn(f"[train] Qualitative eval skipped due to error: {exc}")
    finally:
        model.train()



def main() -> None:
    parser = argparse.ArgumentParser(description="Train GroundedDocumentRestorer.")
    parser.add_argument("--images", required=True, help="Folder of clean source page images.")
    parser.add_argument("--texts", required=True, help="Folder of matching .txt transcripts.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--decoder_lr", type=float, default=1e-4, help="Learning rate for newly initialized decoder/fusion layers.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None, help="Optional max training batches per epoch.")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu).")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing checkpoint to resume training from.")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Directory to store checkpoint files.")
    parser.add_argument("--ckpt_interval", type=int, default=1800, help="Checkpoint interval in seconds (default 1800).")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    if device.type == "cuda":
        print(f"[train] Utilizing GPU: {torch.cuda.get_device_name(0)}")
        print(f"[train] GPU Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB | Reserved: {torch.cuda.memory_reserved(0)/(1024**2):.1f} MB")
    else:
        print(f"[train] Utilizing device: {device}")
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    dataset = SideBurnDocumentDataset(config, args.images, args.texts, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=SideBurnDocumentDataset.collate_fn,
        drop_last=True,
    )
    print(f"[train] Dataset size: {len(dataset)} pages, {len(dataloader)} batches/epoch")

    model = ForensicT5Restorer(config).to(device)

    # Resume from checkpoint if specified or found
    if args.resume and os.path.isfile(args.resume):
        print(f"[train] Resuming weights from checkpoint: {args.resume}")
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
        except Exception as e:
            print(f"[train] Could not resume exact weights ({e}). Starting fresh initialization.")

    # Parameter groups: higher LR for visual adapter layers
    visual_params = list(model.visual_backbone.parameters()) + list(model.visual_proj.parameters()) + list(model.visual_norm.parameters())
    t5_params = list(model.t5.parameters())
    param_groups = [
        {"params": visual_params, "lr": 1e-4},
        {"params": t5_params, "lr": config.learning_rate},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)

            loss, logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                line_burn_crop=batch["line_burn_crop"],
                labels=batch["labels"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            if global_step % config.log_every == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    # Compute accuracy against non-pad target labels
                    valid_mask = batch["labels"] != -100
                    if valid_mask.any():
                        acc = (preds[valid_mask] == batch["labels"][valid_mask]).float().mean().item()
                    else:
                        acc = 0.0
                print(
                    f"[epoch {epoch + 1}/{config.epochs}] step {step + 1}/{len(dataloader)} "
                    f"(global {global_step}) | loss: {loss.item():.4f} | "
                    f"token_acc: {acc:.4f}"
                )
                if global_step % (config.log_every * 5) == 0:
                    run_qualitative_eval(model, tokenizer, batch, config, device)

            global_step += 1
            if args.max_steps_per_epoch is not None and (step + 1) >= args.max_steps_per_epoch:
                print(f"[epoch {epoch + 1}] Reached max_steps_per_epoch ({args.max_steps_per_epoch}). Finishing epoch early.")
                break

    print("[train] Training complete.")

    # Save model checkpoint for later testing/evaluation
    final_ckpt_path = os.path.join(ckpt_dir, "final_checkpoint.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"[train] Model checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
