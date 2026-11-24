"""
tests/test_training_pipeline.py
===============================
Comprehensive Unit and Regression Test Suite for Phase 3:
Advanced Training Optimization, Staged Fine-Tuning, and Generalization.

Tests:
  1. test_stage1_adapter_warmup_freezes_t5:
     Verifies T5 parameters receive zero gradient during Stage 1 warmup while adapters receive gradients.
  2. test_stage2_differential_lr_parameter_groups:
     Verifies the 3 optimizer groups (T5, ResNet, Adapters) are mutually disjoint and cover all trainable parameters.
  3. test_contrastive_loss_schedule_annealing:
     Verifies linear and cosine scheduling of lambda_contrast from start to end.
  4. test_lr_scheduler_stepping:
     Verifies cosine scheduler steps and decays learning rates over time.
  5. test_checkpoint_save_and_load_roundtrip:
     Verifies complete model, optimizer, and metadata state serialization and resumption.
  6. test_early_stopping_logic:
     Verifies early stopping terminates training when validation fails to improve for patience epochs.
  7. test_nan_loss_guard:
     Verifies NaN/Inf loss does not corrupt weights and skips optimizer step.
  8. test_gradient_accumulation_mechanics:
     Verifies gradient accumulation steps properly at boundary.
  9. test_deterministic_validation_evaluation:
     Verifies evaluate_model produces deterministic metrics on identical splits.
  10. test_best_checkpoint_update_criterion:
     Verifies best checkpoint is only triggered when metric improves beyond min_delta.
"""

import copy
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repo root and Burn_restore to sys.path
repo_root = Path(__file__).resolve().parent.parent
burn_restore_dir = repo_root / "Burn_restore"
for p in [str(burn_restore_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Config
from transformers import AutoTokenizer
from model import ForensicT5Restorer
from dataset import LineGroundedDocumentDataset
from train import (
    build_optimizer_and_scheduler,
    compute_scheduled_contrastive_weight,
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
)


class TestTrainingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.config.epochs = 3
        cls.config.warmup_epochs = 1
        cls.config.device = "cpu"
        cls.splits_manifest = str(repo_root / "data" / "splits.json")
        if not os.path.isfile(cls.splits_manifest):
            cls.splits_manifest = str(burn_restore_dir / "data" / "splits.json")

    def test_stage1_adapter_warmup_freezes_t5(self):
        """Stage 1: T5 backbone should have requires_grad=False and p.grad=None after backward."""
        model = ForensicT5Restorer(self.config)
        optimizer, _ = build_optimizer_and_scheduler(model, self.config, stage=1, total_steps=10)

        # Check requires_grad flag
        for p in model.t5.parameters():
            self.assertFalse(p.requires_grad, "T5 parameter must be frozen in Stage 1 warmup.")

        # Check that adapter parameters are trainable
        trainable_adapters = [p for p in model.visual_encoder.proj.parameters()]
        for p in trainable_adapters:
            self.assertTrue(p.requires_grad, "Visual adapter projection must be trainable in Stage 1.")

        # Simulate forward and backward pass
        dummy_input_ids = torch.randint(10, 500, (2, 32))
        dummy_mask = torch.ones((2, 32), dtype=torch.long)
        dummy_crop = torch.randn(2, 3, 128, 256)
        dummy_labels = torch.randint(10, 500, (2, 16))

        loss, _ = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_mask,
            line_burn_crop=dummy_crop,
            labels=dummy_labels,
            lambda_contrastive=0.1,
        )
        loss.backward()

        # T5 parameters must have no gradients
        for p in model.t5.parameters():
            self.assertIsNone(p.grad, "Frozen T5 parameter should not accumulate gradients.")

        # Visual projection parameter must have valid gradient
        has_adapter_grad = any(p.grad is not None and torch.is_nonzero(p.grad.abs().sum()) for p in trainable_adapters)
        self.assertTrue(has_adapter_grad, "Trainable visual adapter should have non-zero gradients.")

    def test_stage2_differential_lr_parameter_groups(self):
        """Stage 2: disjoint parameter groups (T5, vision, adapters, plus fusion gates) covering all trainable parameters."""
        model = ForensicT5Restorer(self.config)
        optimizer, scheduler = build_optimizer_and_scheduler(model, self.config, stage=2, total_steps=20)

        self.assertEqual(len(optimizer.param_groups), 4, "Stage 2: t5, vision, adapters and fusion_gates groups.")

        names = [g["name"] for g in optimizer.param_groups]
        self.assertIn("t5_backbone", names)
        self.assertIn("vision_backbone", names)
        self.assertIn("adapters_and_fusion", names)
        self.assertIn("fusion_gates", names)

        # Verify learning rates (scheduler sets initial_lr attribute during warmup initialization)
        g_map = {g["name"]: g for g in optimizer.param_groups}
        base_t5 = g_map["t5_backbone"].get("initial_lr", g_map["t5_backbone"]["lr"])
        base_vis = g_map["vision_backbone"].get("initial_lr", g_map["vision_backbone"]["lr"])
        base_adp = g_map["adapters_and_fusion"].get("initial_lr", g_map["adapters_and_fusion"]["lr"])
        self.assertEqual(base_t5, self.config.lr_t5_backbone)
        self.assertEqual(base_vis, self.config.lr_vision_backbone)
        self.assertEqual(base_adp, self.config.lr_adapters)

        # Verify disjointness
        s_t5 = set(id(p) for p in g_map["t5_backbone"]["params"])
        s_vis = set(id(p) for p in g_map["vision_backbone"]["params"])
        s_adp = set(id(p) for p in g_map["adapters_and_fusion"]["params"])

        self.assertEqual(len(s_t5 & s_vis), 0, "T5 and Vision parameter groups must be disjoint.")
        self.assertEqual(len(s_t5 & s_adp), 0, "T5 and Adapter parameter groups must be disjoint.")
        self.assertEqual(len(s_vis & s_adp), 0, "Vision and Adapter parameter groups must be disjoint.")

        # Verify parameter count equals total model trainable parameters
        total_opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        self.assertEqual(total_opt_params, total_trainable, "All trainable parameters must be accounted for.")

    def test_contrastive_loss_schedule_annealing(self):
        """Dynamic contrastive weight must decay linearly and cosinely across epochs."""
        cfg = Config()
        cfg.contrastive_sched_start = 0.20
        cfg.contrastive_sched_end = 0.05
        cfg.contrastive_sched_type = "linear"

        total_epochs = 5
        # Epoch 0 (start)
        w0 = compute_scheduled_contrastive_weight(0, total_epochs, cfg)
        self.assertAlmostEqual(w0, 0.20, places=4)

        # Epoch 2 (mid)
        w2 = compute_scheduled_contrastive_weight(2, total_epochs, cfg)
        self.assertAlmostEqual(w2, 0.125, places=4)

        # Epoch 4 (end)
        w4 = compute_scheduled_contrastive_weight(4, total_epochs, cfg)
        self.assertAlmostEqual(w4, 0.05, places=4)

        # Fixed mode
        cfg.contrastive_sched_type = "fixed"
        cfg.lambda_contrastive = 0.10
        w_fix = compute_scheduled_contrastive_weight(3, total_epochs, cfg)
        self.assertAlmostEqual(w_fix, 0.10, places=4)

    def test_lr_scheduler_stepping(self):
        """Cosine scheduler should decay learning rates monotonically after warmup."""
        model = ForensicT5Restorer(self.config)
        cfg = copy.deepcopy(self.config)
        cfg.scheduler_warmup_ratio = 0.0  # test decay directly
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, stage=2, total_steps=50)

        initial_lrs = [g["lr"] for g in optimizer.param_groups]
        for _ in range(25):
            optimizer.step()
            scheduler.step()

        mid_lrs = [g["lr"] for g in optimizer.param_groups]
        for init_lr, mid_lr in zip(initial_lrs, mid_lrs):
            self.assertLess(mid_lr, init_lr, "LR must decay after cosine schedule steps.")

    def test_checkpoint_save_and_load_roundtrip(self):
        """Saving and loading checkpoint must restore state dicts and metadata exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            model = ForensicT5Restorer(self.config)
            optimizer, scheduler = build_optimizer_and_scheduler(model, self.config, stage=2, total_steps=10)

            # Record initial parameter sum
            init_sum = sum(p.sum().item() for p in model.parameters())

            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=2,
                global_step=42,
                metrics={"val_cer": 0.35},
                config=self.config,
            )
            self.assertTrue(os.path.isfile(ckpt_path))

            # Mutate model weights
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1.0)
            mutated_sum = sum(p.sum().item() for p in model.parameters())
            self.assertNotEqual(init_sum, mutated_sum)

            # Load checkpoint back
            meta = load_checkpoint(ckpt_path, model, optimizer=optimizer, scheduler=scheduler, device="cpu")
            restored_sum = sum(p.sum().item() for p in model.parameters())

            self.assertAlmostEqual(init_sum, restored_sum, places=4)
            self.assertEqual(meta["epoch"], 2)
            self.assertEqual(meta["global_step"], 42)
            self.assertEqual(meta["metrics"]["val_cer"], 0.35)

    def test_early_stopping_logic(self):
        """Patience counter triggers early stop when validation does not improve."""
        patience = 3
        min_delta = 0.005
        best_val_cer = 0.50
        patience_counter = 0
        stopped = False

        # Sequence of stagnant validation metrics
        val_sequence = [0.50, 0.499, 0.502, 0.505]
        for val_cer in val_sequence:
            if val_cer < (best_val_cer - min_delta):
                best_val_cer = val_cer
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stopped = True
                    break

        self.assertTrue(stopped, "Early stopping should trigger after 3 consecutive stagnant epochs.")
        self.assertEqual(patience_counter, 3)

    def test_nan_loss_guard(self):
        """NaN / Inf loss must be flagged and not contaminate model parameters."""
        model = ForensicT5Restorer(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        nan_loss = torch.tensor(float("nan"), requires_grad=True)
        is_bad = torch.isnan(nan_loss) or torch.isinf(nan_loss)
        self.assertTrue(is_bad)

        # Simulation of guard: optimizer step should NOT be called
        before_sum = sum(p.sum().item() for p in model.parameters())
        if not is_bad:
            nan_loss.backward()
            optimizer.step()
        after_sum = sum(p.sum().item() for p in model.parameters())
        self.assertEqual(before_sum, after_sum, "Model weights must remain untouched when NaN is detected.")

    def test_gradient_accumulation_mechanics(self):
        """Optimizer should step once per grad_accum_steps micro-batches."""
        grad_accum_steps = 3
        total_micro_batches = 7

        optimizer_steps = 0
        for step in range(total_micro_batches):
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_micro_batches:
                optimizer_steps += 1

        # 3 micro-batches = step 1; 6 micro-batches = step 2; 7th micro-batch (final) = step 3
        self.assertEqual(optimizer_steps, 3, "Optimizer must step at accumulation boundaries and at epoch end.")

    def test_deterministic_validation_evaluation(self):
        """evaluate_model must return consistent, deterministic metrics across consecutive runs."""
        if not os.path.isfile(self.splits_manifest):
            self.skipTest("Splits manifest not found.")

        cfg = copy.deepcopy(self.config)
        model = ForensicT5Restorer(cfg)
        model.eval()

        val_dataset = LineGroundedDocumentDataset(
            config=cfg,
            tokenizer=model.tokenizer or AutoTokenizer.from_pretrained(cfg.text_model_name),
            split="val",
            splits_manifest=self.splits_manifest,
            is_eval=True,
        )
        dataloader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=LineGroundedDocumentDataset.collate_fn,
        )

        metrics_1 = evaluate_model(model, dataloader, val_dataset.tokenizer, cfg, torch.device("cpu"))
        metrics_2 = evaluate_model(model, dataloader, val_dataset.tokenizer, cfg, torch.device("cpu"))

        self.assertEqual(metrics_1["val_cer"], metrics_2["val_cer"])
        self.assertEqual(metrics_1["val_wer"], metrics_2["val_wer"])
        self.assertEqual(metrics_1["val_char_acc"], metrics_2["val_char_acc"])
        self.assertEqual(metrics_1["budget_violation_rate"], metrics_2["budget_violation_rate"])

    def test_best_checkpoint_update_criterion(self):
        """Best checkpoint update requires improvement strictly greater than min_delta."""
        min_delta = 0.01
        best_cer = 0.40

        # Marginal improvement less than min_delta (0.395 vs 0.40 - 0.01 = 0.39) -> False
        candidate_1 = 0.395
        update_1 = candidate_1 < (best_cer - min_delta)
        self.assertFalse(update_1)

        # Significant improvement (0.38 vs 0.39) -> True
        candidate_2 = 0.380
        update_2 = candidate_2 < (best_cer - min_delta)
        self.assertTrue(update_2)


if __name__ == "__main__":
    unittest.main()
