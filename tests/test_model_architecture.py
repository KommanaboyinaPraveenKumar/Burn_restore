"""
test_model_architecture.py
==========================
Comprehensive unit and integration test suite for Phase 2:
Multimodal Model Architecture, Forensic Visual Grounding, and Physical Constraints.

Validates:
  1. Native ResNet-34 feature extraction & multi-scale layer3/layer4 fusion.
  2. 2D spatial positional embeddings (4 rows x 8 cols = 32 tokens).
  3. Gated CrossModalFusion forward pass, attention shape, and learnable gate.
  4. ContrastiveStrokeAligner (InfoNCE loss, confidence weighting, B=1 and B>1).
  5. End-to-end multimodal forward pass and loss computation.
  6. Gradient backpropagation through visual backbone, cross-attention, gate, and T5.
  7. PhysicalBudgetLogitsProcessor SentencePiece incremental character counting.
  8. PhysicalBudgetLogitsProcessor escalating EOS bias and hard cutoff.
  9. PhysicalBudgetLogitsProcessor beam search expansion mapping.
  10. Autoregressive greedy decoding (num_beams=1) and beam search (num_beams=4).
  11. generate_topk candidate generation.
  12. Metric repairs: compute_token_accuracy rejects min_len truncation flaw.
  13. New metrics: compute_wer, compute_character_accuracy, compute_word_accuracy,
      compute_exact_match_line_accuracy, compute_exact_match_document_accuracy.
  14. CPU device compatibility and stability.
"""

import os
import sys
import unittest
import torch
from transformers import AutoTokenizer

# Path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
burn_restore_dir = os.path.join(repo_root, "Burn_restore")

for p in [burn_restore_dir, repo_root, current_dir]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from config import Config
from model import (
    NativeResNet34,
    MultiScaleForensicEncoder,
    GatedCrossModalFusion,
    ContrastiveStrokeAligner,
    PhysicalBudgetLogitsProcessor,
    ForensicT5Restorer,
    GroundedDocumentRestorer,
)
import metrics


class TestModelArchitecture(unittest.TestCase):
    """Exhaustive test suite for Phase 2 multimodal model components."""

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.config.pretrained_vision = False
        cls.config.device = "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.config.text_model_name)
        cls.d_model = 512

    def test_01_native_resnet34_shapes(self):
        """Verify Native ResNet-34 extracts layer3 and layer4 multi-scale maps."""
        resnet = NativeResNet34(in_channels=3)
        x = torch.randn(2, 3, 128, 256)
        f3, f4 = resnet.forward_multiscale(x)
        self.assertEqual(f3.shape, (2, 256, 8, 16), "Layer3 must be (B, 256, 8, 16)")
        self.assertEqual(f4.shape, (2, 512, 4, 8), "Layer4 must be (B, 512, 4, 8)")

    def test_02_multiscale_visual_encoder_and_spatial_pos(self):
        """Verify MultiScaleForensicEncoder produces 32 tokens with 2D spatial embeddings."""
        encoder = MultiScaleForensicEncoder(d_model=self.d_model, pretrained=False, use_multi_scale=True)
        x = torch.randn(2, 3, 128, 256)
        tokens, fmap = encoder(x)
        self.assertEqual(tokens.shape, (2, 32, self.d_model), "Visual tokens must be (B, 32, 512)")
        self.assertEqual(fmap.shape, (2, 512, 4, 8), "Spatial feature map must be (B, 512, 4, 8)")
        self.assertTrue(torch.isfinite(tokens).all(), "Visual tokens must contain finite values")

    def test_03_gated_cross_modal_fusion(self):
        """Verify GatedCrossModalFusion applies cross-attention with learnable sigmoid gate."""
        fusion = GatedCrossModalFusion(d_model=self.d_model, num_heads=4, dropout=0.0)
        text_embeds = torch.randn(2, 16, self.d_model)
        vis_tokens = torch.randn(2, 32, self.d_model)
        out = fusion(text_embeds, vis_tokens)
        self.assertEqual(out.shape, (2, 16, self.d_model), "Fused text shape must match input text shape")
        self.assertTrue(fusion.gate.requires_grad, "Fusion gate parameter must require grad")

    def test_04_contrastive_stroke_aligner(self):
        """Verify ContrastiveStrokeAligner calculates valid InfoNCE loss."""
        aligner = ContrastiveStrokeAligner(d_model=self.d_model, proj_dim=128, temperature=0.07)
        fmap = torch.randn(4, 512, 4, 8)
        target_embeds = torch.randn(4, self.d_model)
        conf = torch.tensor([1.0, 0.8, 0.9, 0.7])

        loss_batch = aligner(fmap, target_embeds, conf)
        self.assertTrue(torch.isfinite(loss_batch), "Batch InfoNCE loss must be finite")
        self.assertGreater(loss_batch.item(), 0.0, "Loss must be positive")

        # Single sample test (B=1)
        loss_single = aligner(fmap[:1], target_embeds[:1])
        self.assertTrue(torch.isfinite(loss_single), "Single sample contrastive loss must be finite")

    def test_05_multimodal_forward_and_loss(self):
        """Verify full ForensicT5Restorer forward pass returns valid CE + contrastive loss."""
        model = ForensicT5Restorer(self.config)
        b = 2
        input_ids = torch.randint(10, 1000, (b, 24))
        mask = torch.ones((b, 24), dtype=torch.long)
        crop = torch.randn(b, 3, 128, 256)
        labels = torch.randint(10, 1000, (b, 12))

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=mask,
            line_burn_crop=crop,
            labels=labels,
        )
        self.assertTrue(torch.isfinite(loss), "Combined loss must be finite")
        self.assertEqual(logits.shape, (b, 12, model.t5.config.vocab_size), "Logits must match target length & vocab")

    def test_06_gradient_propagation(self):
        """Verify gradients propagate backwards through visual backbone, gate, and cross-attention."""
        model = ForensicT5Restorer(self.config)
        model.train()
        b = 2
        input_ids = torch.randint(10, 1000, (b, 16))
        mask = torch.ones((b, 16), dtype=torch.long)
        crop = torch.randn(b, 3, 128, 256)
        labels = torch.randint(10, 1000, (b, 8))

        loss, _ = model(input_ids=input_ids, attention_mask=mask, line_burn_crop=crop, labels=labels)
        loss.backward()

        stem_grad = model.visual_encoder.stem[0].weight.grad
        self.assertIsNotNone(stem_grad, "Visual stem must receive gradients")
        self.assertGreater(stem_grad.norm().item(), 0.0, "Visual stem gradient norm must be > 0")

        fusion = model.cross_modal_fusion
        gate = fusion.attn_gate if hasattr(fusion, "attn_gate") else fusion.gate
        self.assertIsNotNone(gate.grad, "Fusion gate must receive gradients")

        cross_attn_grad = model.cross_modal_fusion.cross_attn.in_proj_weight.grad
        self.assertIsNotNone(cross_attn_grad, "Cross-attention must receive gradients")

    def test_07_physical_budget_processor_character_counting(self):
        """Verify PhysicalBudgetLogitsProcessor counts subword characters and applies escalating EOS bias."""
        budgets = torch.tensor([5.0])  # 5 characters budget
        proc = PhysicalBudgetLogitsProcessor(
            tokenizer=self.tokenizer,
            budgets=budgets,
            eos_token_id=self.tokenizer.eos_token_id,
            penalty_weight=10.0,
            margin_buffer=0.0,
            hard_cutoff_excess=5.0,
        )

        # 1. Under budget string ("dog" = 3 characters)
        under_tokens = self.tokenizer.encode("dog", return_tensors="pt")
        scores_under = torch.zeros(1, self.tokenizer.vocab_size)
        out_under = proc(under_tokens, scores_under.clone())
        self.assertEqual(out_under[0, self.tokenizer.eos_token_id].item(), 0.0, "Under-budget string must receive zero bias")

        # 2. Moderately over budget string ("elephant" = 8 chars, excess = 3)
        over_tokens = self.tokenizer.encode("elephant", return_tensors="pt")
        scores_over = torch.zeros(1, self.tokenizer.vocab_size)
        out_over = proc(over_tokens, scores_over.clone())
        self.assertGreater(out_over[0, self.tokenizer.eos_token_id].item(), 0.0, "Over-budget string must receive positive EOS bias")
        self.assertLess(out_over[0, 100].item(), 0.0, "Non-EOS tokens must receive negative penalty")

        # 3. Severely over budget string (excess >= 5) -> triggers hard cutoff
        huge_tokens = self.tokenizer.encode("supercalifragilisticexpialidocious", return_tensors="pt")
        scores_huge = torch.zeros(1, self.tokenizer.vocab_size)
        out_huge = proc(huge_tokens, scores_huge.clone())
        self.assertEqual(out_huge[0, self.tokenizer.eos_token_id].item(), 100.0, "Hard cutoff sets EOS to 100.0")
        self.assertEqual(out_huge[0, 100].item(), -10000.0, "Hard cutoff sets non-EOS to -10000.0")

    def test_08_physical_budget_processor_beam_expansion(self):
        """Verify PhysicalBudgetLogitsProcessor correctly maps beams across batch items."""
        budgets = torch.tensor([4.0, 30.0])  # Sample 0 budget = 4, Sample 1 budget = 30
        proc = PhysicalBudgetLogitsProcessor(
            tokenizer=self.tokenizer,
            budgets=budgets,
            eos_token_id=self.tokenizer.eos_token_id,
            penalty_weight=10.0,
            margin_buffer=0.0,
        )

        long_str_tokens = self.tokenizer.encode("internationalization", return_tensors="pt")  # ~20 chars
        # 4 beams per sample -> 8 beam rows
        batch_beams = long_str_tokens.repeat(8, 1)
        scores = torch.zeros(8, self.tokenizer.vocab_size)
        out = proc(batch_beams, scores)

        # Beams 0..3 (Sample 0, budget 4) should be penalized
        for b in range(4):
            self.assertGreater(out[b, self.tokenizer.eos_token_id].item(), 0.0, f"Beam {b} must have EOS bias")

        # Beams 4..7 (Sample 1, budget 30) should NOT be penalized (20 < 30)
        for b in range(4, 8):
            self.assertEqual(out[b, self.tokenizer.eos_token_id].item(), 0.0, f"Beam {b} must have zero EOS bias")

    def test_09_generate_greedy_and_beam(self):
        """Verify model.generate() works with greedy, beam search, and physical grounding."""
        model = ForensicT5Restorer(self.config)
        b = 2
        input_ids = torch.randint(10, 1000, (b, 16))
        mask = torch.ones((b, 16), dtype=torch.long)
        crop = torch.randn(b, 3, 128, 256)
        budgets = torch.tensor([8.0, 12.0])

        # 1. Greedy decoding (num_beams=1)
        gen_greedy = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            line_burn_crop=crop,
            line_budget=budgets,
            num_beams=1,
            max_len=8,
        )
        self.assertEqual(gen_greedy.shape[0], b, "Greedy batch size must match b")

        # 2. Beam search (num_beams=3)
        gen_beam = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            line_burn_crop=crop,
            line_budget=budgets,
            num_beams=3,
            max_len=8,
        )
        self.assertEqual(gen_beam.shape[0], b, "Beam batch size must match b")

    def test_10_generate_topk(self):
        """Verify generate_topk returns top-k candidate beams per sample."""
        model = ForensicT5Restorer(self.config)
        b = 2
        input_ids = torch.randint(10, 1000, (b, 16))
        mask = torch.ones((b, 16), dtype=torch.long)
        crop = torch.randn(b, 3, 128, 256)

        cands = model.generate_topk(input_ids, mask, crop, top_k=3, max_len=6)
        self.assertEqual(len(cands), b, "Must return b candidate lists")
        self.assertEqual(len(cands[0]), 3, "Each sample must have 3 candidates")

    def test_11_metric_token_accuracy_truncation_fix(self):
        """Verify compute_token_accuracy rejects min_len truncation and strictly penalizes missing tokens."""
        pad_id = 0
        # Target has 10 tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        target = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, pad_id, pad_id])

        # Prediction has only 1 token: [1]
        pred_incomplete = torch.tensor([1, pad_id])
        acc_incomplete = metrics.compute_token_accuracy(pred_incomplete, target, pad_id=pad_id)
        # In the old buggy implementation, this returned 1.0 (100%)!
        self.assertAlmostEqual(acc_incomplete, 0.10, places=2, msg="Incomplete prediction must yield 0.10, NOT 1.0")

        # Prediction matches completely
        pred_complete = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        acc_complete = metrics.compute_token_accuracy(pred_complete, target, pad_id=pad_id)
        self.assertEqual(acc_complete, 1.0, "Complete matching prediction must yield 1.0")

    def test_12_repaired_text_metrics(self):
        """Verify CER, WER, Character Accuracy, Word Accuracy, and Exact-Match Line Accuracy."""
        pred = "The quick brown fox"
        target = "The quick brown fox"
        self.assertEqual(metrics.compute_cer(pred, target), 0.0)
        self.assertEqual(metrics.compute_wer(pred, target), 0.0)
        self.assertEqual(metrics.compute_character_accuracy(pred, target), 1.0)
        self.assertEqual(metrics.compute_word_accuracy(pred, target), 1.0)
        self.assertEqual(metrics.compute_exact_match_line_accuracy(pred, target), 1.0)

        # Partial substitution
        pred_diff = "The quick blue fox"
        self.assertGreater(metrics.compute_cer(pred_diff, target), 0.0)
        self.assertAlmostEqual(metrics.compute_wer(pred_diff, target), 0.25, places=2)
        self.assertAlmostEqual(metrics.compute_word_accuracy(pred_diff, target), 0.75, places=2)
        self.assertEqual(metrics.compute_exact_match_line_accuracy(pred_diff, target), 0.0)

        # Document exact match
        self.assertEqual(metrics.compute_exact_match_document_accuracy(["line 1", "line 2"], ["line 1", "line 2"]), 1.0)
        self.assertEqual(metrics.compute_exact_match_document_accuracy(["line 1", "mismatch"], ["line 1", "line 2"]), 0.0)

    def test_13_backward_compatibility_alias(self):
        """Verify GroundedDocumentRestorer is an exact alias for ForensicT5Restorer."""
        self.assertIs(GroundedDocumentRestorer, ForensicT5Restorer)


if __name__ == "__main__":
    unittest.main()

