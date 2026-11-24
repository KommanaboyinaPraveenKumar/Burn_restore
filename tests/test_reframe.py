"""Tests for the reframed method: residual fusion, read/inferred split,
visible-only contrastive loss and legacy checkpoint compatibility."""

import os

import pytest
import torch

from config import Config
from line_data import count_visible, split_visible, visible_limit_x
from model import ForensicT5Restorer, ResidualGatedFusion, apply_saved_architecture


def test_split_visible_handles_leading_space():
    read, inf = split_visible(" ed to", 3)  # space + 'e' + 'd' visible
    assert read == "ed"
    assert read + inf == "ed to"
    assert split_visible("abc", 0) == ("", "abc")
    assert split_visible("abc", 10) == ("abc", "")


def test_count_visible_stops_at_limit():
    boxes = [("a", 0, 10), ("b", 10, 20), ("c", 20, 30), ("d", 30, 40)]
    # prefix = 'a'; target centres at 15, 25, 35; limit 30 -> 'b', 'c' visible
    assert count_visible(boxes, 1, 30.0) == 2
    assert visible_limit_x(100, 55) == pytest.approx(89.0)


def test_residual_fusion_is_identity_at_init():
    fus = ResidualGatedFusion(d_model=32, num_heads=4, embed_scale=16.0).eval()
    text = torch.randn(2, 5, 32) * 16
    vis = torch.randn(2, 7, 32)
    assert torch.allclose(fus(text, vis), text)


@pytest.fixture(scope="module")
def small_model():
    cfg = Config()
    cfg.pretrained_vision = False
    return ForensicT5Restorer(cfg)


def test_contrastive_uses_only_visible_ids(small_model):
    m = small_model.train()
    b = 4
    ids = torch.randint(5, 100, (b, 12))
    mask = torch.ones_like(ids)
    crop = torch.randn(b, 3, 128, 256)
    labels = torch.randint(5, 100, (b, 6))
    pad = m.pad_id
    none_visible = torch.full((b, 3), pad)
    m(input_ids=ids, attention_mask=mask, line_burn_crop=crop, labels=labels,
      lambda_contrastive=0.2, visible_ids=none_visible)
    assert m.last_contrast_loss == 0.0
    some_visible = none_visible.clone()
    some_visible[:3, 0] = torch.tensor([10, 20, 30])
    loss, _ = m(input_ids=ids, attention_mask=mask, line_burn_crop=crop, labels=labels,
                lambda_contrastive=0.2, visible_ids=some_visible)
    assert m.last_contrast_loss > 0.0
    loss.backward()
    assert m.cross_modal_fusion.attn_gate.grad is not None


def test_apply_saved_architecture_marks_old_checkpoints_legacy():
    cfg = Config()
    apply_saved_architecture(cfg, {"use_vision": True})
    assert cfg.fusion_style == "legacy"
    cfg2 = Config()
    apply_saved_architecture(cfg2, {"use_vision": True, "fusion_style": "flamingo"})
    assert cfg2.fusion_style == "flamingo"
    cfg3 = Config()
    apply_saved_architecture(cfg3, {})
    assert cfg3.fusion_style == "flamingo"


LEGACY = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "full_v1", "best_checkpoint.pt")


@pytest.mark.skipif(not os.path.isfile(LEGACY), reason="legacy checkpoint not present")
def test_legacy_checkpoint_loads_strict():
    ckpt = torch.load(LEGACY, map_location="cpu", weights_only=False)
    cfg = Config()
    cfg.pretrained_vision = False
    apply_saved_architecture(cfg, ckpt.get("config_dict", {}))
    m = ForensicT5Restorer(cfg)
    m.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert m.fusion_style == "legacy"
