"""Tests for the calibrated upper budget: cut-offset regressor, hard look-ahead
budget processor and the upper-bound helper."""

import pytest
import torch
from transformers import AutoTokenizer

from config import Config
from line_data import extract_infill
from model import CutOffsetRegressor, ForensicT5Restorer, PhysicalBudgetLogitsProcessor, upper_budget


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("t5-small")


def test_cut_regressor_shape():
    m = CutOffsetRegressor().eval()
    out = m(torch.randn(3, 3, 128, 256))
    assert out.shape == (3,)


def test_upper_budget():
    assert upper_budget(10.0, None) == 10.0
    assert upper_budget(10.0, {"ub_ratio": 1.3, "ub_add": 1.0}) == pytest.approx(14.0)


def test_hard_mode_masks_tokens_that_would_overflow(tok):
    vocab = 32128
    proc = PhysicalBudgetLogitsProcessor(tok, torch.tensor([6.0]), eos_token_id=tok.eos_token_id, mode="hard")
    prefix = tok("hello", add_special_tokens=False).input_ids     # 5 chars already generated
    ids = torch.tensor([[0] + prefix])
    scores = proc(ids, torch.zeros(1, vocab))
    allowed = torch.isfinite(scores[0])
    assert allowed[tok.eos_token_id]
    for t in range(len(tok)):
        if allowed[t] and t not in tok.all_special_ids:
            piece = tok.convert_ids_to_tokens(t).replace("▁", " ")
            assert 5 + len(piece) <= 6
    word = tok(" world", add_special_tokens=False).input_ids[0]
    assert not allowed[word]


def test_hard_mode_never_exceeds_bound_in_beam_search(tok):
    cfg = Config()
    cfg.pretrained_vision = False
    cfg.use_vision = False
    cfg.use_reading_head = False
    m = ForensicT5Restorer(cfg).eval()
    ub = torch.tensor([2.0, 4.0, 6.5, 9.0, 13.0, 20.0])
    enc = tok(["line: the quick brown fox jumps over the<extra_id_0> next: lazy dog"] * len(ub), return_tensors="pt")
    out = m.generate_topk(enc.input_ids, enc.attention_mask, torch.zeros(len(ub), 3, 128, 256), line_budget=ub,
                          top_k=4, max_len=16, budget_mode="hard")
    for b, cands in enumerate(out):
        for c in cands:
            assert len(extract_infill(tok.decode(c, skip_special_tokens=False))) <= ub[b].item()


def test_v3e_prompt_uses_estimated_length():
    from line_data import prompt_from_record
    r = {"prefix": "abc ", "prev_prefix": "p", "next_prefix": "n", "budget": 10.2, "budget_est": 13.6, "target": "x"}
    assert prompt_from_record(r, style="v3").rstrip().endswith("length: 10")
    assert prompt_from_record(r, style="v3e").rstrip().endswith("length: 14")
    with pytest.raises(KeyError):
        prompt_from_record({k: v for k, v in r.items() if k != "budget_est"}, style="v3e")
