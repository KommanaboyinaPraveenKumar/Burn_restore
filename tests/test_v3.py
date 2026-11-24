"""Tests for the v3 pipeline: fine visual grid, CTC reading head, v3 prompt
(2-line context + length hint), OCR noise, realistic layout and randomised burns."""

import os
import random

import numpy as np
import pytest
import torch

from config import Config
from line_data import (PARA, alpha_x, apply_burn_v3, build_prompt, ocr_noise, prompt_from_record,
                       render_page)
from model import (CTC_CHARSET, ForensicT5Restorer, MultiScaleForensicEncoder, apply_saved_architecture,
                   decode_ctc_greedy, encode_ctc_text)

FONT = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "times.ttf")
needs_font = pytest.mark.skipif(not os.path.isfile(FONT), reason="Times font not available")


def test_fine_grid_gives_128_tokens():
    enc = MultiScaleForensicEncoder(d_model=64, pretrained=False, grid="fine").eval()
    tokens, fmap = enc(torch.randn(2, 3, 128, 256))
    assert tokens.shape == (2, 128, 64)
    assert fmap.shape == (2, 512, 8, 16)
    assert enc.last_h2.shape == (2, 128, 16, 32)
    coarse = MultiScaleForensicEncoder(d_model=64, pretrained=False, grid="coarse").eval()
    assert coarse(torch.randn(1, 3, 128, 256))[0].shape == (1, 32, 64)


def test_ctc_encode_decode_roundtrip():
    ids = encode_ctc_text("Hi, you!")
    assert all(1 <= i <= len(CTC_CHARSET) for i in ids)
    # build a perfect frame sequence with blanks between repeats
    T = 2 * len(ids) + 1
    lp = torch.full((1, T, len(CTC_CHARSET) + 1), -10.0)
    for t in range(T):
        lp[0, t, 0 if t % 2 == 0 else ids[t // 2]] = 0.0
    assert decode_ctc_greedy(lp) == ["Hi, you!"]


def test_v3_prompt_has_context_and_length():
    p = build_prompt("the quick bro", "line above", "line below", budget=12.4,
                     prev2_prefix="two above", next2_prefix="two below", style="v3")
    assert "two above || line above" in p
    assert "line below || two below" in p
    assert p.rstrip().endswith("length: 12")
    assert "line: the quick bro<extra_id_0>" in p
    # v2 stays byte-identical to the legacy template
    assert build_prompt("abc", "p", "n") == "previous: p line: abc<extra_id_0> next: n"


def test_prompt_from_record_applies_noise_only_to_context():
    r = {"prefix": "aaaa", "prev_prefix": "bbbb", "next_prefix": "cccc", "budget": 5.0, "target": "zz"}
    p = prompt_from_record(r, style="v3", noise=lambda t: t.upper())
    assert "AAAA" in p and "BBBB" in p and "zz" not in p


def test_ocr_noise_rate():
    rng = random.Random(0)
    text = "the quick brown fox jumps over the lazy dog " * 50
    assert ocr_noise(text, rng, 0.0) == text
    noisy = ocr_noise(text, rng, 0.05)
    from metrics import _levenshtein
    rate = _levenshtein(noisy, text) / len(text)
    assert 0.01 < rate < 0.12


@needs_font
def test_layout_paragraphs_and_hyphenation():
    words = (("extraordinarily " * 3 + "short words here and there ").split() * 6) + [PARA] + "a b c d e f".split() * 20
    page, used = render_page(words, FONT, random.Random(1),
                             layout={"indent": 2.0, "hyphen_p": 1.0, "justify": True})
    assert used > 0
    assert any(l.endswith("-") for l in page.lines), "expected at least one hyphenated line"
    assert any(page.meta["para_end"]), "paragraph break should end a line early"
    for text, boxes in zip(page.lines, page.char_boxes):
        assert "".join(ch for ch, _, _ in boxes) == text
        xs = [x1 for _, x1, _ in boxes]
        assert xs == sorted(xs)


def test_burn_v3_and_alpha_inverse():
    img = np.full((64, 300, 3), 240, np.uint8)
    boundary = np.full(64, 200)
    out, params = apply_burn_v3(img, boundary, np.random.RandomState(3))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[:, 250].mean() < out[:, 50].mean()  # charred side is darker
    T, g = params["T"], params["gamma"]
    for a in (0.2, 0.5, 0.8):
        x = alpha_x(200, T, g, a)
        assert ((x - (200 - T)) / T) ** g == pytest.approx(a, abs=1e-6)


def test_legacy_defaults_for_old_checkpoints():
    cfg = Config()
    apply_saved_architecture(cfg, {"use_vision": True})
    assert (cfg.fusion_style, cfg.vision_grid, cfg.use_reading_head, cfg.prompt_style) == \
        ("legacy", "coarse", False, "v2")


def test_full_v3_model_forward_with_ctc():
    cfg = Config()
    cfg.pretrained_vision = False
    m = ForensicT5Restorer(cfg).train()
    b = 2
    ids = torch.randint(5, 100, (b, 20))
    ctc = [encode_ctc_text("hello there"), encode_ctc_text("abc")]
    T = torch.zeros(b, max(map(len, ctc)), dtype=torch.long)
    for i, c in enumerate(ctc):
        T[i, :len(c)] = torch.tensor(c)
    loss, _ = m(input_ids=ids, attention_mask=torch.ones_like(ids), line_burn_crop=torch.randn(b, 3, 128, 256),
                labels=torch.randint(5, 100, (b, 6)), ctc_targets=T,
                ctc_lengths=torch.tensor([len(c) for c in ctc]))
    loss.backward()
    assert m.last_read_loss > 0
    assert m.reading_head.out.weight.grad is not None
    assert len(m.read_crop(torch.randn(1, 3, 128, 256))) == 1
