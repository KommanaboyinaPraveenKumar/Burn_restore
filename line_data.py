"""
line_data.py
============
Single source of truth for everything that must be identical between training,
evaluation and inference:

  * text normalisation          (``normalize_line_text``)
  * prompt construction         (``build_prompt`` / ``build_target``)
  * forensic line-crop geometry (``crop_line`` / ``crop_to_tensor``)
  * physical character budget   (``estimate_budget``)

It also contains the synthetic page renderer and the burn/split logic used to
build the line-level training cache (see ``scripts/build_line_dataset.py``).

Design notes
------------
* The burn is split at *character* level, and the cut point sits inside the
  ghost-fade transition zone (where ink is ~50% charred).  Characters to the
  right of the cut are the restoration target, so faint ghost strokes of the
  first target characters are visible in the crop.  This is the visual evidence
  the vision stream is supposed to exploit.
* The prompt includes the surviving prefix of the previous and next lines.
  Because prose wraps, the next line's prefix continues the sentence that the
  burnt line ends.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Constants shared by train / eval / inference
# --------------------------------------------------------------------------- #
CROP_H, CROP_W = 128, 256          # model input (H, W)
CROP_LEFT_PX = 200                 # px of surviving text kept left of the boundary
CROP_RIGHT_PX = 56                 # px kept right of the boundary (charcoal)
CROP_PAD_Y = 8                     # vertical padding around the line band
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_QUOTE_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def normalize_line_text(text: str) -> str:
    """ASCII-folds text so every character is representable by the T5 vocab."""
    for k, v in _QUOTE_MAP.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[{}<>\\^~`|_]", "", text)   # chars T5 maps to <unk>
    return re.sub(r"\s+", " ", text).strip()


def build_prompt(
    prefix: str,
    prev_prefix: str = "",
    next_prefix: str = "",
    budget: Optional[float] = None,
    prev2_prefix: str = "",
    next2_prefix: str = "",
    style: str = "v2",
) -> str:
    """T5 span-infilling prompt.  ``prefix`` keeps its trailing space (if any) so
    the model can tell a mid-word cut from a word-boundary cut.

    style "v2": one line of context above/below (legacy checkpoints).
    style "v3": two lines above/below and an approximate length hint
                (``length:`` = estimated number of missing characters).
    style "v3e": same template; the caller supplies the estimate computed with
                the predicted cut position (see prompt_from_record).
    """
    if style == "v2":
        return (
            f"previous: {prev_prefix.strip()} "
            f"line: {prefix}<extra_id_0> "
            f"next: {next_prefix.strip()}"
        )
    length = f" length: {int(round(budget))}" if budget is not None else ""
    return (
        f"previous: {prev2_prefix.strip()} || {prev_prefix.strip()} "
        f"line: {prefix}<extra_id_0> "
        f"next: {next_prefix.strip()} || {next2_prefix.strip()}{length}"
    )


def prompt_from_record(r: Dict, style: str = "v2", noise=None) -> str:
    """Builds the prompt for a cached record; ``noise`` (callable) perturbs the
    context strings to simulate OCR errors."""
    f = noise or (lambda t: t)
    # "v3e" = v3 template with the length hint from the predicted cut position
    # (scripts/add_estimated_budget.py); "v3" keeps the fixed-offset estimate
    budget = r["budget_est"] if style == "v3e" else r.get("budget")
    return build_prompt(
        f(r["prefix"]), f(r.get("prev_prefix", "")), f(r.get("next_prefix", "")),
        budget=budget, prev2_prefix=f(r.get("prev2_prefix", "")),
        next2_prefix=f(r.get("next2_prefix", "")), style=style,
    )


# OCR-style confusions used to perturb context during training / robustness eval
_OCR_CONFUSIONS = {
    "m": ["rn", "in"], "rn": ["m"], "l": ["1", "I"], "I": ["l", "1"], "e": ["c", "o"],
    "c": ["e"], "o": ["0", "a"], "a": ["o"], "h": ["b", "li"], "n": ["u", "ri"],
    "u": ["n"], "i": ["l", "1"], "t": ["f", "l"], "f": ["t"], ",": ["."], ".": [","],
    "s": ["5"], "S": ["5"], "B": ["8"], "g": ["q"], "d": ["cl"], "w": ["vv"],
}


def ocr_noise(text: str, rng: random.Random, p: float) -> str:
    """Applies per-character OCR-like errors (confusion, deletion, insertion)
    with total probability ``p`` per character."""
    if p <= 0 or not text:
        return text
    out, i = [], 0
    while i < len(text):
        if rng.random() < p:
            two = text[i:i + 2]
            r = rng.random()
            if two in _OCR_CONFUSIONS and r < 0.3:
                out.append(rng.choice(_OCR_CONFUSIONS[two])); i += 2; continue
            if text[i] in _OCR_CONFUSIONS and r < 0.7:
                out.append(rng.choice(_OCR_CONFUSIONS[text[i]])); i += 1; continue
            if r < 0.85:
                i += 1; continue                      # deletion
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz .,'"))  # insertion
        out.append(text[i]); i += 1
    return "".join(out)


def build_target(target: str) -> str:
    return f"<extra_id_0>{target}<extra_id_1>"


def extract_infill(raw_decoded: str) -> str:
    """Returns the text between <extra_id_0> and the next sentinel / EOS."""
    text = raw_decoded
    if "<extra_id_0>" in text:
        text = text.split("<extra_id_0>", 1)[1]
    text = re.split(r"<extra_id_\d+>|</s>", text, maxsplit=1)[0]
    return text.replace("<pad>", "").strip()


# --------------------------------------------------------------------------- #
# Crop
# --------------------------------------------------------------------------- #
def crop_line(image_bgr: np.ndarray, y0: int, y1: int, b_x: int) -> np.ndarray:
    """Extracts the forensic burn-edge crop for one line.

    Returns an RGB uint8 array of shape (CROP_H, CROP_W, 3).
    """
    h, w = image_bgr.shape[:2]
    cy0, cy1 = max(0, y0 - CROP_PAD_Y), min(h, y1 + CROP_PAD_Y)
    cx0, cx1 = b_x - CROP_LEFT_PX, b_x + CROP_RIGHT_PX
    # Pad with white on the left / charcoal-ish on the right so geometry is fixed
    pad_l = max(0, -cx0)
    pad_r = max(0, cx1 - w)
    region = image_bgr[cy0:cy1, max(0, cx0):min(w, cx1)]
    if region.size == 0:
        return np.full((CROP_H, CROP_W, 3), 255, dtype=np.uint8)
    if pad_l or pad_r:
        region = cv2.copyMakeBorder(region, 0, 0, pad_l, pad_r, cv2.BORDER_REPLICATE)
    region = cv2.resize(region, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(region, cv2.COLOR_BGR2RGB)


def crop_to_tensor(crop_rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) -> ImageNet-normalised float tensor (3, H, W)."""
    x = crop_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------- #
# Physical budget
# --------------------------------------------------------------------------- #
GHOST_CUT_PX = 27  # legibility cut sits ~half a transition zone left of the char boundary
# A target character still carries visible stroke evidence if its centre lies
# left of the point where the ghost fade reaches this alpha (0 = clean, 1 = charred).
VISIBLE_ALPHA = 0.8


def visible_limit_x(b_x: float, transition_px: float) -> float:
    """x position where the ghost fade reaches VISIBLE_ALPHA."""
    return b_x - (1.0 - VISIBLE_ALPHA) * transition_px


def count_visible(boxes: List[Tuple[str, float, float]], n_prefix: int, limit_x: float) -> int:
    """Number of *target* characters (after the first n_prefix) whose centre is
    left of limit_x, i.e. that are faint but still legible in the crop."""
    n = 0
    for ch, x1, x2 in boxes[n_prefix:]:
        if (x1 + x2) / 2.0 < limit_x:
            n += 1
        else:
            break
    return n


def split_visible(target: str, visible_chars: int) -> Tuple[str, str]:
    """Splits a raw target (as stored, possibly with a leading space) into the
    stripped (read-from-strokes, inferred-from-context) parts, so that
    ``read + inferred == target.strip()``."""
    lead = len(target) - len(target.lstrip())
    t = target.strip()
    k = max(0, min(int(visible_chars) - lead, len(t)))
    return t[:k], t[k:]


def estimate_budget(b_x: int, ink_x0: int, page_w: int, prefix_len: int,
                    ghost_cut_px: float = None) -> float:
    """Upper bound on the number of missing characters.

    Assumes symmetric page margins (right text edge = page_w - ink_x0) and a
    glyph width calibrated on the surviving (legible) prefix.
    """
    prefix_len = max(prefix_len, 1)
    cut_x = b_x - (GHOST_CUT_PX if ghost_cut_px is None else ghost_cut_px)
    char_w = float(np.clip(max(cut_x - ink_x0, 1) / prefix_len, 5.0, 40.0))
    right_edge = page_w - ink_x0
    return max(0.0, (right_edge - cut_x) / char_w)


# --------------------------------------------------------------------------- #
# Synthetic page rendering
# --------------------------------------------------------------------------- #
@dataclass
class RenderedPage:
    image_bgr: np.ndarray
    lines: List[str]
    # per line: list of (char, x1, x2); y band per line
    char_boxes: List[List[Tuple[str, float, float]]]
    bands: List[Tuple[int, int]]
    ink_x0: int
    meta: Dict = field(default_factory=dict)


def _load_font(path: str, size: int):
    from PIL import ImageFont
    return ImageFont.truetype(path, size)


PARA = "<P>"  # paragraph-break token in the word stream

_VOWELS = set("aeiouyAEIOUY")


def _hyphen_split(word: str, font, avail: float) -> Optional[int]:
    """Largest split k (>=2 chars kept, >=3 moved) such that word[:k] + '-'
    fits in ``avail`` px.  Prefers vowel|consonant boundaries."""
    core = word.rstrip(".,;:!?'\")\"")
    if len(core) < 6 or not core.isalpha():
        return None
    fits = [k for k in range(len(core) - 3, 1, -1) if font.getlength(word[:k] + "-") <= avail]
    if not fits:
        return None
    for k in fits:
        if word[k - 1] in _VOWELS and word[k] not in _VOWELS:
            return k
    return fits[0]


def render_page(
    words: Sequence[str],
    font_path: str,
    rng: random.Random,
    page_size: Tuple[int, int] = (1024, 1024),
    layout: Optional[Dict] = None,
    scale: float = 1.0,
) -> Tuple[RenderedPage, int]:
    """Word-wraps ``words`` onto a page.  Returns the page and the number of
    words consumed.

    ``layout`` (None = legacy behaviour) may contain
      indent      paragraph first-line indent in em (0 = none)
      hyphen_p    probability of hyphenating a word that does not fit
      justify     right-justify all but paragraph-final lines
    Paragraph breaks are marked by ``PARA`` tokens in ``words``.
    """
    from PIL import Image, ImageDraw

    # ``scale`` renders the same layout at higher resolution (for printing);
    # all geometry scales, the random draws do not change.
    w, h = int(page_size[0] * scale), int(page_size[1] * scale)
    font_size = rng.randint(18, 28)
    margin = int(rng.randint(70, 150) * scale)
    top = int(rng.randint(60, 120) * scale)
    spacing = int(font_size * rng.uniform(1.35, 1.8) * scale)
    font_size = int(round(font_size * scale))
    font = _load_font(font_path, font_size)
    paper = tuple(int(c) for c in np.clip(np.array([250, 248, 240]) + rng.randint(-12, 5), 200, 255))
    ink = rng.randint(5, 45)
    lay = layout or {}
    indent_px = float(lay.get("indent", 0.0)) * font_size
    hyphen_p = float(lay.get("hyphen_p", 0.0))
    justify = bool(lay.get("justify", False))

    img = Image.new("RGB", (w, h), paper)
    draw = ImageDraw.Draw(img)
    space_w = font.getlength(" ")
    max_x = w - margin

    words = list(words)
    lines: List[str] = []
    char_boxes: List[List[Tuple[str, float, float]]] = []
    bands: List[Tuple[int, int]] = []
    para_end_flags: List[bool] = []
    y = top
    i = 0
    new_para = layout is not None  # first line of a page starts a paragraph in v3 layout
    ascent, descent = font.getmetrics()
    while i < len(words) and y + ascent + descent < h - 60 * scale:
        if words[i] == PARA:
            new_para = True
            i += 1
            continue
        x0 = margin + (indent_px if new_para else 0.0)
        line_words: List[str] = []
        x = x0
        para_end = False
        while i < len(words):
            wd = words[i]
            if wd == PARA:
                para_end = True
                break
            wl = font.getlength(wd)
            if line_words and x + wl > max_x:
                if hyphen_p > 0 and rng.random() < hyphen_p:
                    k = _hyphen_split(wd, font, max_x - x)
                    if k:
                        line_words.append(wd[:k] + "-")
                        words[i] = wd[k:]
                break
            line_words.append(wd)
            x += wl + space_w
            i += 1
        if not line_words:
            break
        new_para = para_end
        if i >= len(words):
            para_end = True

        # word x positions (optionally justified)
        widths = [font.getlength(wd) for wd in line_words]
        gap = space_w
        if justify and not para_end and len(line_words) > 1:
            natural = sum(widths) + space_w * (len(line_words) - 1)
            extra = (max_x - x0) - natural
            if 0 < extra < 4 * space_w * (len(line_words) - 1):
                gap = space_w + extra / (len(line_words) - 1)
        boxes = []
        xc = x0
        for j, (wd, wl) in enumerate(zip(line_words, widths)):
            draw.text((xc, y), wd, fill=(ink, ink, ink), font=font)
            for k, ch in enumerate(wd):
                boxes.append((ch, xc + font.getlength(wd[:k]), xc + font.getlength(wd[:k + 1])))
            xc += wl
            if j < len(line_words) - 1:
                boxes.append((" ", xc, xc + gap))
                xc += gap
        lines.append(" ".join(line_words))
        char_boxes.append(boxes)
        bands.append((y, y + ascent + descent))
        para_end_flags.append(para_end)
        y += spacing

    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # mild scan degradation
    if rng.random() < 0.5:
        arr = cv2.GaussianBlur(arr, (3, 3), rng.uniform(0.3, 0.9))
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1, 5), arr.shape)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    page = RenderedPage(arr, lines, char_boxes, bands, margin,
                        meta={"font": font_path, "font_size": font_size, "margin": margin,
                              "para_end": para_end_flags, "layout": lay})
    return page, i


# --------------------------------------------------------------------------- #
# Burn appearance
# --------------------------------------------------------------------------- #
def apply_burn_v3(image_bgr: np.ndarray, boundary: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """Randomised burn appearance (domain randomisation): fade width and curve,
    char colour, brown scorch halo, blotchy texture, noise, blur and JPEG.

    Opacity of the char layer at column x of a row with boundary b:
        alpha = clip((x - (b - T)) / T, 0, 1) ** gamma
    """
    h, w = image_bgr.shape[:2]
    T = float(rng.uniform(30, 80))
    gamma = float(rng.uniform(0.6, 1.8))
    base = rng.uniform(12, 45)
    char_col = np.array([base, base * rng.uniform(1.0, 1.25), base * rng.uniform(1.1, 1.6)], np.float32)
    halo_w = float(rng.uniform(10, 120))
    halo_s = float(rng.uniform(0.0, 0.45))
    halo_col = np.array([rng.uniform(40, 90), rng.uniform(90, 140), rng.uniform(140, 190)], np.float32)
    noise_std = float(rng.uniform(3, 14))
    tex_amp = float(rng.uniform(0, 14))

    out = image_bgr.astype(np.float32)
    x = np.arange(w, dtype=np.float32)[None, :]
    b = boundary.astype(np.float32).reshape(-1, 1)
    fade_start = b - T
    alpha = np.clip((x - fade_start) / T, 0, 1) ** gamma
    halo = np.clip((x - (fade_start - halo_w)) / max(halo_w, 1.0), 0, 1) * halo_s
    out = out * (1 - halo[..., None]) + halo_col * halo[..., None]
    zone = (x >= fade_start).astype(np.float32)[..., None]
    tex = cv2.resize(rng.normal(0, 1, (h // 32 + 2, w // 32 + 2)).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_CUBIC)[..., None] * tex_amp
    noise = rng.normal(0, noise_std, (h, w, 1)).astype(np.float32)
    out = out * (1 - alpha[..., None]) + char_col * alpha[..., None] + (noise + tex) * zone
    out = np.clip(out, 0, 255).astype(np.uint8)
    blur = float(rng.uniform(0.3, 1.2)) if rng.rand() < 0.5 else 0.0
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    jpeg = int(rng.randint(35, 91)) if rng.rand() < 0.5 else 0
    if jpeg:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    params = {"T": round(T, 1), "gamma": round(gamma, 2), "halo": round(halo_s, 2),
              "blur": round(blur, 2), "jpeg": jpeg}
    return out, params


def alpha_x(b_x: float, T: float, gamma: float, a: float) -> float:
    """x position where the char opacity reaches ``a`` (inverse of the fade)."""
    return b_x - T * (1.0 - a ** (1.0 / gamma))


def split_at(boxes: List[Tuple[str, float, float]], cut_x: float) -> Tuple[str, str]:
    """Character-level split: a char survives if its centre is left of cut_x."""
    kept = [ch for ch, x1, x2 in boxes if (x1 + x2) / 2.0 < cut_x]
    n = len(kept)
    text = "".join(ch for ch, _, _ in boxes)
    return text[:n], text[n:]


def burn_page_to_records(
    page: RenderedPage,
    generator,
    seed: int,
    min_prefix_chars: int = 8,
    min_target_chars: int = 2,
    appearance: str = "legacy",
) -> Tuple[np.ndarray, List[Dict], List[np.ndarray]]:
    """Applies a jagged side burn to a rendered page and returns
    (corrupted_image, line_records, line_crops).

    appearance "legacy": fixed linear 55 px ghost fade (v2 dataset).
    appearance "v3":     randomised fade / colour / halo / blur / JPEG.
    In both cases the cut sits where the char layer reaches 50% opacity and a
    target character counts as READ (visible) up to VISIBLE_ALPHA.
    """
    cfg = generator.config
    h, w = page.image_bgr.shape[:2]
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    burn_ratio = py_rng.uniform(cfg.burn_min_ratio, cfg.burn_max_ratio)
    base_x = int(w * (1.0 - burn_ratio))
    amp = int(w * cfg.jagged_amplitude_ratio)
    boundary = np.clip(generator._generate_jagged_boundary(h, base_x, amp, rng=rng), 0, w - 1)
    if appearance == "v3":
        corrupted, burn_params = apply_burn_v3(page.image_bgr, boundary, rng)
        T, gamma = burn_params["T"], burn_params["gamma"]
    else:
        corrupted = generator._apply_ghost_fade(page.image_bgr, boundary, rng=rng)
        burn_params, T, gamma = {}, float(cfg.transition_zone_px), 1.0

    para_end = page.meta.get("para_end", [False] * len(page.lines))
    # First pass: surviving prefix of every line (needed for context)
    splits = []
    for (y0, y1), boxes in zip(page.bands, page.char_boxes):
        b_x = int(np.median(boundary[y0:y1 + 1]))
        if appearance == "v3":
            cut_x, vis_x = alpha_x(b_x, T, gamma, 0.5), alpha_x(b_x, T, gamma, VISIBLE_ALPHA)
        else:
            cut_x, vis_x = b_x - GHOST_CUT_PX, visible_limit_x(b_x, cfg.transition_zone_px)
        prefix, target = split_at(boxes, cut_x)
        n_vis = count_visible(boxes, len(prefix), vis_x)
        x_lo = b_x - CROP_LEFT_PX
        crop_text = "".join(ch for ch, x1, x2 in boxes if x_lo <= (x1 + x2) / 2.0 < vis_x)
        splits.append((y0, y1, b_x, prefix, target, n_vis, crop_text))

    def ctx(j):
        return splits[j][3] if 0 <= j < len(splits) else ""

    records, crops = [], []
    for idx, (y0, y1, b_x, prefix, target, n_vis, crop_text) in enumerate(splits):
        if len(prefix.strip()) < min_prefix_chars or len(target.strip()) < min_target_chars:
            continue
        rec = {
            "line_idx": idx,
            "prefix": prefix,
            "target": target,
            "visible_chars": int(n_vis),
            "prev_prefix": ctx(idx - 1),
            "next_prefix": ctx(idx + 1),
            "full_line": page.lines[idx],
            "y0": int(y0), "y1": int(y1), "b_x": int(b_x),
            "ink_x0": int(page.ink_x0),
            "page_w": int(w),
            "budget": estimate_budget(b_x, page.ink_x0, w, len(prefix)),
        }
        if appearance == "v3":
            rec.update({
                "prev2_prefix": ctx(idx - 2),
                "next2_prefix": ctx(idx + 2),
                "crop_text": crop_text,
                "para_end": bool(para_end[idx]) if idx < len(para_end) else False,
                "hyphenated": page.lines[idx].endswith("-"),
                "burn": burn_params,
            })
        records.append(rec)
        crops.append(crop_line(corrupted, y0, y1, b_x))
    return corrupted, records, crops
