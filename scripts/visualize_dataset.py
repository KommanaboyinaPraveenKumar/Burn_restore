"""
visualize_dataset.py
====================
Dataset Inspection & Alignment Visualization Suite.

Generates visual diagnostic contact sheets showing:
  - Original clean document scan with text line bounds.
  - Corrupted burnt document with organic jagged boundary overlay.
  - Binary burn mask with physical bounding box.
  - Line-level text-to-image alignment overlays (green = surviving, red = missing).
  - Multi-profile comparison: Mild, Moderate, Severe, Irregular.
"""

import argparse
import json
import os
import sys

# Ensure repository root and Burn_restore are on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
burn_restore_dir = os.path.join(repo_root, "Burn_restore")
for p in [burn_restore_dir, repo_root]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from config import Config
    from dataset_generator import SideBurnGenerator
except ImportError:
    from Burn_restore.config import Config
    from Burn_restore.dataset_generator import SideBurnGenerator


def create_inspection_panel(
    clean_image: np.ndarray,
    res: Dict,
    text_lines: List[str],
    profile_name: str,
    output_width: int = 1200,
) -> np.ndarray:
    """Assembles a 4-panel visual strip for a single corruption instance."""
    h, w = clean_image.shape[:2]
    corrupted = res["corrupted_image"].copy()
    mask = res["burn_mask"]
    boundary = res["boundary"]
    severed_lines = res["severed_lines"]

    # 1. Annotate clean image with line bounds
    vis_clean = clean_image.copy()
    for s in res["all_line_splits"][:12]:
        y0, y1 = s["y0"], s["y1"]
        cv2.rectangle(vis_clean, (20, y0), (w - 20, y1), (0, 180, 0), 1)

    # 2. Annotate corrupted image with jagged boundary
    vis_corrupt = corrupted.copy()
    pts = np.array([[boundary[y], y] for y in range(0, h, 2)], dtype=np.int32)
    cv2.polylines(vis_corrupt, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    # 3. Mask visualization (3 channels)
    vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    bx, by, bw, bh = res["burn_bbox"]
    cv2.rectangle(vis_mask, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

    # 4. Alignment detail panel (crop of top severed line)
    if severed_lines:
        active = severed_lines[0]
        y0, y1 = active["y0"], active["y1"]
        b_x = active["b_x"]
        pad = 20
        cy0, cy1 = max(0, y0 - pad), min(h, y1 + pad)
        strip_clean = clean_image[cy0:cy1, :].copy()
        strip_burnt = corrupted[cy0:cy1, :].copy()

        # Combine strips vertically
        comp_strip = np.vstack([strip_clean, strip_burnt])
        comp_strip = cv2.resize(comp_strip, (w, 160))
    else:
        comp_strip = np.full((160, w, 3), 240, dtype=np.uint8)

    # Resize all panels to uniform display size
    disp_w = 280
    disp_h = 280
    p1 = cv2.resize(vis_clean, (disp_w, disp_h))
    p2 = cv2.resize(vis_corrupt, (disp_w, disp_h))
    p3 = cv2.resize(vis_mask, (disp_w, disp_h))
    p4 = cv2.resize(comp_strip, (disp_w, disp_h))

    # Assemble horizontally
    strip = np.hstack([p1, p2, p3, p4])

    # Header banner
    header = np.full((40, strip.shape[1], 3), 30, dtype=np.uint8)
    banner_pil = Image.fromarray(header)
    draw = ImageDraw.Draw(banner_pil)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    sample_target = severed_lines[0]["target"] if severed_lines else "None"
    sample_prefix = severed_lines[0]["prefix"] if severed_lines else "None"
    status_str = f"PROFILE: {profile_name.upper()} | Target: '{sample_target[:30]}' | Prefix: '{sample_prefix[:30]}...'"
    draw.text((12, 10), status_str, fill=(240, 240, 240), font=font)
    header = np.array(banner_pil)

    combined = np.vstack([header, strip])
    return combined


def generate_contact_sheet(
    image_path: str,
    text_path: str,
    output_path: str = "assets/dataset_inspection_contact_sheet.png",
) -> str:
    """Generates a complete contact sheet contrasting Mild, Moderate, and Severe corruptions."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    img = cv2.resize(img, (1024, 1024))

    with open(text_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    profiles = ["mild", "moderate", "severe"]
    panels = []

    for prof in profiles:
        cfg = Config()
        cfg.set_burn_profile(prof)
        gen = SideBurnGenerator(cfg)
        res = gen.apply_burn(img, lines, seed=42)
        panel = create_inspection_panel(img, res, lines, prof)
        panels.append(panel)

    contact_sheet = np.vstack(panels)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, contact_sheet)
    print(f"[visualize_dataset] Generated inspection contact sheet -> {output_path} ({contact_sheet.shape})")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset inspection visualizations.")
    parser.add_argument("--image", default="tests/fixtures/mini_dataset/clean_images/sample_page_001.png")
    parser.add_argument("--text", default="tests/fixtures/mini_dataset/clean_texts/sample_page_001.txt")
    parser.add_argument("--out", default="assets/dataset_inspection_contact_sheet.png")
    args = parser.parse_args()

    generate_contact_sheet(args.image, args.text, args.out)
