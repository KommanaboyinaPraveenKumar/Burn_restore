#!/usr/bin/env python3
"""
scripts/restore_document.py
===========================
Command-Line Interface for Grounded Side-Burnt Document Text Restoration.

Usage:
    python scripts/restore_document.py --input sample.png --output results/
    python scripts/restore_document.py --input sample.png --mode manual --bbox 700,50,980,950 --save-overlay
    python scripts/restore_document.py --input sample.png --checkpoint checkpoints/v3_base_full/best_checkpoint.pt --verbose

Features:
    - Automatic right-edge burn boundary detection or manual bounding-box override.
    - Physical character-budget estimation and compliance auditing.
    - Autoregressively grounded text infilling using ForensicT5Restorer.
    - Diagnostic overlay visualization rendering.
    - Structured JSON report export.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add repo root and Burn_restore to sys.path
repo_root = Path(__file__).resolve().parent.parent
burn_dir = repo_root / "Burn_restore"
for p in [str(burn_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.restoration_engine import DEFAULT_CHECKPOINT, RestorationEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grounded Side-Burnt Document Text Restoration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input document image (.png, .jpg, .tif)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Directory to save restoration results and overlay images",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device for inference (default: auto-detect)",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="Burn detection mode: 'auto' (OpenCV edge detector) or 'manual' (requires --bbox)",
    )
    parser.add_argument(
        "--bbox", "-b",
        type=str,
        default=None,
        help="Manual damage bounding box as 'x1,y1,x2,y2' (required if --mode manual)",
    )
    parser.add_argument(
        "--transcripts", "-t",
        type=str,
        default=None,
        help="Optional text file containing surviving line prefixes (one line per text line)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of severed text lines to restore",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum new tokens generated per line",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam size for autoregressive search",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="Save diagnostic visual overlay showing boundaries and restorations",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save structured JSON restoration report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose diagnostic details during execution",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    manual_bbox = None
    if args.mode == "manual":
        if not args.bbox:
            print("[ERROR] --mode manual requires --bbox 'x1,y1,x2,y2'", file=sys.stderr)
            sys.exit(1)
        try:
            parts = [int(v.strip()) for v in args.bbox.split(",")]
            if len(parts) != 4:
                raise ValueError("Must provide exactly 4 integers: x1,y1,x2,y2")
            manual_bbox = tuple(parts)
        except Exception as e:
            print(f"[ERROR] Invalid --bbox format '{args.bbox}': {e}", file=sys.stderr)
            sys.exit(1)

    # Optional transcripts
    transcripts = None
    if args.transcripts and os.path.isfile(args.transcripts):
        with open(args.transcripts, "r", encoding="utf-8") as f:
            transcripts = [line.strip() for line in f if line.strip()]

    print("=" * 80)
    print("GROUNDED SIDE-BURNT DOCUMENT TEXT RESTORATION CLI")
    print("=" * 80)
    print(f"Input image:        {args.input}")
    print(f"Output directory:   {args.output}")
    print(f"Model checkpoint:   {args.checkpoint}")
    print(f"Detection mode:     {args.mode.upper()}")
    if manual_bbox:
        print(f"Manual BBox:        {manual_bbox}")
    print("-" * 80)

    # Initialize Engine
    t0 = time.time()
    engine = RestorationEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    init_time = time.time() - t0

    if args.verbose:
        status = engine.get_model_status()
        print(f"[DEBUG] Device: {status['device']} | Params: {status['total_parameters']:,} | Load: {init_time:.2f}s")

    os.makedirs(args.output, exist_ok=True)
    stem = Path(args.input).stem
    overlay_path = os.path.join(args.output, f"{stem}_restoration_overlay.png") if args.save_overlay else None
    json_path = os.path.join(args.output, f"{stem}_restoration_report.json")

    # Perform full-page restoration
    print("[CLI] Processing document and executing multimodal restoration...")
    results = engine.restore_page(
        image=args.input,
        manual_bbox=manual_bbox,
        page_transcripts=transcripts,
        max_lines=args.max_lines,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        save_overlay_path=overlay_path,
    )

    # Console Summary Table
    print("\n" + "=" * 80)
    print("RESTORATION SUMMARY")
    print("=" * 80)
    print(f"{'Line':<6} | {'Prefix Context':<25} | {'Restored Text':<22} | {'Budget':<8} | {'Latency':<8}")
    print("-" * 80)

    for r in results["restorations"]:
        p_sub = r["surviving_prefix"][:23] + ".." if len(r["surviving_prefix"]) > 25 else r["surviving_prefix"]
        t_sub = r["predicted_restoration"][:20] + ".." if len(r["predicted_restoration"]) > 22 else r["predicted_restoration"]
        b_val = f"{r['physical_budget']['estimated_budget']:.1f} ch"
        l_val = f"{r['inference_time_ms']:.0f} ms"
        print(f"{r['line_idx']:<6} | {p_sub:<25} | {t_sub:<22} | {b_val:<8} | {l_val:<8}")

    print("=" * 80)
    print(f"Total severed lines detected: {results['total_lines_severed']}")
    print(f"Total lines restored:         {results['restored_lines_count']}")
    print(f"Total page inference latency: {results['total_page_inference_ms']:.1f} ms")
    print(f"Average latency per line:     {results['avg_latency_per_line_ms']:.1f} ms")

    if overlay_path and os.path.isfile(overlay_path):
        print(f"Diagnostic overlay saved to:  {overlay_path}")

    if args.save_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Structured JSON saved to:     {json_path}")

    print("-" * 80)
    print("Note: Character restoration quality is conditioned on the trained checkpoint.")
    print("Physical character budgets are geometric bounds estimated from page dimensions.")
    print("=" * 80)


if __name__ == "__main__":
    main()

