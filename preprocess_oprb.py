import os
import glob
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import time

def parse_gt_file(gt_path: str) -> List[str]:
    """
    Parses an OPRB ground truth TSV file where each line is:
    word \t x1 \t y1 \t x2 \t y2 \t r \t g \t b \t font \t tag
    Groups words into natural lines based on vertical center alignment and horizontal reading order.
    """
    words = []
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 5:
                word = parts[0]
                try:
                    x1 = float(parts[1])
                    y1 = float(parts[2])
                    x2 = float(parts[3])
                    y2 = float(parts[4])
                    if word.strip() and x2 > x1 and y2 > y1:
                        words.append((word, x1, y1, x2, y2))
                except ValueError:
                    continue

    if not words:
        return []

    # Sort words by vertical center, then x1
    words.sort(key=lambda w: ((w[2] + w[4]) / 2.0, w[1]))

    lines: List[List[Tuple[str, float, float, float, float]]] = []
    for w in words:
        cy = (w[2] + w[4]) / 2.0
        h = w[4] - w[2]
        
        matched_line = None
        min_dist = float("inf")
        for line in lines:
            line_cy = sum((item[2] + item[4]) / 2.0 for item in line) / len(line)
            line_h = sum(item[4] - item[2] for item in line) / len(line)
            dist = abs(cy - line_cy)
            if dist < 0.45 * max(h, line_h) and dist < min_dist:
                min_dist = dist
                matched_line = line
                
        if matched_line is not None:
            matched_line.append(w)
        else:
            lines.append([w])

    # Sort lines vertically from top to bottom
    lines.sort(key=lambda line: sum((item[2] + item[4]) / 2.0 for item in line) / len(line))

    # For each line, sort words horizontally left-to-right
    text_lines = []
    for line in lines:
        line.sort(key=lambda w: w[1])
        text = " ".join(w[0] for w in line).strip()
        if text:
            text_lines.append(text)

    return text_lines


def process_single_file(args):
    gt_path, output_dir = args
    stem = os.path.splitext(os.path.basename(gt_path))[0]
    out_path = os.path.join(output_dir, f"{stem}.txt")
    text_lines = parse_gt_file(gt_path)
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(text_lines))
    return True


def process_dataset(gt_dir: str, output_dir: str, limit: int = None, workers: int = None):
    os.makedirs(output_dir, exist_ok=True)
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
    print(f"Found {len(gt_files)} ground truth files in {gt_dir}")
    
    if limit:
        gt_files = gt_files[:limit]

    tasks = [(f, output_dir) for f in gt_files]
    num_workers = workers or min(os.cpu_count() or 4, 16)
    print(f"Processing {len(tasks)} files using {num_workers} parallel workers...")
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_file, tasks, chunksize=100))

    elapsed = time.time() - start_time
    print(f"Successfully processed {len(results)} files to {output_dir} in {elapsed:.2f}s!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess OPRB annotations to clean natural text lines.")
    parser.add_argument("--gt_dir", default="data/oprb_train/gt", help="Path to raw gt folder")
    parser.add_argument("--output_dir", default="data/oprb_train/cleaned_texts", help="Path to clean texts folder")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    process_dataset(args.gt_dir, args.output_dir, args.limit, args.workers)
