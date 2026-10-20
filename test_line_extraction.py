import sys, cv2, os, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from dataset_generator import SideBurnGenerator

from config import Config

cfg = Config()
gen = SideBurnGenerator(cfg)

img_dir = 'data/oprb_train/img'
txt_dir = 'data/oprb_train/cleaned_texts'

for root, _, files in os.walk(img_dir):
    for f in files:
        if f.lower().endswith(('.png', '.jpg')):
            fname = f
            img_path = os.path.join(root, f)
            break
    if 'img_path' in locals():
        break

txt_path = os.path.join(txt_dir, os.path.splitext(fname)[0] + '.txt')


with open(txt_path, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

img = cv2.imread(img_path)
img = cv2.resize(img, cfg.page_resize)
res = gen.apply_burn(img, lines)

print('Page Image Shape:', img.shape)
print('Total Lines on Page:', len(lines))

binary = gen._binarize(img)
bands = gen._resolve_line_bands(binary, len(lines))
print('Resolved Bands:', len(bands))

line_samples = []
for i, line in enumerate(lines[:15]):
    if i < len(bands):
        y0, y1 = bands[i]
        b_avg = float(np.mean(res['boundary'][y0:y1])) if y1 > y0 else float(res['boundary'][y0])
        kept_ratio = b_avg / float(img.shape[1])
        kept, clipped = gen._proportional_split(line, kept_ratio)
        if clipped.strip():
            line_samples.append((kept, clipped, y0, y1, int(b_avg)))

print(f'Extracted {len(line_samples)} severed lines from top 15:')
for kept, clipped, y0, y1, b_x in line_samples[:5]:
    print(f'  [LINE y={y0}:{y1}] Prefix: "{kept}" ---> Target: "{clipped}"')
