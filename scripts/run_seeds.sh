#!/bin/sh
# Two extra seeds (1, 2) for the t5-base full and text-only models (seed 42 = checkpoints/v3_base_*),
# then a test-set evaluation for each seed pair.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
D=data/lines_v3
for S in 1 2; do
  $PY -u train.py --data_dir $D --text_model t5-base --gradient_checkpointing --epochs 3 --warmup_epochs 1 \
    --batch_size 16 --val_limit 800 --patience 2 --seed $S --ckpt_dir checkpoints/v3_base_full_s$S \
    --num_workers 4 --log_every 500 > logs/train_v3_base_full_s$S.log 2>&1
  echo "EXIT=$?" >> logs/train_v3_base_full_s$S.log
  $PY -u train.py --data_dir $D --text_model t5-base --gradient_checkpointing --epochs 3 \
    --batch_size 16 --val_limit 800 --patience 2 --seed $S --ckpt_dir checkpoints/v3_base_text_only_s$S \
    --num_workers 4 --log_every 500 --text_only > logs/train_v3_base_text_only_s$S.log 2>&1
  echo "EXIT=$?" >> logs/train_v3_base_text_only_s$S.log
done
for S in 1 2; do
  $PY -u scripts/evaluate_lines.py --data_dir $D --skip_zero_shot \
    --full_ckpt checkpoints/v3_base_full_s$S/best_checkpoint.pt \
    --text_only_ckpt checkpoints/v3_base_text_only_s$S/best_checkpoint.pt \
    --calib_n 1000 --out results/line_eval_v3_base_s$S.json > logs/eval_v3_base_s$S.log 2>&1
  echo "EVAL_EXIT=$?" >> logs/eval_v3_base_s$S.log
done
echo ALL_DONE
