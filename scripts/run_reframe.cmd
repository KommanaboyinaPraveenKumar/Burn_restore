@echo off
REM Retrains the multimodal model with the residual (flamingo) fusion and the
REM visible-only contrastive loss, then runs the full held-out evaluation.
REM The text-only baseline (checkpoints/text_only) is reused; the pre-fix
REM multimodal model (checkpoints/full_v1) is evaluated as full_v1.
REM Usage (from Burn_restore/):  scripts\run_reframe.cmd
cd /d "%~dp0.."
if not exist logs mkdir logs
.venv\Scripts\python.exe -u train.py --data_dir data/lines --epochs 6 --warmup_epochs 1 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/full --num_workers 4 --log_every 200 > logs\train_full.log 2>&1
echo FULL_EXIT=%ERRORLEVEL% >> logs\train_full.log
.venv\Scripts\python.exe -u scripts\evaluate_lines.py --data_dir data/lines --split test --full_ckpt checkpoints/full/best_checkpoint.pt --text_only_ckpt checkpoints/text_only/best_checkpoint.pt --extra_ckpt full_v1=checkpoints/full_v1/best_checkpoint.pt --out results/line_eval.json > logs\eval.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval.log
