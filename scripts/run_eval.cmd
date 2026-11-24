@echo off
REM Held-out evaluation of the trained models on the full test split.
REM Usage (from Burn_restore/):  scripts\run_eval.cmd
cd /d "%~dp0.."
if not exist logs mkdir logs
.venv\Scripts\python.exe -u scripts\evaluate_lines.py --data_dir data/lines --split test --full_ckpt checkpoints/full/best_checkpoint.pt --text_only_ckpt checkpoints/text_only/best_checkpoint.pt --out results/line_eval.json > logs\eval.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval.log
