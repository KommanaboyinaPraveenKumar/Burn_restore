@echo off
REM Re-evaluates the v3 checkpoints with the calibrated upper budget
REM (checkpoints/cut_estimator.pt, scripts/train_cut_estimator.py) and hard look-ahead decoding.
REM Usage (from Burn_restore/):  scripts\run_budget_eval.cmd
cd /d "%~dp0.."
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe
set D=data/lines_v3
%PY% -u scripts\evaluate_lines.py --data_dir %D% --full_ckpt checkpoints/v3_full/best_checkpoint.pt --text_only_ckpt checkpoints/v3_text_only/best_checkpoint.pt --extra_ckpt v2_full=checkpoints/full/best_checkpoint.pt --calib_n 1000 --context_noise 0.03 --out results/line_eval_v3.json > logs\eval_v3_budget.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval_v3_budget.log
%PY% -u scripts\evaluate_lines.py --data_dir %D% --skip_zero_shot --full_ckpt checkpoints/v3_base_full/best_checkpoint.pt --text_only_ckpt checkpoints/v3_base_text_only/best_checkpoint.pt --extra_ckpt small_full=checkpoints/v3_full/best_checkpoint.pt --extra_ckpt small_text_only=checkpoints/v3_text_only/best_checkpoint.pt --calib_n 1000 --context_noise 0.03 --out results/line_eval_v3_base.json > logs\eval_v3_base_budget.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval_v3_base_budget.log
