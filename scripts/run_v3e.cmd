@echo off
REM Retrain with the length hint from the predicted cut position (prompt style v3e).
REM Needs: checkpoints/cut_estimator.pt and budget_est in the records
REM (scripts/train_cut_estimator.py, scripts/add_estimated_budget.py).
REM Same schedules as run_v3.cmd so the only change is the length hint.
REM Usage (from Burn_restore/):  scripts\run_v3e.cmd
cd /d "%~dp0.."
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe
set D=data/lines_v3

%PY% -u train.py --data_dir %D% --prompt_style v3e --epochs 5 --warmup_epochs 1 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3e_full --num_workers 4 --log_every 500 > logs\train_v3e_full.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3e_full.log
%PY% -u train.py --data_dir %D% --prompt_style v3e --epochs 5 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3e_text_only --num_workers 4 --log_every 500 --text_only > logs\train_v3e_text_only.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3e_text_only.log
%PY% -u scripts\evaluate_lines.py --data_dir %D% --skip_zero_shot --full_ckpt checkpoints/v3e_full/best_checkpoint.pt --text_only_ckpt checkpoints/v3e_text_only/best_checkpoint.pt --extra_ckpt v3_full=checkpoints/v3_full/best_checkpoint.pt --extra_ckpt v3_text_only=checkpoints/v3_text_only/best_checkpoint.pt --calib_n 1000 --context_noise 0.03 --out results/line_eval_v3e.json > logs\eval_v3e.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval_v3e.log

%PY% -u train.py --data_dir %D% --prompt_style v3e --text_model t5-base --gradient_checkpointing --epochs 3 --warmup_epochs 1 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3e_base_full --num_workers 4 --log_every 500 > logs\train_v3e_base_full.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3e_base_full.log
%PY% -u train.py --data_dir %D% --prompt_style v3e --text_model t5-base --gradient_checkpointing --epochs 3 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3e_base_text_only --num_workers 4 --log_every 500 --text_only > logs\train_v3e_base_text_only.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3e_base_text_only.log
%PY% -u scripts\evaluate_lines.py --data_dir %D% --skip_zero_shot --full_ckpt checkpoints/v3e_base_full/best_checkpoint.pt --text_only_ckpt checkpoints/v3e_base_text_only/best_checkpoint.pt --extra_ckpt v3_base_full=checkpoints/v3_base_full/best_checkpoint.pt --extra_ckpt v3_base_text_only=checkpoints/v3_base_text_only/best_checkpoint.pt --calib_n 1000 --context_noise 0.03 --out results/line_eval_v3e_base.json > logs\eval_v3e_base.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval_v3e_base.log
