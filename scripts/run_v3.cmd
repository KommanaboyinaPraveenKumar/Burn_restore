@echo off
REM v3 pipeline: finer visual tokens + CTC reading head + length hint + randomised
REM burns + realistic layout + 2-line context + OCR-noised context.
REM Round 1 = t5-small, Round 2 = t5-base.  Each round trains the multimodal model
REM and the text-only baseline on the same data, then evaluates on held-out books.
REM Usage (from Burn_restore/):  scripts\run_v3.cmd
cd /d "%~dp0.."
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe
set D=data/lines_v3

%PY% -u train.py --data_dir %D% --epochs 5 --warmup_epochs 1 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3_full --num_workers 4 --log_every 500 > logs\train_v3_full.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3_full.log
%PY% -u train.py --data_dir %D% --epochs 5 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3_text_only --num_workers 4 --log_every 500 --text_only > logs\train_v3_text_only.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3_text_only.log
%PY% -u scripts\evaluate_lines.py --data_dir %D% --full_ckpt checkpoints/v3_full/best_checkpoint.pt --text_only_ckpt checkpoints/v3_text_only/best_checkpoint.pt --extra_ckpt v2_full=checkpoints/full/best_checkpoint.pt --calib_n 1000 --context_noise 0.03 --out results/line_eval_v3.json > logs\eval_v3.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval_v3.log

%PY% -u train.py --data_dir %D% --text_model t5-base --gradient_checkpointing --epochs 3 --warmup_epochs 1 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3_base_full --num_workers 4 --log_every 500 > logs\train_v3_base_full.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3_base_full.log
%PY% -u train.py --data_dir %D% --text_model t5-base --gradient_checkpointing --epochs 3 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/v3_base_text_only --num_workers 4 --log_every 500 --text_only > logs\train_v3_base_text_only.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\train_v3_base_text_only.log
%PY% -u scripts\evaluate_lines.py --data_dir %D% --skip_zero_shot --full_ckpt checkpoints/v3_base_full/best_checkpoint.pt --text_only_ckpt checkpoints/v3_base_text_only/best_checkpoint.pt --extra_ckpt small_full=checkpoints/v3_full/best_checkpoint.pt --extra_ckpt small_text_only=checkpoints/v3_text_only/best_checkpoint.pt --calib_n 1000 --context_noise 0.03 --out results/line_eval_v3_base.json > logs\eval_v3_base.log 2>&1
echo EVAL_EXIT=%ERRORLEVEL% >> logs\eval_v3_base.log
