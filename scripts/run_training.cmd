@echo off
REM Trains the multimodal model, then the text-only baseline on the same data.
REM Usage (from Burn_restore/):  scripts\run_training.cmd
cd /d "%~dp0.."
if not exist logs mkdir logs
.venv\Scripts\python.exe -u train.py --data_dir data/lines --epochs 6 --warmup_epochs 1 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/full --num_workers 4 --log_every 200 > logs\train_full.log 2>&1
echo FULL_EXIT=%ERRORLEVEL% >> logs\train_full.log
.venv\Scripts\python.exe -u train.py --data_dir data/lines --epochs 6 --batch_size 16 --val_limit 800 --patience 2 --ckpt_dir checkpoints/text_only --num_workers 4 --log_every 200 --text_only > logs\train_text_only.log 2>&1
echo TEXT_EXIT=%ERRORLEVEL% >> logs\train_text_only.log
