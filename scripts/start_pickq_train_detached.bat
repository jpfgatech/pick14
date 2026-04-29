@echo off
REM Detached PickQ training — survives closing SSH/RDP if started locally on the GPU box.
cd /d C:\Users\Administrator\Documents\projects\pick14_git
if not exist train_runs_remote mkdir train_runs_remote
start "pickq_train_30ep" /MIN cmd /c ""C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe" -u scripts\05_train_timed.py --rollout-dir rollout_data --epochs 30 --log-every 1 --batch 64 --lr 1e-3 --opp-weight 0.1 --out-dir train_runs_remote --log-file train_runs_remote\train_pickq_latest.log"
