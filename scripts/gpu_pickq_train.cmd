@echo off
REM GPU server: timed PickQ training from saved rollout shards (see instructions/05.md).
REM Requires rollout_data\shard_*.npz from scripts\05_rollout_mass.py first.
REM Logs every epoch: tr_q / tr_opp / val_q / val_opp (global validation BCE on opp head).
cd /d C:\Users\Administrator\Documents\projects\pick14_git
"C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe" -u "C:\Users\Administrator\Documents\projects\pick14_git\scripts\05_train_timed.py" --rollout-dir rollout_data --epochs 30 --log-every 1 --batch 64 --opp-weight 0.1 --out-dir train_runs_remote --log-file train_runs_remote\train_pickq_latest.log
