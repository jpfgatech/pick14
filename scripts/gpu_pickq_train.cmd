@echo off
REM GPU server: timed PickQ training from saved rollout shards (see instructions/05.md).
REM Requires rollout_data\shard_*.npz from scripts\05_rollout_mass.py first.
REM --epochs 50 runs a fixed-length run (~30 min depending on GPU); tr_q / val_q / val_cv as in script.
cd /d C:\Users\Administrator\Documents\projects\pick14_git
"C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe" -u "C:\Users\Administrator\Documents\projects\pick14_git\scripts\05_train_timed.py" --rollout-dir rollout_data --batch 64 --epochs 50 --log-file train_run.log
