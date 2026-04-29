@echo off
REM Mass Q-rollout on remote Windows machine (same layout as run30_remote.bat).
REM Default: 10_000 deck configs × 8 reps = 80_000 games, fork rollout, shards of 500 games.
REM Pull latest code first if this repo tracks origin on that machine.

cd /d C:\Users\Administrator\Documents\projects\pick14

REM Uncomment if this checkout tracks origin:
git pull

echo [%date% %time%] Starting rollout_mass (80k games planned) >> rollout_mass_80k.log 2>&1
C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe -u scripts\05_rollout_mass.py --deck-configs 10000 --reps-per-deck 8 --shard-every-games 500 --output-dir rollout_data >> rollout_mass_80k.log 2>&1

echo [%date% %time%] Finished rollout_mass exit=%ERRORLEVEL% >> rollout_mass_80k.log
