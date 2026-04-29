@echo off
REM Fork rollout on remote Windows machine (same layout as run30_remote.bat).
REM Default job: 10_000 games, fork tails scaled for ~100 verification rows/game.
REM Pull latest code first if this checkout tracks origin on that machine.

cd /d C:\Users\Administrator\Documents\projects\pick14

REM Uncomment if this checkout tracks origin:
git pull

echo [%date% %time%] Starting rollout_mass (10k games, fork rollout + seg) >> rollout_mass_80k.log 2>&1
C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe -u scripts\05_rollout_mass.py --deck-configs 10000 --reps-per-deck 1 --max-games 10000 --shard-every-games 500 --output-dir rollout_data --clear-output-dir --verification-states-per-game 100 --branch-horizon-turns 4 >> rollout_mass_80k.log 2>&1

echo [%date% %time%] Finished rollout_mass exit=%ERRORLEVEL% >> rollout_mass_80k.log
