@echo off
REM Background worker process for mass rollout (invoked via run_rollout_mass_remote.bat + START).
cd /d C:\Users\Administrator\Documents\projects\pick14
echo [%date% %time%] worker begin >> rollout_mass_80k.log 2>&1
"C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe" -u scripts\05_rollout_mass.py --deck-configs 10000 --reps-per-deck 1 --max-games 10000 --shard-every-games 500 --output-dir rollout_data --clear-output-dir --verification-states-per-game 100 --branch-horizon-turns 4 >> rollout_mass_80k.log 2>&1
echo [%date% %time%] worker end exit=%ERRORLEVEL% >> rollout_mass_80k.log 2>&1
