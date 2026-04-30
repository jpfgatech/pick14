@echo off
cd /d C:\Users\Administrator\Documents\projects\pick14
REM Stops when val loss CV stabilizes (see 05_train_timed.py), not fixed wall time.
C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe -u scripts\05_train_timed.py --games 800 --batch 64 --log-file train_run.log
