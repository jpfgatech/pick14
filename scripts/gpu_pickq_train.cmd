@echo off
REM GPU server: timed PickQ training — absolute paths so SSH/cmd sessions behave (see project_rule.md).
cd /d C:\Users\Administrator\Documents\projects\pick14
"C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe" -u "C:\Users\Administrator\Documents\projects\pick14\scripts\05_train_timed.py" --games 800 --batch 64 --log-file train_run.log
