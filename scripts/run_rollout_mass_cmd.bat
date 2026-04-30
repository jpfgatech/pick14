@echo off
REM One-line smoke / full runner: always correct cwd (avoids SSH defaulting to C:\Users\Administrator).
cd /d C:\Users\Administrator\Documents\projects\pick14
C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe -u scripts\05_rollout_mass.py %*
