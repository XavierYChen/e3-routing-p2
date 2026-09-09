@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
"D:\AI\envs\yolo_master\python.exe" -m e3_p2 robustness --config configs\robustness.yaml
pause
