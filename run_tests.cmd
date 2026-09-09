@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
D:\AI\envs\yolo_master\python.exe -m pytest -q
endlocal
