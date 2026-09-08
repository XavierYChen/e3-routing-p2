@echo off
setlocal
cd /d "%~dp0"
D:\AI\envs\yolo_master\python.exe -m pytest -q
endlocal

