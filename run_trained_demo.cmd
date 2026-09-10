@echo off
setlocal
cd /d "%~dp0"
"D:\AI\envs\yolo_master\python.exe" scripts\serve_trained_demo.py
endlocal
