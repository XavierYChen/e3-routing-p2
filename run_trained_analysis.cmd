@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
"D:\AI\envs\yolo_master\python.exe" -m e3_p2.trained_analysis --config configs\trained_analysis.yaml
endlocal
