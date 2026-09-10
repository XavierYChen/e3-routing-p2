@echo off
setlocal
cd /d "%~dp0"
"D:\AI\envs\yolo_master\python.exe" scripts\run_ablation.py --config configs\ablation.yaml
endlocal
