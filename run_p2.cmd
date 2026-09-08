@echo off
setlocal
cd /d "%~dp0"
if not defined E3_YOLO_MASTER_ROOT set "E3_YOLO_MASTER_ROOT=D:\AI\YOLO-Master"
set "PYTHON=D:\AI\envs\yolo_master\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
set "OUT=results\run-%date:~0,4%%date:~5,2%%date:~8,2%-%time:~0,2%%time:~3,2%%time:~6,2%"
set "OUT=%OUT: =0%"
"%PYTHON%" -m e3_routing_p2.cli --yolo-root "%E3_YOLO_MASTER_ROOT%" --output "%OUT%" --imgsz 160 --device cpu
endlocal

