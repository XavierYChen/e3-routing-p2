@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=D:\AI\envs\yolo_master\python.exe"
if not exist "%PYTHON%" exit /b 2
cd /d "%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONPATH=%ROOT%src"
"%PYTHON%" -m e3_p2 demo %*
exit /b %ERRORLEVEL%
