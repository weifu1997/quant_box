@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

"%PYTHON%" scripts\show_update_progress.py

echo.
pause
exit /b 0
