@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

echo Converting raw CSV data to Qlib and price panel files...
echo.

"%PYTHON%" scripts\run_convert_data.py %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
