@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

echo Running walk-forward parameter optimization...
echo Outputs: outputs\optimization_results.csv
echo.

"%PYTHON%" scripts\run_optimize.py %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
