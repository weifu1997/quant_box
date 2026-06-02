@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

echo Running backtest...
echo Outputs: outputs\backtest_*.csv/json
echo.

"%PYTHON%" scripts\run_backtest.py %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
