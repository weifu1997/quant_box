@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

echo Generating latest daily signal...
echo Outputs: outputs\signal_*.csv and outputs\latest_holdings.csv
echo.

"%PYTHON%" scripts\run_daily_signal.py --date latest %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
