@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

set CHUNK_SIZE=15
set SLEEP_SECONDS=10

echo Running automatic data refresh, parameter tuning, backtest and latest signal...
echo Chunk size: %CHUNK_SIZE%
echo Sleep seconds: %SLEEP_SECONDS%
echo Outputs:
echo   outputs\auto_parameter_summary.csv
echo   outputs\auto_selected_params.json
echo   outputs\auto_backtest_metrics.json
echo   outputs\auto_signal_report.json
echo   outputs\signal_*.csv
echo   outputs\latest_holdings.csv
echo.

"%PYTHON%" scripts\run_auto_signal.py --chunk-size %CHUNK_SIZE% --sleep-seconds %SLEEP_SECONDS% %*
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
  echo Auto signal pipeline finished.
) else (
  echo Auto signal pipeline failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
