@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

set CHUNK_SIZE=15
set SLEEP_SECONDS=10

echo Running quick daily data refresh and latest manual trading signal...
echo This quick entry skips walk-forward optimization and full backtest.
echo Chunk size override: %CHUNK_SIZE%
echo Sleep seconds override: %SLEEP_SECONDS%
echo Outputs:
echo   outputs\auto_run_status.json
echo   outputs\data_health_report.json
echo   outputs\auto_parameter_quality.json
echo   outputs\auto_backtest_metrics.json
echo   outputs\auto_signal_report.json
echo   outputs\daily_signal_report.md
echo   outputs\manual_orders_*.csv
echo   outputs\manual_orders_candidate_*.csv
echo   outputs\signal_*.csv
echo   outputs\candidate_signal_*.csv
echo   outputs\latest_holdings.csv
echo.
echo Note: walk-forward parameter optimization is the separate heavy research step in 08_参数优化.bat.
echo.

"%PYTHON%" scripts\run_auto_signal.py --skip-optimize --skip-backtest --chunk-size %CHUNK_SIZE% --sleep-seconds %SLEEP_SECONDS% %*
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
  echo Quick signal pipeline finished.
) else (
  echo Quick signal pipeline failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
