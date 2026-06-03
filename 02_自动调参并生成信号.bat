@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

set CHUNK_SIZE=300
set SLEEP_SECONDS=1

echo This legacy entry now runs the quick daily signal pipeline.
echo Prefer 02_快速更新并生成信号.bat for new runs.
echo This quick entry skips walk-forward optimization and full backtest.
echo Chunk size override: %CHUNK_SIZE%
echo Sleep seconds override: %SLEEP_SECONDS%
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
