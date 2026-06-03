@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

set CHUNK_SIZE=300
set SLEEP_SECONDS=1

echo Starting incremental stale/missing stock data update...
echo Chunk size: %CHUNK_SIZE%
echo Sleep seconds: %SLEEP_SECONDS%
echo Progress file: outputs\data_update_progress.json
echo.

"%PYTHON%" scripts\run_update_data.py --chunk-size %CHUNK_SIZE% --sleep-seconds %SLEEP_SECONDS% %*
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
  echo Backfill command finished. Run step 05 to check latest coverage.
) else (
  echo Backfill command failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
