@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

set CHUNK_SIZE=50
set SLEEP_SECONDS=60

echo Starting resumable stock data backfill...
echo Chunk size: %CHUNK_SIZE%
echo Sleep seconds: %SLEEP_SECONDS%
echo Progress file: outputs\data_update_progress.json
echo.

"%PYTHON%" scripts\run_update_data.py --chunk-size %CHUNK_SIZE% --sleep-seconds %SLEEP_SECONDS% %*
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
  echo Backfill command finished. Check outputs\data_update_progress.json for status.
) else (
  echo Backfill command failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
