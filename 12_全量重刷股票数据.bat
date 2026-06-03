@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

set CHUNK_SIZE=100
set SLEEP_SECONDS=1

echo Starting full raw stock data refresh from configured/list dates...
echo This is much slower than step 04. Use it only when historical raw data must be rebuilt.
echo Chunk size: %CHUNK_SIZE%
echo Sleep seconds: %SLEEP_SECONDS%
echo Progress file: outputs\data_update_progress.json
echo.

"%PYTHON%" scripts\run_update_data.py --force-full --chunk-size %CHUNK_SIZE% --sleep-seconds %SLEEP_SECONDS% %*
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
  echo Full refresh command finished. Run step 05 to check latest coverage.
) else (
  echo Full refresh command failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
