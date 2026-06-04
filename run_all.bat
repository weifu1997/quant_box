@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

set CHUNK_SIZE=300
set SLEEP_SECONDS=0
set HELP_REQUESTED=0
for %%A in (%*) do (
  if "%%~A"=="--help" set HELP_REQUESTED=1
  if "%%~A"=="-h" set HELP_REQUESTED=1
)

echo Running automatic full pipeline without walk-forward optimization:
echo 1/6 refresh missing and stale raw data
echo 2/6 convert data
echo 3/6 calculate factors
echo 4/6 check data health
echo 5/6 run backtest
echo 6/6 generate latest candidate signal
echo Chunk size override: %CHUNK_SIZE%
echo Sleep seconds override: %SLEEP_SECONDS%
echo.

"%PYTHON%" scripts\run_auto_signal.py --skip-optimize --chunk-size %CHUNK_SIZE% --sleep-seconds %SLEEP_SECONDS% %*
set EXIT_CODE=%errorlevel%

echo.
if %HELP_REQUESTED% equ 1 (
  echo Help displayed.
) else if %EXIT_CODE% equ 0 (
  echo Automatic full pipeline finished.
) else (
  echo Automatic full pipeline failed with exit code %EXIT_CODE%.
)
endlocal
exit /b %EXIT_CODE%
