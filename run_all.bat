@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

echo Running legacy full pipeline without walk-forward optimization:
echo 1/5 update data
echo 2/5 convert data
echo 3/5 calculate factors
echo 4/5 backtest current config
echo 5/5 generate latest candidate signal
echo.

echo [1/5] Updating data...
%PYTHON% scripts\run_update_data.py
if errorlevel 1 goto failed

echo.
echo [2/5] Converting data...
%PYTHON% scripts\run_convert_data.py
if errorlevel 1 goto failed

echo.
echo [3/5] Calculating factors...
%PYTHON% scripts\run_calc_factors.py --force
if errorlevel 1 goto failed

echo.
echo [4/5] Running backtest...
%PYTHON% scripts\run_backtest.py
if errorlevel 1 goto failed

echo.
echo [5/5] Generating latest candidate signal...
%PYTHON% scripts\run_daily_signal.py
if errorlevel 1 goto failed

echo.
echo Legacy full pipeline finished.
endlocal
exit /b 0

:failed
set EXIT_CODE=%errorlevel%
echo.
echo Legacy full pipeline failed with exit code %EXIT_CODE%.
endlocal
exit /b %EXIT_CODE%
