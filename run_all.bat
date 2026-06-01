@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

%PYTHON% scripts\run_update_data.py
if errorlevel 1 exit /b %errorlevel%

%PYTHON% scripts\run_convert_data.py
if errorlevel 1 exit /b %errorlevel%

%PYTHON% scripts\run_calc_factors.py --force
if errorlevel 1 exit /b %errorlevel%

%PYTHON% scripts\run_backtest.py
if errorlevel 1 exit /b %errorlevel%

%PYTHON% scripts\run_daily_signal.py
if errorlevel 1 exit /b %errorlevel%

endlocal
