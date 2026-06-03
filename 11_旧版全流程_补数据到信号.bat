@echo off
setlocal

cd /d "%~dp0"

echo Running legacy full pipeline without walk-forward optimization:
echo 1. update data
echo 2. convert data
echo 3. calculate factors
echo 4. backtest current config
echo 5. generate latest candidate signal
echo.

call "%~dp0run_all.bat" %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
