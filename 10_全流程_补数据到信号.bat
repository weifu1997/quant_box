@echo off
setlocal

cd /d "%~dp0"

echo Running full pipeline:
echo 1. update data
echo 2. convert data
echo 3. calculate factors
echo 4. backtest
echo 5. generate latest signal
echo.

call "%~dp0run_all.bat" %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
