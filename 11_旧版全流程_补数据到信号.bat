@echo off
setlocal

cd /d "%~dp0"

echo Running automatic full pipeline without walk-forward optimization:
echo 1. refresh existing and missing raw data
echo 2. convert data
echo 3. calculate factors
echo 4. check data health
echo 5. run backtest
echo 6. generate latest candidate signal
echo.

call "%~dp0run_all.bat" %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
