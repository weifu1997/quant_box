@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

echo Computing or loading Alpha158 factors...
echo Tip: add --force after this bat to recompute from scratch.
echo.

"%PYTHON%" scripts\run_calc_factors.py %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
