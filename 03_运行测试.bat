@echo off
setlocal

cd /d "%~dp0"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

echo Running test suite...
echo.

"%PYTHON%" -m pytest -q %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
