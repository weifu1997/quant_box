@echo off
setlocal

cd /d "%~dp0"

if not exist requirements.txt (
  echo requirements.txt not found. Please run this bat from the project root.
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  echo Existing virtual environment found: .venv
) else (
  echo Creating virtual environment: .venv
  python -m venv .venv
  if errorlevel 1 (
    echo Failed with "python". Trying "py -3"...
    py -3 -m venv .venv
  )
  if errorlevel 1 (
    echo Failed to create virtual environment. Please install Python 3.10 or 3.11 and try again.
    pause
    exit /b 1
  )
)

set PYTHON=%~dp0.venv\Scripts\python.exe
set PIP_COMMON_ARGS=--disable-pip-version-check --timeout 60 --retries 10

echo.
echo Checking pip...
"%PYTHON%" -m pip --version >nul 2>&1
if errorlevel 1 (
  echo pip is missing in .venv. Bootstrapping pip with ensurepip...
  "%PYTHON%" -m ensurepip --upgrade
  if errorlevel 1 (
    echo Failed to bootstrap pip with ensurepip.
    echo Please delete .venv and run this script again.
    pause
    exit /b 1
  )
)

echo.
echo Installing compatible pip version...
"%PYTHON%" -m pip install %PIP_COMMON_ARGS% "pip<24.1"
if errorlevel 1 (
  echo Failed to install a compatible pip version.
  pause
  exit /b 1
)

echo.
echo Installing project dependencies...
"%PYTHON%" -m pip install %PIP_COMMON_ARGS% -r requirements.txt
if errorlevel 1 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo.
echo Environment is ready.
echo Python:
"%PYTHON%" --version
echo.
pause
exit /b 0
