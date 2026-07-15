@echo off
chcp 65001 >nul
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "scripts\dev_env.py" (
  echo 未找到 scripts\dev_env.py，请确认代码已完整拉取。
  if /I "%QUANT_BOX_NO_PAUSE%"=="1" exit /b 1
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\dev_env.py sync
) else (
  where python >nul 2>nul
  if not errorlevel 1 (
    python scripts\dev_env.py sync
  ) else (
    where py >nul 2>nul
    if errorlevel 1 (
      echo 未找到 Python 3.11。请先安装 Python 3.11。
      if /I "%QUANT_BOX_NO_PAUSE%"=="1" exit /b 1
      pause
      exit /b 1
    )
    py -3.11 scripts\dev_env.py sync
  )
)

if errorlevel 1 (
  echo.
  echo 开发环境同步失败，请根据上方 ERROR/FAIL 提示处理。
  if /I "%QUANT_BOX_NO_PAUSE%"=="1" exit /b 1
  pause
  exit /b 1
)

echo.
echo 开发环境已同步完成。
if /I "%QUANT_BOX_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0
