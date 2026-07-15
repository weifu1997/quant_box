@echo off
chcp 65001 >nul
setlocal EnableExtensions

cd /d "%~dp0"

set "QUANT_BOX_NO_PAUSE=1"
call "%~dp000_安装依赖环境.bat"
if errorlevel 1 exit /b 1
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "outputs\logs" mkdir "outputs\logs"

echo 启动 FastAPI 后端：http://127.0.0.1:8000
echo 后端运行日志会显示在新打开的 backend 窗口中。
start "quant_box dashboard backend" /D "%~dp0" "%PYTHON%" scripts\run_dashboard.py --host 127.0.0.1 --port 8000

echo 启动 React/Vite 前端：http://127.0.0.1:5173
echo 前端运行日志会显示在新打开的 frontend 窗口中。
start "quant_box dashboard frontend" /D "%~dp0web" npm run dev -- --host 127.0.0.1 --port 5173

echo.
echo 正在等待前后端服务就绪...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; function Wait-Url([string]$Url) { for ($i = 0; $i -lt 60; $i++) { try { $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { Write-Host ('OK ' + $Url); return } } catch { Start-Sleep -Seconds 1 } }; throw ('Timed out waiting for ' + $Url) }; Wait-Url 'http://127.0.0.1:8000/api/health'; Wait-Url 'http://127.0.0.1:5173'"
if errorlevel 1 (
  echo.
  echo 启动检查失败。请查看新打开的 backend 和 frontend 窗口。
  echo.
  pause
  exit /b 1
)

echo.
echo Web 仪表盘已启动：http://127.0.0.1:5173
if /I not "%DASHBOARD_NO_BROWSER%"=="1" start "" "http://127.0.0.1:5173"
echo.
if /I "%DASHBOARD_NO_PAUSE%"=="1" exit /b 0
pause
