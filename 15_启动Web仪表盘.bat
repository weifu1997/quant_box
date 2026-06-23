@echo off
chcp 65001 >nul
setlocal EnableExtensions

cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

where npm >nul 2>nul
if errorlevel 1 (
  echo 未找到 npm。请先安装 Node.js，或重新运行 00_安装依赖环境.bat。
  echo.
  pause
  exit /b 1
)

if not exist "web\package.json" (
  echo 未找到 web\package.json，无法启动前端。
  echo.
  pause
  exit /b 1
)

if not exist "outputs\logs" mkdir "outputs\logs"

if not exist "web\node_modules" (
  echo 首次启动 Web 仪表盘，正在安装前端依赖...
  pushd web
  call npm install
  set "NPM_EXIT=%errorlevel%"
  popd
  if not "%NPM_EXIT%"=="0" (
    echo 前端依赖安装失败。
    echo.
    pause
    exit /b %NPM_EXIT%
  )
)

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
