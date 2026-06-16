@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 未找到 .venv\Scripts\python.exe，请先运行 00_安装依赖环境.bat
  pause
  exit /b 1
)

echo 构建历史股票池：沪深300 + 中证500 + 中证1000权重前300
".venv\Scripts\python.exe" scripts\run_build_universe.py %*
if errorlevel 1 (
  echo 历史股票池构建失败，请查看上方错误信息。
  pause
  exit /b 1
)

echo 历史股票池已生成到 data\raw\historical_universe.csv
pause
