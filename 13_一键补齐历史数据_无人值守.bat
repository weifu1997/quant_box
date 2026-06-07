@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

set "START_DATE=2012-01-01"
set "END_DATE=auto"
set "RAW_LOOPS=100"
set "RAW_CHUNK_SIZE=40"
set "RAW_SLEEP_SECONDS=0"
set "DAILY_BASIC_LOOPS=80"
set "DAILY_BASIC_MAX_DATES=60"
set "DAILY_BASIC_SLEEP_SECONDS=0"
set "RUN_FACTOR_CALC=1"
set "FACTOR_ARGS=--force"
set "LOG_DIR=outputs\logs"
set "LOG_FILE=%LOG_DIR%\history_data_backfill.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

call :log "无人值守历史数据补齐开始"
call :log "raw 区间：%START_DATE% 到 %END_DATE%，每轮 %RAW_CHUNK_SIZE% 只，共 %RAW_LOOPS% 轮"
call :log "daily_basic 每轮 %DAILY_BASIC_MAX_DATES% 个交易日，共 %DAILY_BASIC_LOOPS% 轮"
call :log "日志文件：%LOG_FILE%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$running = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'scripts[\\/](run_update_data|run_update_daily_basic)\.py' }; if ($running) { $running | Select-Object ProcessId,CommandLine | Format-List; exit 1 }"
if errorlevel 1 (
  call :log "检测到已有数据抓取进程正在运行，本脚本已停止，避免重复请求。"
  goto fail
)

for /L %%I in (1,1,%RAW_LOOPS%) do (
  call :log "raw 日线补齐：第 %%I/%RAW_LOOPS% 轮"
  "%PYTHON%" scripts\run_update_data.py --start-date %START_DATE% --end-date %END_DATE% --force-full --include-existing --chunk-size %RAW_CHUNK_SIZE% --sleep-seconds %RAW_SLEEP_SECONDS% --max-chunks 1 >> "%LOG_FILE%" 2>&1
  if errorlevel 1 (
    call :log "raw 日线补齐失败，停止后续步骤。"
    goto fail
  )
  "%PYTHON%" scripts\show_update_progress.py >> "%LOG_FILE%" 2>&1
)

call :log "raw 日线补齐循环完成，开始转换数据。"
"%PYTHON%" scripts\run_convert_data.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  call :log "数据转换失败，停止后续步骤。"
  goto fail
)

for /L %%I in (1,1,%DAILY_BASIC_LOOPS%) do (
  call :log "daily_basic 补齐：第 %%I/%DAILY_BASIC_LOOPS% 轮"
  "%PYTHON%" scripts\run_update_daily_basic.py --start-date %START_DATE% --end-date %END_DATE% --sleep-seconds %DAILY_BASIC_SLEEP_SECONDS% --max-dates %DAILY_BASIC_MAX_DATES% >> "%LOG_FILE%" 2>&1
  if errorlevel 1 (
    call :log "daily_basic 补齐失败，停止后续步骤。"
    goto fail
  )
)

if "%RUN_FACTOR_CALC%"=="1" (
  call :log "开始重算 Alpha158 因子。"
  "%PYTHON%" scripts\run_calc_factors.py %FACTOR_ARGS% >> "%LOG_FILE%" 2>&1
  if errorlevel 1 (
    call :log "因子计算失败。"
    goto fail
  )
) else (
  call :log "已按配置跳过因子计算。"
)

call :log "无人值守历史数据补齐完成。"
"%PYTHON%" scripts\show_update_progress.py

echo.
echo 完成。详细日志：%LOG_FILE%
echo 如果有失败股票，请查看：data\raw\failed_fetches.csv
pause
exit /b 0

:fail
echo.
echo 脚本中止。详细日志：%LOG_FILE%
echo 如果有失败股票，请查看：data\raw\failed_fetches.csv
pause
exit /b 1

:log
echo [%date% %time%] %~1
echo [%date% %time%] %~1>> "%LOG_FILE%"
exit /b 0
