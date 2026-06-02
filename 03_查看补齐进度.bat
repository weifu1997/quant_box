@echo off
setlocal

cd /d "%~dp0"

echo Local raw CSV count:
powershell -NoProfile -Command "(Get-ChildItem data\raw -Filter *.csv | Measure-Object).Count"
echo.

echo Progress file:
if exist outputs\data_update_progress.json (
  type outputs\data_update_progress.json
) else (
  echo outputs\data_update_progress.json does not exist yet.
)

echo.
pause
exit /b 0
