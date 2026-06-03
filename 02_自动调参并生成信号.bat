@echo off
setlocal

cd /d "%~dp0"

echo This entry has been renamed to 02_快速更新并生成信号.bat.
echo Forwarding to the quick daily signal pipeline...
echo.

call "%~dp002_快速更新并生成信号.bat" %*
exit /b %errorlevel%
