@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set /a count=0

for %%f in ("%SCRIPT_DIR%*.py") do (
    set /a count+=1
)

if %count%==0 (
    echo  未找到任何 .py 文件。
    pause
    exit /b
)

for %%f in ("%SCRIPT_DIR%*.py") do (
    echo ================================================
    echo  正在执行: %%~nxf
    echo ================================================
    python "%%~f"
    echo.
)

echo  全部执行完毕，按任意键退出...
pause >nul
