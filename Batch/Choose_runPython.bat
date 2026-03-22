@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set /a count=0

for %%f in ("%SCRIPT_DIR%*.py") do (
    set /a count+=1
    set "file[!count!]=%%~f"
    set "name[!count!]=%%~nxf"
)

if !count! EQU 0 (
    echo  未找到任何 .py 文件。
    pause
    exit /b
)

echo  请选择要执行的 Python 脚本：
for /l %%i in (1,1,!count!) do (
    echo   %%i. !name[%%i]!
)

echo.
set /p choice=请输入编号（1-!count!）：

if not defined choice goto invalid

echo(!choice!| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 goto invalid

if !choice! LSS 1 goto invalid
if !choice! GTR !count! goto invalid

echo ================================================
echo  正在执行: !name[%choice%]!
echo ================================================
python "!file[%choice%]!"
echo.

echo  执行完毕，按任意键退出...
pause >nul
exit /b

:invalid
echo  输入无效，请输入 1 到 !count! 之间的数字。
pause
exit /b 1
