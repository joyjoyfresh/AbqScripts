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

:menu
echo.
echo  请选择要执行的 Python 脚本（输入 0 退出）：
for /l %%i in (1,1,!count!) do (
    echo   %%i. !name[%%i]!
)

echo.
set /p choice=请输入编号（可多个，用空格或逗号分隔）：

if not defined choice goto invalid
set "choice=!choice:,= !"
if "!choice!"=="0" goto end

set "allValid=1"
for %%t in (!choice!) do (
    echo(%%t| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 set "allValid=0"
)
if !allValid! NEQ 1 goto invalid

for %%t in (!choice!) do (
    if %%t LSS 1 set "allValid=0"
    if %%t GTR !count! set "allValid=0"
)
if !allValid! NEQ 1 goto invalid

echo ================================================
echo  开始按输入顺序执行脚本...
echo ================================================
for %%t in (!choice!) do (
    echo  正在执行: !name[%%t]!
    python "!file[%%t]!"
    echo.
)

echo  本轮执行完毕。
goto menu

:invalid
echo  输入无效，请输入 1 到 !count! 之间的数字（可多个，空格或逗号分隔）。
goto menu

:end
echo  已退出。
exit /b
