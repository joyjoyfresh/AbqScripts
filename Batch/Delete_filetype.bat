@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "PSHELPER=%~dp0Delete_filetype_helper.ps1"
set "EXTS_DEFAULT=inp odb jnl msg prt dat sta sim com"
set "EXTS_INPUT="

echo Default extensions: .inp  .odb  .jnl  .msg  .prt  .dat  .sta  .sim  .com
echo Leave blank to use defaults, or enter space-separated extensions (e.g.: odb sta msg):
set /p "EXTS_INPUT=Extensions to delete: "

if not defined EXTS_INPUT (
    set "EXTS_INPUT=%EXTS_DEFAULT%"
    echo Using defaults: .inp  .odb  .jnl  .msg  .prt  .dat  .sta  .sim  .com
)

rem Strip leading dots from each extension
set "EXTS_CLEAN="
for %%E in (!EXTS_INPUT!) do (
    set "_e=%%E"
    if "!_e:~0,1!"=="." set "_e=!_e:~1!"
    if defined _e set "EXTS_CLEAN=!EXTS_CLEAN! !_e!"
)
set "EXTS_CLEAN=!EXTS_CLEAN:~1!"

if not defined EXTS_CLEAN (
    echo No valid extensions found. Operation canceled.
    pause
    endlocal
    exit /b 1
)

echo.
echo Scanning "%ROOT%" recursively...
echo.

set "EXTS_FOR_PS=!EXTS_CLEAN: =;!"

set "DFT_MODE=scan"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PSHELPER%"

rem Re-count total in CMD to decide whether to proceed
set "_TOTAL_CHECK=0"
for %%E in (!EXTS_CLEAN!) do (
    for /f %%C in ('dir /s /b /a:-d "%ROOT%\*.%%E" 2^>nul ^| find /c /v ""') do set /a "_TOTAL_CHECK+=%%C"
)

if !_TOTAL_CHECK! EQU 0 (
    echo No matching files found. Nothing to do.
    pause
    endlocal
    exit /b 0
)

echo.
choice /m "Recycle all listed files"
if errorlevel 2 (
    echo Operation canceled by user. No files were deleted.
    pause
    endlocal
    exit /b 0
)

echo.
echo Recycling files...

set "DFT_MODE=recycle"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PSHELPER%"

echo.
pause
endlocal
