@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set /a COUNT=0
set /a FOUND=0

echo Scanning "%ROOT%" and subfolders for .odb files...

for /f "delims=" %%F in ('dir /s /b /a:-d "%ROOT%\*.odb" 2^>nul') do (
    set /a FOUND+=1
    attrib -r "%%F" >nul 2>&1
    del /f /q "%%F" >nul 2>&1
    if not exist "%%F" set /a COUNT+=1
)

if !FOUND! EQU 0 (
    echo No .odb files found. Nothing to delete.
) else (
    echo Done. Matched !FOUND! files, deleted !COUNT! files.
)

pause
endlocal
