@echo off
chcp 65001 >nul
setlocal

set "EXIT_CODE=0"
set "NEED_PAUSE=0"
echo %CMDCMDLINE% | findstr /i /c:" /c " >nul && set "NEED_PAUSE=1"

set "TARGET_DIR=%~1"
if not defined TARGET_DIR set "TARGET_DIR=%~dp0"

if not exist "%TARGET_DIR%" (
    echo 错误: 找不到指定的路径。
  set "EXIT_CODE=1"
  goto :finish
)

if not exist "%TARGET_DIR%\*" (
    echo 错误: 指定的路径不是一个文件夹。
  set "EXIT_CODE=1"
  goto :finish
)

echo.
echo 正在扫描目录: %TARGET_DIR%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$path = $env:TARGET_DIR;" ^
  "function Format-Size([double]$size) {" ^
  "  $units = 'B','KB','MB','GB','TB';" ^
  "  $index = 0;" ^
  "  while ($size -ge 1024 -and $index -lt ($units.Length - 1)) { $size /= 1024; $index++ }" ^
  "  return ('{0:N2} {1}' -f $size, $units[$index]);" ^
  "}" ^
  "function Get-DirSize([string]$dirPath) {" ^
  "  $sum = (Get-ChildItem -LiteralPath $dirPath -File -Recurse -ErrorAction SilentlyContinue |" ^
  "    Where-Object { -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) } |" ^
  "    Measure-Object -Property Length -Sum).Sum;" ^
  "  if ($null -eq $sum) { return 0 }" ^
  "  return [double]$sum;" ^
  "}" ^
  "function Show-Current([string]$currentPath) {" ^
  "  Write-Host '';" ^
  "  Write-Host ('当前目录: {0}' -f $currentPath);" ^
  "  Write-Host ('=' * 90);" ^
  "  $files = Get-ChildItem -LiteralPath $currentPath -File -ErrorAction SilentlyContinue |" ^
  "    Where-Object { -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) } |" ^
  "    Sort-Object -Property Length -Descending;" ^
  "  Write-Host '文件大小排序:';" ^
  "  if (-not $files -or $files.Count -eq 0) {" ^
  "    Write-Host '  (当前目录没有文件)';" ^
  "  } else {" ^
  "    Write-Host ('  {0,-42} | {1}' -f 'File Name', 'Size');" ^
  "    Write-Host ('  ' + ('-' * 62));" ^
  "    foreach ($file in $files) {" ^
  "      Write-Host ('  {0,-42} | {1}' -f $file.Name, (Format-Size $file.Length));" ^
  "    }" ^
  "  }" ^
  "  Write-Host '';" ^
  "  $dirs = Get-ChildItem -LiteralPath $currentPath -Directory -ErrorAction SilentlyContinue;" ^
  "  $dirInfo = @();" ^
  "  foreach ($d in $dirs) {" ^
  "    $dirInfo += [PSCustomObject]@{ Name = $d.Name; FullName = $d.FullName; Size = (Get-DirSize $d.FullName) };" ^
  "  }" ^
  "  $dirInfo = $dirInfo | Sort-Object -Property Size -Descending;" ^
  "  Write-Host '子文件夹大小排序:';" ^
  "  if (-not $dirInfo -or $dirInfo.Count -eq 0) {" ^
  "    Write-Host '  (当前目录没有子文件夹可进入)';" ^
  "  } else {" ^
  "    Write-Host ('  {0,3} | {1,-34} | {2}' -f 'No.', 'Folder Name', 'Size');" ^
  "    Write-Host ('  ' + ('-' * 62));" ^
  "    for ($i = 0; $i -lt $dirInfo.Count; $i++) {" ^
  "      Write-Host ('  {0,3} | {1,-34} | {2}' -f ($i + 1), $dirInfo[$i].Name, (Format-Size $dirInfo[$i].Size));" ^
  "    }" ^
  "  }" ^
  "  return ,$dirInfo;" ^
  "}" ^
  "function Resolve-InputPath([string]$basePath, [string]$inputPath) {" ^
  "  if ([string]::IsNullOrWhiteSpace($inputPath)) { return $null }" ^
  "  $raw = $inputPath.Trim();" ^
  "  $raw = $raw.Trim([char]34);" ^
  "  if ($raw -match '^[A-Za-z]:\\' -or $raw -like '\\*') {" ^
  "    return $raw;" ^
  "  }" ^
  "  return [System.IO.Path]::GetFullPath((Join-Path -Path $basePath -ChildPath $raw));" ^
  "}" ^
  "function Normalize-DirPath([string]$p) {" ^
  "  $resolved = (Resolve-Path -LiteralPath $p).Path;" ^
  "  if ($resolved -match '^[A-Za-z]:\\$') {" ^
  "    return $resolved;" ^
  "  }" ^
  "  return $resolved.TrimEnd('\');" ^
  "}" ^
  "if (-not (Test-Path -LiteralPath $path -PathType Container)) {" ^
  "  Write-Host '错误: 找不到指定的路径。';" ^
  "  exit 1;" ^
  "}" ^
  "$currentPath = Normalize-DirPath -p $path;" ^
  "while ($true) {" ^
  "  $dirInfo = Show-Current -currentPath $currentPath;" ^
  "  Write-Host '';" ^
  "  Write-Host '操作: 输入序号进入子文件夹, b 返回上一级, cd 路径 进入任意文件夹, q 退出';" ^
  "  $choice = Read-Host '请输入';" ^
  "  if ([string]::IsNullOrWhiteSpace($choice)) { continue }" ^
  "  $choice = $choice.Trim();" ^
  "  if ($choice -match '^[Qq]$') { break }" ^
  "  if ($choice -match '^(?i)cd\s+(.+)$') {" ^
  "    $targetInput = $Matches[1];" ^
  "    $targetPath = Resolve-InputPath -basePath $currentPath -inputPath $targetInput;" ^
  "    if (-not $targetPath -or -not (Test-Path -LiteralPath $targetPath -PathType Container)) {" ^
  "      Write-Host '目标路径不存在或不是文件夹。';" ^
  "      continue;" ^
  "    }" ^
  "    $currentPath = Normalize-DirPath -p $targetPath;" ^
  "    continue;" ^
  "  }" ^
  "  if ($choice -match '^[Bb]$') {" ^
  "    $parentPath = [System.IO.Path]::GetFullPath((Join-Path -Path $currentPath -ChildPath '..'));" ^
  "    if ((Normalize-DirPath -p $parentPath) -eq (Normalize-DirPath -p $currentPath)) {" ^
  "      Write-Host '已在磁盘根目录，无法再返回。';" ^
  "    } else {" ^
  "      $currentPath = Normalize-DirPath -p $parentPath;" ^
  "    }" ^
  "    continue;" ^
  "  }" ^
  "  if ($choice -notmatch '^[0-9]+$') {" ^
  "    Write-Host '输入无效，请输入序号、b、cd 路径 或 q。';" ^
  "    continue;" ^
  "  }" ^
  "  if (-not $dirInfo -or $dirInfo.Count -eq 0) {" ^
  "    Write-Host '当前目录没有可进入的子文件夹。';" ^
  "    continue;" ^
  "  }" ^
  "  $idx = [int]$choice;" ^
  "  if ($idx -lt 1 -or $idx -gt $dirInfo.Count) {" ^
  "    Write-Host ('输入超出范围，请输入 1 到 {0}。' -f $dirInfo.Count);" ^
  "    continue;" ^
  "  }" ^
  "  $currentPath = Normalize-DirPath -p $dirInfo[$idx - 1].FullName;" ^
  "}"

set "EXIT_CODE=%ERRORLEVEL%"

:finish
if "%NO_PAUSE%"=="" if "%NEED_PAUSE%"=="1" pause
exit /b %EXIT_CODE%