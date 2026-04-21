@echo off & rem 关闭命令回显以保持输出简洁
setlocal EnableExtensions EnableDelayedExpansion & rem 启用命令扩展和延迟变量展开

set "ROOT=%~dp0" & rem 将脚本所在目录设置为递归扫描根目录
set /a COUNT=0 & rem 初始化成功移入回收站计数
set /a FOUND=0 & rem 初始化匹配文件计数
set "EXT=" & rem 初始化用户输入的扩展名变量
set "PATTERN=" & rem 初始化用于匹配文件的通配模式变量

set /p "EXT=请输入要删除的扩展名（如 odb 或 .odb）: " & rem 提示用户输入要删除的文件扩展名
if not defined EXT ( & rem 判断用户是否未输入任何内容
    echo 未输入扩展名，操作已取消。 & rem 输出未输入时的取消提示
    pause & rem 暂停以便用户查看提示
    endlocal & rem 结束局部环境并恢复变量环境
    exit /b 1 & rem 以非零状态退出表示未执行删除
) & rem 结束未输入分支

if "!EXT:~0,1!"=="." set "EXT=!EXT:~1!" & rem 若用户输入以点开头则去掉前导点
if not defined EXT ( & rem 再次判断去掉前导点后是否为空
    echo 扩展名无效，操作已取消。 & rem 输出无效扩展名提示
    pause & rem 暂停以便用户查看提示
    endlocal & rem 结束局部环境并恢复变量环境
    exit /b 1 & rem 以非零状态退出表示参数无效
) & rem 结束无效扩展名分支

set "PATTERN=*.!EXT!" & rem 生成用于 dir 搜索的通配模式
echo Scanning "%ROOT%" and subfolders for !PATTERN! files... & rem 显示当前扫描范围和目标文件类型
choice /m "确认删除上述范围内的 !PATTERN! 文件" & rem 提示用户确认是否继续执行删除
if errorlevel 2 ( & rem 当用户选择“否”时进入取消分支
    echo Operation canceled by user. No files were deleted. & rem 输出取消信息
    pause & rem 暂停以便用户查看提示
    endlocal & rem 结束局部环境并恢复变量环境
    exit /b 0 & rem 正常退出脚本且不执行删除
) & rem 结束取消分支

for /f "delims=" %%F in ('dir /s /b /a:-d "%ROOT%\!PATTERN!" 2^>nul') do ( & rem 递归遍历所有匹配扩展名的文件
    set /a FOUND+=1 & rem 匹配文件总数加一
    attrib -r "%%F" >nul 2>&1 & rem 去除只读属性避免删除失败
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName Microsoft.VisualBasic; $p=$args[0]; if (Test-Path -LiteralPath $p) {[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p,'OnlyErrorDialogs','SendToRecycleBin')}" "%%~fF" >nul 2>&1 & rem 调用 PowerShell 将当前文件移入回收站
    if not exist "%%F" set /a COUNT+=1 & rem 若文件已不存在则统计为移入回收站成功
) & rem 结束遍历删除循环

if !FOUND! EQU 0 ( & rem 判断是否没有找到任何匹配文件
    echo No !PATTERN! files found. Nothing to recycle. & rem 输出未找到匹配文件提示
) else ( & rem 存在匹配文件时进入结果输出分支
    echo Done. Matched !FOUND! files, recycled !COUNT! files. & rem 输出匹配数量与成功移入回收站数量
) & rem 结束结果输出分支

pause & rem 暂停以便用户查看最终结果
endlocal & rem 结束局部环境并恢复变量环境
