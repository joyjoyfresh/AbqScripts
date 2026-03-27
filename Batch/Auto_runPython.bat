@echo off
rem 关闭命令回显，保持输出整洁
chcp 65001 >nul
rem 切换到 UTF-8 代码页，避免中文乱码
setlocal enabledelayedexpansion
rem 启用延迟变量扩展，便于循环中读取变量

set "SCRIPT_DIR=%~dp0"
rem 获取当前批处理所在目录
set /a count=0
rem 初始化 Python 文件计数器
set "TMP_LIST=%TEMP%\py_order_%RANDOM%%RANDOM%.txt"
rem 定义临时文件路径，用于保存排序清单
set "TMP_SORTED=%TMP_LIST%.sorted"
rem 定义排序后的临时文件路径

for /f "delims=" %%f in ('dir /b /a-d "%SCRIPT_DIR%*.py" 2^>nul') do (
    rem 统计当前目录中的 .py 文件数量
    set /a count+=1
    rem 计数器递增
)

if %count%==0 (
    rem 如果没有任何 Python 文件则提示并退出
    echo  未找到任何 .py 文件。
    exit /b
)

if exist "%TMP_LIST%" del /q "%TMP_LIST%"
rem 若历史临时文件存在则先删除
if exist "%TMP_SORTED%" del /q "%TMP_SORTED%"
rem 若历史排序文件存在则先删除

for /f "delims=" %%f in ('dir /b /a-d "%SCRIPT_DIR%*.py" 2^>nul') do (
    rem 遍历目录中的每个 Python 文件名
    set "BASE_NAME=%%~nf"
    rem 取不带扩展名的文件名
    set "NUM_PART=999999999"
    rem 默认将无前导数字的文件放到最后
    for /f "tokens=1 delims=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-. " %%n in ("!BASE_NAME!") do (
        rem 尝试从文件名开头提取连续数字
        set "NUM_PART=%%n"
        rem 保存提取到的数字片段
    )
    set "SORT_KEY=000000000!NUM_PART!"
    rem 左侧补零，确保按数值顺序排序
    set "SORT_KEY=!SORT_KEY:~-9!"
    rem 仅保留最后9位作为固定长度排序键
    echo !SORT_KEY!^|%%~f>>"%TMP_LIST%"
    rem 将排序键与完整路径写入临时清单
)

sort "%TMP_LIST%" /o "%TMP_SORTED%"
rem 对临时清单按排序键进行升序排序

for /f "tokens=1,* delims=|" %%a in ('type "%TMP_SORTED%"') do (
    rem 读取排序后的清单并拆分排序键与路径
    echo ================================================
    rem 打印分隔线，便于阅读日志
    echo  正在执行: %%~nxb
    rem 显示当前正在执行的脚本名
    echo ================================================
    rem 打印分隔线，便于阅读日志
    python "%%~b"
    rem 调用 Python 解释器执行当前脚本
    echo.
    rem 输出空行分隔每次执行结果
)

if exist "%TMP_LIST%" del /q "%TMP_LIST%"
rem 删除执行过程中的临时清单文件
if exist "%TMP_SORTED%" del /q "%TMP_SORTED%"
rem 删除排序结果临时文件

echo  全部执行完毕，窗口即将自动关闭...
rem 所有脚本执行完成后给出结束提示
exit /b 0
rem 正常退出批处理，终端窗口自动关闭
