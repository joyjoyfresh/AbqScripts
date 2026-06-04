# -*- coding: utf-8 -*-  # 指定源文件编码为UTF-8
"""
将指定扩展名的文件递归扫描并送入回收站的交互式脚本。

等价于 Delete_filetype.bat + Delete_filetype_helper.ps1 的 Python 实现：
- 默认从脚本所在目录开始递归扫描；可通过命令行参数指定目标目录。
- 先扫描并按扩展名统计数量，确认后再将匹配文件送入回收站（而非永久删除）。
"""

import os  # 导入os模块，用于路径与文件属性操作
import sys  # 导入sys模块，用于读取命令行参数
import stat  # 导入stat模块，用于处理只读属性
from pathlib import Path  # 从pathlib导入Path，便于路径与递归遍历

# 在Windows下将标准输入输出统一设为UTF-8，避免中文乱码（等价于批处理的chcp 65001）
if sys.platform == "win32":  # 判断是否为Windows平台
    try:  # 尝试重新配置流编码
        sys.stdout.reconfigure(encoding="utf-8")  # 将标准输出编码设为UTF-8
        sys.stdin.reconfigure(encoding="utf-8")  # 将标准输入编码设为UTF-8
    except Exception:  # 处理老版本Python不支持reconfigure的情况
        pass  # 忽略错误，沿用默认编码

# 默认扫描的扩展名列表（与原批处理保持一致）
EXTS_DEFAULT = ["inp", "odb", "jnl", "msg", "prt", "dat", "sta", "sim", "com"]


def send_to_recycle_bin(file_path):  # 定义将单个文件送入回收站的函数
    """
    file_path: 待删除文件的绝对路径字符串
    返回值: 成功送入回收站返回True，否则返回False
    优先使用 send2trash 库；若不可用则回退到 Windows 的 SHFileOperation。
    """
    try:  # 尝试使用send2trash库
        from send2trash import send2trash  # 导入send2trash函数
        send2trash(file_path)  # 将文件送入回收站
        return True  # 成功则返回True
    except ImportError:  # 处理send2trash未安装的情况
        pass  # 跳过，改用系统API回退方案
    except Exception:  # 处理send2trash删除过程中的其他异常
        return False  # 删除失败返回False

    try:  # 回退方案：调用Windows原生SHFileOperation
        import ctypes  # 导入ctypes，用于调用Win32 API
        from ctypes import wintypes  # 导入wintypes，提供Windows类型定义

        class SHFILEOPSTRUCT(ctypes.Structure):  # 定义SHFILEOPSTRUCT结构体
            _fields_ = [  # 结构体字段列表
                ("hwnd", wintypes.HWND),  # 父窗口句柄
                ("wFunc", wintypes.UINT),  # 操作类型（删除等）
                ("pFrom", wintypes.LPCWSTR),  # 源文件路径（双空字符结尾）
                ("pTo", wintypes.LPCWSTR),  # 目标路径（删除时为空）
                ("fFlags", ctypes.c_uint16),  # 操作标志位
                ("fAnyOperationsAborted", wintypes.BOOL),  # 是否有操作被中止
                ("hNameMappings", ctypes.c_void_p),  # 名称映射句柄
                ("lpszProgressTitle", wintypes.LPCWSTR),  # 进度对话框标题
            ]

        FO_DELETE = 3  # 操作类型：删除
        FOF_ALLOWUNDO = 0x0040  # 标志：允许撤销（即送入回收站）
        FOF_NOCONFIRMATION = 0x0010  # 标志：不弹出确认对话框
        FOF_SILENT = 0x0004  # 标志：不显示进度界面

        op = SHFILEOPSTRUCT()  # 创建结构体实例
        op.wFunc = FO_DELETE  # 设置为删除操作
        op.pFrom = file_path + "\0\0"  # 源路径需以双空字符结尾
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT  # 组合标志位
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))  # 执行文件操作
        return result == 0  # 返回0表示成功
    except Exception:  # 处理回退方案中的异常
        return False  # 失败返回False


def clear_readonly(path):  # 定义清除只读属性的函数
    """path: 文件路径；若为只读则尝试改为可写，避免删除失败。"""
    try:  # 尝试修改文件权限
        if not os.access(path, os.W_OK):  # 判断文件是否不可写
            os.chmod(path, stat.S_IWRITE)  # 添加可写权限
    except Exception:  # 处理修改权限失败的情况
        pass  # 忽略错误，继续后续删除尝试


def scan_files(root, exts):  # 定义按扩展名递归扫描文件的函数
    """
    root: 扫描根目录的Path对象
    exts: 已清洗的扩展名列表（不含点号）
    返回值: 字典 {扩展名: [匹配文件Path列表]}
    """
    result = {ext: [] for ext in exts}  # 初始化每个扩展名对应的空列表
    for ext in exts:  # 遍历每个扩展名
        for f in root.rglob("*." + ext):  # 递归匹配该扩展名的所有文件
            if f.is_file():  # 仅保留普通文件（排除目录）
                result[ext].append(f)  # 将匹配文件加入对应列表
    return result  # 返回扫描结果字典


def main():  # 定义主函数
    """脚本主流程：读取目录、获取扩展名、扫描、确认、删除。"""
    if len(sys.argv) > 1:  # 判断是否提供了命令行目录参数
        root = Path(sys.argv[1]).resolve()  # 使用参数指定的目录并转为绝对路径
    else:  # 未提供参数时
        root = Path(__file__).resolve().parent  # 默认使用脚本所在目录

    if not root.is_dir():  # 判断根目录是否存在且为文件夹
        print("错误: 找不到指定的路径或它不是一个文件夹。")  # 提示路径错误
        sys.exit(1)  # 以错误码退出

    # 打印默认扩展名提示信息
    print("默认扩展名: ." + "  .".join(EXTS_DEFAULT))  # 显示默认扩展名列表
    print("留空使用默认值，或输入以空格分隔的扩展名（例如: odb sta msg）:")  # 提示输入方式
    exts_input = input("要删除的扩展名: ").strip()  # 读取用户输入并去除首尾空白

    if not exts_input:  # 判断输入是否为空
        exts = list(EXTS_DEFAULT)  # 为空则使用默认扩展名
        print("使用默认值: ." + "  .".join(EXTS_DEFAULT))  # 提示正在使用默认值
    else:  # 处理用户提供输入的情况
        exts = [e.lstrip(".") for e in exts_input.split() if e.lstrip(".")]  # 拆分并去除前导点号

    if not exts:  # 判断清洗后是否还有有效扩展名
        print("没有有效的扩展名，操作已取消。")  # 提示无有效扩展名
        sys.exit(1)  # 以错误码退出

    print()  # 输出空行
    print('正在递归扫描 "%s" ...' % root)  # 提示正在扫描的目录
    print()  # 输出空行

    found = scan_files(root, exts)  # 执行扫描得到结果字典
    total_found = sum(len(v) for v in found.values())  # 统计匹配文件总数

    for ext in exts:  # 遍历每个扩展名输出统计
        count = len(found[ext])  # 取该扩展名的文件数量
        print("  ." + ext.ljust(8) + str(count).rjust(5) + " 个文件")  # 按格式打印统计行

    print()  # 输出空行
    print("总计: %d 个文件待送入回收站。" % total_found)  # 打印待删除文件总数

    if total_found == 0:  # 判断是否没有匹配文件
        print("没有匹配的文件，无需操作。")  # 提示无需操作
        sys.exit(0)  # 正常退出

    print()  # 输出空行
    answer = input("将以上所有文件送入回收站? [y/N]: ").strip().lower()  # 读取确认输入并转小写
    if answer not in ("y", "yes"):  # 判断用户是否确认
        print("操作已被用户取消，未删除任何文件。")  # 提示已取消
        sys.exit(0)  # 正常退出

    print()  # 输出空行
    print("正在送入回收站...")  # 提示开始删除

    total_recycled = 0  # 初始化成功删除计数
    for ext in exts:  # 遍历每个扩展名执行删除
        recycled = 0  # 初始化当前扩展名的删除计数
        for f in found[ext]:  # 遍历该扩展名下的每个文件
            clear_readonly(str(f))  # 删除前清除只读属性
            if send_to_recycle_bin(str(f)):  # 尝试将文件送入回收站
                recycled += 1  # 成功则计数加一
        total_recycled += recycled  # 累加到总成功数
        if len(found[ext]) > 0:  # 仅对有匹配文件的扩展名输出
            print("  ." + ext.ljust(8) + "已回收 " + str(recycled) + " / " + str(len(found[ext])))  # 打印删除结果

    print()  # 输出空行
    print("完成。共回收: %d / %d 个文件。" % (total_recycled, total_found))  # 打印最终汇总


if __name__ == "__main__":  # 判断是否作为主程序运行
    main()  # 调用主函数
