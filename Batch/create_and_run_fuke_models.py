# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""批量创建 fuke-10-* 文件夹并按顺序执行脚本。"""  # 说明脚本用途

import os  # 导入操作系统路径与目录模块
import re  # 导入正则模块用于替换参数
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器

ROOT_DIR = r"D:\Abaqus\fuke"  # 设置目标模型根目录
FOLDER_PREFIX = "fuke-10-"  # 设置目标文件夹前缀
FOLDER_START = 1  # 设置起始文件夹编号
FOLDER_END = 12  # 设置结束文件夹编号
STOP_ON_ERROR = True  # 设置出现脚本错误时是否立即停止

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)  # 获取工作区根目录

SOURCE_FILES = {  # 定义需要复制到每个模型目录的源文件映射
    "El_Centro_scaled.txt": os.path.join(WORKSPACE_DIR, "Wave", "Seismic", "El_Centro_scaled.txt"),  # 指定 El_Centro 加速度时程文件来源
    "Loma_Prieta_scaled.txt": os.path.join(WORKSPACE_DIR, "Wave", "Seismic", "Loma_Prieta_scaled.txt"),  # 指定 Loma_Prieta 加速度时程文件来源
    "Northridge_scaled.txt": os.path.join(WORKSPACE_DIR, "Wave", "Seismic", "Northridge_scaled.txt"),  # 指定 Northridge 加速度时程文件来源
    "VAB_oblique_noGUI_v13.py": os.path.join(WORKSPACE_DIR, "Modeling", "Single", "VAB_oblique_noGUI_v13.py"),  # 指定建模脚本来源
    "Distribution_PGA_v4.py": os.path.join(WORKSPACE_DIR, "Postprocess", "Distribution_PGA_v4.py"),  # 指定分布后处理脚本来源
    "Postprocess_PGA_v6.py": os.path.join(WORKSPACE_DIR, "Postprocess", "Postprocess_PGA_v6.py"),  # 指定 PGA 后处理脚本来源
}  # 结束源文件映射定义

RUN_ORDER = ["VAB_oblique_noGUI_v13.py", "Postprocess_PGA_v6.py", "Distribution_PGA_v4.py"]  # 定义每个文件夹内脚本执行顺序


def ensure_sources_exist():  # 定义检查源文件是否存在的函数
    missing = []  # 初始化缺失文件列表
    for name, path in SOURCE_FILES.items():  # 遍历所有源文件映射
        if not os.path.isfile(path):  # 判断当前源文件是否存在
            missing.append((name, path))  # 记录缺失文件信息
    if missing:  # 判断是否存在缺失文件
        print("错误：以下源文件不存在，请先检查路径：")  # 输出错误提示标题
        for name, path in missing:  # 遍历缺失文件信息
            print("  - {} -> {}".format(name, path))  # 输出缺失文件明细
        return False  # 返回检查失败
    return True  # 返回检查成功


def update_mesh_size_manual(vab_script_path, mesh_value):  # 定义更新网格参数的函数
    with open(vab_script_path, "r", encoding="utf-8", errors="ignore") as f:  # 读取目标脚本文本
        content = f.read()  # 读取完整脚本内容
    pattern = r"('mesh_size_manual'\s*:\s*)(\d+(?:\.\d+)?)"  # 定义参数匹配正则表达式
    replaced_content, replace_count = re.subn(pattern, r"\g<1>{}".format(mesh_value), content, count=1)  # 仅替换首个匹配项
    if replace_count == 0:  # 判断是否未匹配到目标参数
        raise ValueError("未在 {} 中找到参数 'mesh_size_manual'".format(vab_script_path))  # 抛出参数缺失异常
    with open(vab_script_path, "w", encoding="utf-8", newline="") as f:  # 以写入模式覆盖脚本内容
        f.write(replaced_content)  # 写回替换后的脚本文本
    print("已更新 {} 的 mesh_size_manual={}".format(vab_script_path, mesh_value))  # 输出参数更新结果


def create_and_fill_folder(folder_path, folder_index):  # 定义创建并填充单个模型目录的函数
    os.makedirs(folder_path, exist_ok=True)  # 创建目标目录并允许已存在
    for target_name, src_path in SOURCE_FILES.items():  # 遍历所有需要复制的文件
        dst_path = os.path.join(folder_path, target_name)  # 拼接目标文件完整路径
        shutil.copy2(src_path, dst_path)  # 复制文件并保留时间戳等元数据
    vab_script_path = os.path.join(folder_path, "VAB_oblique_noGUI_v13.py")  # 定位当前文件夹中的建模脚本
    update_mesh_size_manual(vab_script_path, folder_index)  # 将网格参数更新为当前文件夹编号


def run_scripts_in_folder(folder_path):  # 定义在单个目录内按顺序执行脚本的函数
    for script_name in RUN_ORDER:  # 按预设顺序遍历脚本
        script_path = os.path.join(folder_path, script_name)  # 生成当前脚本完整路径
        if not os.path.isfile(script_path):  # 检查当前脚本是否存在
            print("错误：脚本不存在 -> {}".format(script_path))  # 输出脚本缺失错误
            return False  # 返回执行失败
        print("开始执行：{}".format(script_path))  # 输出脚本开始执行信息
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)  # 使用当前 Python 解释器在目标目录执行脚本
        if result.returncode != 0:  # 判断脚本是否执行失败
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode))  # 输出脚本失败信息
            return False  # 返回执行失败
        print("完成执行：{}".format(script_name))  # 输出脚本执行完成信息
    return True  # 返回全部脚本执行成功


def main():  # 定义主函数
    root_dir = ROOT_DIR  # 使用默认根目录
    if len(sys.argv) >= 2:  # 判断是否通过命令行传入根目录参数
        root_dir = sys.argv[1]  # 使用命令行传入的根目录覆盖默认值

    print("目标根目录：{}".format(root_dir))  # 输出本次运行根目录
    if not ensure_sources_exist():  # 检查所有源文件可用性
        sys.exit(1)  # 源文件缺失时退出程序

    os.makedirs(root_dir, exist_ok=True)  # 确保根目录存在
    failed_folders = []  # 初始化失败文件夹记录列表

    for index in range(FOLDER_START, FOLDER_END + 1):  # 按编号顺序遍历所有目标文件夹
        folder_name = "{}{}".format(FOLDER_PREFIX, index)  # 生成文件夹名称
        folder_path = os.path.join(root_dir, folder_name)  # 生成文件夹完整路径
        print("\n==============================")  # 输出分隔线便于阅读日志
        print("处理文件夹：{}".format(folder_path))  # 输出当前处理文件夹信息
        create_and_fill_folder(folder_path, index)  # 创建并填充当前文件夹并同步网格参数
        ok = run_scripts_in_folder(folder_path)  # 在当前文件夹内按顺序执行脚本
        if not ok:  # 判断当前文件夹是否执行失败
            failed_folders.append(folder_path)  # 记录失败文件夹路径
            if STOP_ON_ERROR:  # 判断是否配置为失败即停止
                print("检测到失败，按配置停止后续文件夹处理。")  # 输出停止处理提示
                break  # 跳出循环结束批处理

    print("\n==============================")  # 输出结束分隔线
    if failed_folders:  # 判断是否存在失败文件夹
        print("批处理结束：存在失败文件夹。")  # 输出失败总结标题
        for path in failed_folders:  # 遍历失败文件夹列表
            print("  - {}".format(path))  # 输出失败文件夹路径
        sys.exit(2)  # 以非零状态码退出
    print("批处理结束：全部文件夹处理完成。")  # 输出全部成功信息


if __name__ == "__main__":  # 判断是否为直接运行入口
    main()  # 执行主函数