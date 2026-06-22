# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软楔 2D 共振【一键研究】编排脚本（Python 形式，与 Autorun 同套路）。

顺序执行三步，串成一次跑完：
  ① 模态提取(快)：判定坡顶软楔 2D 共振模是否存在(eigen_softwedge_v1.py)；
  ② 收敛批处理(慢)：7 工况 网格(4/2/1m)×单元(CPE4R/CPE4/CPE8R)×尾段(0/4s)(Autorun_softwedge_convergence_v1.py)；
  ③ 汇总报告：对照表 + 验证闸门 + 关键对照(softwedge_report_v1.py)。

运行方式（与你跑 Autorun 系列一致，用同一个能跑 Abaqus 的 Python）：
    python run_softwedge_study_v1.py [可选:工作目录]
  不传工作目录则用下面的 WORKDIR 默认值。所有输出(模态ODB/各工况/报告)都进工作目录。

设计：各步用 subprocess 以【当前 Python 解释器(sys.executable)】调用，
      与 Autorun 编排脚本 spawn 建模脚本的方式完全相同，故运行环境一致、无需额外配置。
"""

import os  # 路径与目录
import sys  # 解释器路径与命令行参数
import subprocess  # 子进程调用各步脚本

# ===================== 配置(按需修改) =====================
REPO = r"C:\Users\12462\Documents\Code\AbqScripts"  # 仓库根目录
EIGEN = os.path.join(REPO, "Modeling", "Multi", "test", "eigen_softwedge_v1.py")  # ① 模态提取脚本
CONV = os.path.join(REPO, "Batch", "Autorun_softwedge_convergence_v1.py")  # ② 收敛批处理(编排)脚本
REPORT = os.path.join(REPO, "Modeling", "Multi", "test", "softwedge_report_v1.py")  # ③ 结果汇总脚本
WORKDIR_DEFAULT = r"C:\Abaqus\softwedge_study"  # 默认工作目录(模态ODB+各工况+报告都放这里)
# ==========================================================


def _banner(text):  # 打印醒目分隔横幅
    """在步骤之间打印统一格式的标题，便于在长日志里定位。"""
    print("\n" + "=" * 64)  # 上分隔线
    print(" " + text)  # 标题文本
    print("=" * 64)  # 下分隔线


def _run_step(cmd, cwd=None, allow_fail=False):  # 运行一步并处理返回码
    """以当前 Python 解释器运行一个脚本步骤。

    cmd        : [脚本路径, 其余参数...]（会自动在前面加 sys.executable）
    cwd        : 子进程工作目录(模态步用工作目录，使 ODB/summary 落在那里)
    allow_fail : True=失败仅告警继续；False=失败则中止整个流程
    返回：子进程返回码(0 表示成功)。
    """
    full = [sys.executable] + cmd  # 用当前解释器执行(与 Autorun 同口径)
    script = os.path.basename(cmd[0])  # 脚本名(仅日志用)
    if not os.path.isfile(cmd[0]):  # 脚本缺失
        print("错误：脚本不存在 -> {}".format(cmd[0]))
        if not allow_fail:
            sys.exit(1)
        return 1
    print("开始执行：{}{}".format(script, ("  (cwd=%s)" % cwd) if cwd else ""))  # 开始日志
    result = subprocess.run(full, cwd=cwd, check=False)  # 运行子进程
    if result.returncode != 0:  # 失败
        msg = "{} 返回非零(返回码={})".format(script, result.returncode)
        if allow_fail:  # 允许失败：告警继续
            print("[警告] {}；继续后续步骤。".format(msg))
        else:  # 不允许失败：中止
            print("错误：{}，流程中止。".format(msg))
            sys.exit(2)
    else:
        print("完成执行：{}".format(script))  # 成功日志
    return result.returncode  # 返回码


def main():  # 主流程：依次跑 模态→收敛→汇总
    """组织三步一键执行；工作目录可由命令行参数覆盖默认值。"""
    workdir = sys.argv[1] if len(sys.argv) >= 2 else WORKDIR_DEFAULT  # 工作目录(可命令行覆盖)
    if not os.path.isdir(workdir):  # 工作目录不存在则创建
        os.makedirs(workdir, exist_ok=True)
    print("工作目录：{}".format(workdir))  # 打印工作目录
    print("Python 解释器：{}".format(sys.executable))  # 打印解释器(便于确认是能跑 Abaqus 的那个)

    # ① 模态提取(快)：cwd=工作目录，使 softwedge_eigen.odb / eigen_softwedge_summary.txt 落在这里
    _banner("步骤 1/3 : 软楔模态提取(快, 判定 2D 共振模是否存在)")
    _run_step([EIGEN], cwd=workdir, allow_fail=True)  # 模态失败不阻断后续(可单独排查)

    # ② 收敛批处理(慢)：把工作目录作为根目录传入，各工况文件夹建在其下
    _banner("步骤 2/3 : 软楔收敛批处理(慢, 7 工况; 含 CPE8R 二次单元, 由建模脚本 v3 支持)")
    _run_step([CONV, workdir], cwd=None, allow_fail=True)  # 部分工况失败仍尝试汇总已完成的

    # ③ 汇总报告：读各工况 TAF + case_meta，出对照表/验证闸门/关键对照
    _banner("步骤 3/3 : 汇总对照 + 验证闸门")
    _run_step([REPORT, workdir], cwd=None, allow_fail=True)

    # 收尾提示
    _banner("全部完成")
    print("输出位置：{}".format(workdir))
    print("  - eigen_softwedge_summary.txt / softwedge_eigen.odb   (模态: 共振模是否存在)")
    print("  - multi-conv-*  各工况 + results\\index.csv             (收敛: 各工况 TAF)")
    print("  - softwedge_report.txt                                 (对照表 + 验证闸门 + 关键对照)")
    print("把 softwedge_report.txt 发回即可定论。")


if __name__ == "__main__":  # 主入口
    main()  # 运行一键流程
