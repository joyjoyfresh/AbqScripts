# -*- coding: utf-8 -*-
"""ML v3 — Stage 4 出版级对照图表。

整合 Stage 0~3 的全部指标 JSON，产出论文级「v2 深度学习 vs v3 经典 ML」对照图。
遵循项目出版约定：中文核心期刊样式（宋体正文 + Times New Roman 数字混排）、
色盲安全配色、合成图 + 各子面板同时输出到「同名文件夹 + panels/」。

数据来源（均在 outputs_v3/ 与 outputs_v2/）：
  diagnostics_v3.json       — v2 四深度模型 TAF_h MAE
  baseline_metrics.json     — B1/B2/B3 基线
  train_metrics.json        — v3 XGBoost / GPR / 留一波（TAF_h）
  evaluate_metrics.json     — v3 四通道 + 留一高度/角度
  outputs_v2/summary_metrics.json — v2 四深度模型全通道指标

产出（outputs_v3/figures_pub/）：
  fig_v2_vs_v3/fig_v2_vs_v3.{png,pdf}   — 合成主图（4 面板）
  fig_v2_vs_v3/panels/*.{png,pdf}        — 各子面板单独文件
  table_final.md                          — 论文用最终对照表（Markdown）

运行：py -3 ML/figures_v3.py
"""

import os  # 路径操作
import json  # JSON 读写
import numpy as np  # 数值计算
import matplotlib  # 绘图框架
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt  # 绘图子模块
import matplotlib.font_manager as fm  # 字体管理（检测可用中文衬线）

# ==========================================================================
# 配置区
# ==========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本目录
OUT_V3     = os.path.join(SCRIPT_DIR, "outputs_v3")  # v3 输出目录
OUT_V2     = os.path.join(SCRIPT_DIR, "outputs_v2")  # v2 输出目录
PUB_DIR    = os.path.join(OUT_V3, "figures_pub")  # 出版图根目录
os.makedirs(PUB_DIR, exist_ok=True)  # 创建出版图目录

# 色盲安全配色（Okabe-Ito 调色板子集）
CB = {
    "gray":   "#999999",  # 中性灰（弱基线）
    "blue":   "#0072B2",  # 蓝（v3 XGBoost）
    "green":  "#009E73",  # 绿（v3 GPR / 达标）
    "orange": "#E69F00",  # 橙（v2 深度模型）
    "red":    "#D55E00",  # 朱红（v2 最强 / 病态警示）
    "purple": "#CC79A7",  # 紫（外推）
    "sky":    "#56B4E9",  # 天蓝（网格插值）
    "yellow": "#F0E442",  # 黄
}


def setup_cn_journal_style():
    """配置中文核心期刊样式：Times New Roman 数字 + 中文衬线（宋体）混排。

    关键：font.family 设为【真实字体名列表】而非 'serif'，靠 matplotlib 逐字形
    回退实现混排——拉丁字符走 Times New Roman，汉字回退到中文衬线字体。
    """
    avail = set(f.name for f in fm.fontManager.ttflist)  # 本机已注册字体名
    cn_serif = next((n for n in  # 按优先级选第一个可用中文衬线
                     ["SimSun", "Noto Serif SC", "Source Han Serif SC", "STSong"]
                     if n in avail), "SimHei")  # 均无则退黑体
    plt.rcParams.update({
        "font.family":        ["Times New Roman", cn_serif, "DejaVu Serif"],  # 真实字体名列表
        "axes.unicode_minus": False,   # 用 ASCII 连字符当负号（避免缺字形方块）
        "mathtext.fontset":   "stix",  # 数学公式用类 Times 字体
        "font.size":          11,      # 基础字号
        "axes.titlesize":     12,      # 子图标题
        "axes.labelsize":     11,      # 轴标签
        "xtick.labelsize":    10,      # x 刻度
        "ytick.labelsize":    10,      # y 刻度
        "legend.fontsize":    9,       # 图例
        "axes.linewidth":     0.8,     # 坐标轴线宽
        "figure.dpi":         150,     # 屏显分辨率
        "savefig.dpi":        300,     # 保存分辨率（出版级）
    })
    print(f"出版样式：拉丁=Times New Roman  中文衬线={cn_serif}")


def load_all_metrics():
    """读取全部指标 JSON，返回统一字典。"""
    def rd(path):  # 读单个 JSON
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    diag = rd(os.path.join(OUT_V3, "diagnostics_v3.json"))     # 诊断（v2 MAE）
    base = rd(os.path.join(OUT_V3, "baseline_metrics.json"))   # 基线
    train = rd(os.path.join(OUT_V3, "train_metrics.json"))     # 主模型
    evalm = rd(os.path.join(OUT_V3, "evaluate_metrics.json"))  # 多通道+外推
    v2    = rd(os.path.join(OUT_V2, "summary_metrics.json"))   # v2 全通道
    return diag, base, train, evalm, v2


# ==========================================================================
# 各面板绘制函数（合成图与单面板复用同一函数，保证一致性）
# ==========================================================================

def panel_a_tafh_mae(ax, base, train, v2):
    """面板 a：TAF_h 全方法 MAE 对照（基线 + v2 深度 + v3 经典）。"""
    methods = [  # (标签, MAE, 颜色)
        ("全局均值",      base["B1_global_mean"]["all"]["mae"],   CB["gray"]),
        ("网格插值",      base["B3_grid_interp"]["all"]["mae"],   CB["sky"]),
        ("v2 CNN",        v2["cnn"]["TAF_h_MAE"],                 CB["orange"]),
        ("v2 LSTM",       v2["lstm"]["TAF_h_MAE"],                CB["orange"]),
        ("v2 Trans.",     v2["transformer"]["TAF_h_MAE"],         CB["orange"]),
        ("v2 DeepONet",   v2["deeponet"]["TAF_h_MAE"],            CB["red"]),
        ("v3 XGBoost",    train["XGB"]["all"]["mae"],             CB["blue"]),
        ("v3 GPR",        train["GPR"]["all"]["mae"],             CB["green"]),
    ]
    x    = np.arange(len(methods))  # x 坐标
    vals = [m[1] for m in methods]  # MAE 值
    bars = ax.bar(x, vals, color=[m[2] for m in methods], edgecolor="white", linewidth=0.6)
    b2   = base["B2_oracle_per_geom"]["all"]["mae"]  # 理论下限
    ax.axhline(b2, color=CB["green"], linestyle="--", linewidth=1.2,
               label=f"傻瓜基线下限 = {b2:.4f}")  # 下限参考线
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], rotation=28, ha="right")
    ax.set_ylabel("TAF_h 平均绝对误差 MAE")
    ax.set_title("(a) TAF_h 全方法误差对照")
    ax.legend(loc="upper right", frameon=False)
    for bar, v in zip(bars, vals):  # 柱顶标注
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0008, f"{v:.4f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(vals) * 1.18)


def panel_b_channel_r2(ax, evalm, v2):
    """面板 b：四通道 R2 对照（v2 DeepONet vs v3 GPR）。"""
    chans   = ["TAF_h", "PGA_h", "PGA_v", "TAF_v"]  # 通道
    chan_lb = ["TAF_h", "PGA_h", "PGA_v", "TAF_v*"]  # 显示标签（*=病态）
    v2_r2 = [v2["deeponet"]["TAF_h_R2"], v2["deeponet"]["PGA_h_R2"],
             v2["deeponet"]["PGA_v_R2"], v2["deeponet"]["TAF_v_R2"]]  # v2 最强
    v3_r2 = [evalm["TAF_h"]["all"]["r2"], evalm["PGA_h"]["all"]["r2"],
             evalm["PGA_v"]["all"]["r2"], evalm["TAF_v_reliable_only"]["all"]["r2"]]  # v3 GPR
    x = np.arange(len(chans)); w = 0.36  # 坐标与柱宽
    b1 = ax.bar(x - w/2, v2_r2, w, color=CB["red"],   label="v2 DeepONet（最强深度）", edgecolor="white")
    b2 = ax.bar(x + w/2, v3_r2, w, color=CB["green"], label="v3 GPR（经典 ML）",      edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(chan_lb)
    ax.set_ylabel("决定系数 R²")
    ax.set_title("(b) 四通道 R² 对照")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left", frameon=False)
    for bars in (b1, b2):  # 柱顶标注
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)


def panel_c_channel_mae(ax, evalm, v2):
    """面板 c：三良态通道 MAE 对照（TAF_v 病态另文，此处对数轴含全四通道）。"""
    chans   = ["TAF_h", "PGA_h", "PGA_v", "TAF_v"]  # 通道
    chan_lb = ["TAF_h", "PGA_h", "PGA_v", "TAF_v*"]  # 标签
    v2_mae = [v2["deeponet"]["TAF_h_MAE"], v2["deeponet"]["PGA_h_MAE"],
              v2["deeponet"]["PGA_v_MAE"], v2["deeponet"]["TAF_v_MAE"]]  # v2 最强
    v3_mae = [evalm["TAF_h"]["all"]["mae"], evalm["PGA_h"]["all"]["mae"],
              evalm["PGA_v"]["all"]["mae"], evalm["TAF_v_reliable_only"]["all"]["mae"]]  # v3 GPR
    x = np.arange(len(chans)); w = 0.36  # 坐标与柱宽
    b1 = ax.bar(x - w/2, v2_mae, w, color=CB["red"],   label="v2 DeepONet", edgecolor="white")
    b2 = ax.bar(x + w/2, v3_mae, w, color=CB["green"], label="v3 GPR",      edgecolor="white")
    ax.set_yscale("log")  # 对数轴（TAF_v MAE 比其它大一个量级）
    ax.set_xticks(x); ax.set_xticklabels(chan_lb)
    ax.set_ylabel("MAE（对数轴）")
    ax.set_title("(c) 四通道 MAE 对照")
    ax.legend(loc="upper left", frameon=False)
    for bars in (b1, b2):  # 柱顶标注
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)


def panel_d_extrapolation(ax, train, evalm):
    """面板 d：v3 适用域 R2（内插强 vs 外插弱，诚实示警）。"""
    loh = evalm["leave_one_height"]  # 留一高度
    loa = evalm["leave_one_angle"]   # 留一角度
    # 区分内插与外插（h 两端 10/400 为外插，angle 端点 0/30 略弱）
    h_keys = sorted(float(k) for k in loh.keys())  # 高度取值
    h_inner = [loh[_key(loh, k)]["all"]["r2"] for k in h_keys[1:-1]]  # 内插高度（去掉两端外插）
    items = [  # (标签, R2, 颜色)
        ("标准\nGroupKFold",  evalm["TAF_h"]["all"]["r2"],                       CB["green"]),
        ("留一角度\n(均值)",   float(np.mean([loa[k]["all"]["r2"] for k in loa])), CB["blue"]),
        ("留一高度\n内插均值", float(np.mean(h_inner)),                            CB["sky"]),
        ("留一高度\n外插(h=400)", loh[_key(loh, 400)]["all"]["r2"],               CB["orange"]),
        ("留一高度\n外插(h=10)",  loh[_key(loh, 10)]["all"]["r2"],                CB["red"]),
        ("留一波\n(仅3条)",    train["GPR_leave_one_wave"]["all"]["r2"],          CB["purple"]),
    ]
    x    = np.arange(len(items))  # 坐标
    vals = [m[1] for m in items]  # R2
    bars = ax.bar(x, vals, color=[m[2] for m in items], edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)  # R2=0 参考线（等于猜均值）
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in items], fontsize=8.5)
    ax.set_ylabel("TAF_h R²")
    ax.set_title("(d) 泛化与外推适用域（内插强 / 外插弱）")
    for bar, v in zip(bars, vals):  # 柱顶/柱底标注
        va  = "bottom" if v >= 0 else "top"
        off = 0.03 if v >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width()/2, v + off, f"{v:.2f}",
                ha="center", va=va, fontsize=8)
    lo = min(vals)  # 最小值（外插可能为负）
    ax.set_ylim(min(-0.3, lo * 1.15), 1.05)


def _key(d, val):
    """在字典里找匹配数值 val 的键（兼容 '10'/'10.0' 两种写法）。"""
    for k in d:  # 遍历键
        if abs(float(k) - val) < 1e-6:  # 数值匹配
            return k
    raise KeyError(val)  # 未找到则报错


# ==========================================================================
# 合成图 + 各面板输出（遵循「同名文件夹 + panels/」约定）
# ==========================================================================

def save_composite_with_panels(fig_name, panel_funcs, panel_names, data):
    """绘制合成图并保存，同时把每个面板单独导出到 panels/ 子目录。

    Parameters
    ----------
    fig_name    : str，图名（同名文件夹名）
    panel_funcs : list of callable(ax, *data)，各面板绘制函数
    panel_names : list of str，各面板语义名（panels/ 文件名）
    data        : tuple，传给每个 panel_func 的数据（按需切片）
    """
    fig_dir   = os.path.join(PUB_DIR, fig_name)  # 同名文件夹
    panel_dir = os.path.join(fig_dir, "panels")  # 子面板目录
    os.makedirs(panel_dir, exist_ok=True)  # 创建目录

    # ── 1. 合成图（2×2）─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))  # 四面板布局
    fig.suptitle("v2 深度学习 vs v3 经典机器学习：地形放大代理模型全面对照",
                 fontsize=14, fontweight="bold", y=0.995)
    for func, ax in zip(panel_funcs, axes.ravel()):  # 逐面板绘制
        func(ax)  # 调用（data 已通过闭包绑定）
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # 留出总标题空间
    for ext in ("png", "pdf"):  # 矢量+位图双格式
        fig.savefig(os.path.join(fig_dir, f"{fig_name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"合成图已保存：{fig_dir}/{fig_name}.{{png,pdf}}")

    # ── 2. 各面板单独导出 ───────────────────────────────────────
    for func, pname in zip(panel_funcs, panel_names):  # 逐面板
        fig_p, ax_p = plt.subplots(figsize=(6.5, 5))  # 单面板画布
        func(ax_p)  # 复用同一绘制函数
        fig_p.tight_layout()
        for ext in ("png", "pdf"):
            fig_p.savefig(os.path.join(panel_dir, f"{pname}.{ext}"), bbox_inches="tight")
        plt.close(fig_p)
    print(f"各面板已保存：{panel_dir}/ （{len(panel_names)} 个）")


# ==========================================================================
# 论文用最终对照表（Markdown）
# ==========================================================================

def write_final_table(base, train, evalm, v2, outpath):
    """生成论文用最终对照表 Markdown 文件。"""
    lines = []  # 行缓存
    lines.append("# v3 最终结果汇总表（自动生成，勿手改）\n")
    lines.append(f"> 生成脚本：`ML/figures_v3.py`\n")

    # 表 1：TAF_h 全方法对照
    lines.append("\n## 表 1　TAF_h 全方法对照（5 折 GroupKFold）\n")
    lines.append("| 方法 | 全区 MAE | 峰值区 MAE | 远场区 MAE | 全区 R2 | 备注 |")
    lines.append("|------|---------|-----------|-----------|---------|------|")
    b1, b2, b3 = base["B1_global_mean"], base["B2_oracle_per_geom"], base["B3_grid_interp"]
    rows = [
        ("B1 全局均值",     b1, "R2 零点"),
        ("B3 规则网格插值", b3, "不用 ML 的真正下限"),
        ("B2 oracle 均值",  b2, "理论下限（可见标签）"),
        ("v3 XGBoost",      train["XGB"], "秒级训练 · CPU"),
        ("v3 GPR",          train["GPR"], "**最优** · 含不确定性"),
    ]
    for name, r, note in rows:
        a, p, f = r["all"], r["peak"], r["far"]
        lines.append(f"| {name} | {a['mae']:.4f} | {p['mae']:.4f} | {f['mae']:.4f} | {a['r2']:.4f} | {note} |")
    # v2 深度模型行
    for m, lb in [("cnn", "v2 CNN"), ("lstm", "v2 LSTM"),
                  ("transformer", "v2 Transformer"), ("deeponet", "v2 DeepONet")]:
        lines.append(f"| {lb} | {v2[m]['TAF_h_MAE']:.4f} | — | — | {v2[m]['TAF_h_R2']:.4f} | 小时级训练 · GPU |")

    # 表 2：四通道 v2 vs v3
    lines.append("\n## 表 2　四通道对照（v2 DeepONet vs v3 GPR）\n")
    lines.append("| 通道 | v2 R2 | v3 R2 | v2 MAE | v3 MAE | 说明 |")
    lines.append("|------|-------|-------|--------|--------|------|")
    ch_map = [("TAF_h", "TAF_h", "水平地形放大"),
              ("PGA_h", "PGA_h", "水平峰值加速度"),
              ("PGA_v", "PGA_v", "竖向峰值加速度"),
              ("TAF_v", "TAF_v_reliable_only", "竖向放大（病态，仅可靠点）")]
    for v2k, v3k, desc in ch_map:
        v2r2 = v2["deeponet"][f"{v2k}_R2"]; v2mae = v2["deeponet"][f"{v2k}_MAE"]
        v3r2 = evalm[v3k]["all"]["r2"];     v3mae = evalm[v3k]["all"]["mae"]
        lines.append(f"| {v2k} | {v2r2:.3f} | {v3r2:.3f} | {v2mae:.4f} | {v3mae:.4f} | {desc} |")

    # 表 3：外推适用域
    lines.append("\n## 表 3　外推适用域（TAF_h，POD+GPR）\n")
    lines.append("| 测试 | R2 | MAE | 性质 |")
    lines.append("|------|-----|-----|------|")
    loh, loa = evalm["leave_one_height"], evalm["leave_one_angle"]
    lines.append(f"| 标准 GroupKFold | {evalm['TAF_h']['all']['r2']:.3f} | {evalm['TAF_h']['all']['mae']:.4f} | 未见过该几何（内插）|")
    loa_r2 = np.mean([loa[k]['all']['r2'] for k in loa])
    loa_mae = np.mean([loa[k]['all']['mae'] for k in loa])
    lines.append(f"| 留一角度（均值）| {loa_r2:.3f} | {loa_mae:.4f} | 角度内插，稳健 |")
    for v in sorted(loh.keys(), key=float):
        r = loh[v]["all"]
        prop = "外插边界，崩溃" if float(v) in (10.0, 400.0) else "高度内插"
        lines.append(f"| 留一高度 h={float(v):g}m | {r['r2']:.3f} | {r['mae']:.4f} | {prop} |")
    low = train["GPR_leave_one_wave"]["all"]
    lines.append(f"| 留一波（仅 3 条）| {low['r2']:.3f} | {low['mae']:.4f} | 波外推上限测试 |")

    lines.append("\n## 核心结论\n")
    lines.append("1. **经典 ML 完胜深度学习**：v3 GPR（TAF_h MAE "
                 f"{train['GPR']['all']['mae']:.4f}）显著优于 v2 最强 DeepONet "
                 f"（{v2['deeponet']['TAF_h_MAE']:.4f}），训练成本低几个数量级。")
    lines.append(f"2. **波特征有效**：GPR 峰值区 MAE {train['GPR']['peak']['mae']:.4f} "
                 f"< 傻瓜基线 {b2['peak']['mae']:.4f}，证明反应谱等波特征捕捉了三波差异。")
    lines.append("3. **诚实适用域**：参数范围内插值可靠（R2≈0.95），但坡高/地震波"
                 "边界外推不可信（h=10/400 R2 崩溃，留一波 R2≈0.27）。")
    lines.append("4. **TAF_v 物理病态**：斜入射 SV 波下平地竖向 PGA 近零，"
                 f"原始 TAF_v max={evalm['TAF_v_meta']['tafv_raw_max']:.2e}，"
                 f"即便排除 {(1-evalm['TAF_v_meta']['reliable_frac'])*100:.1f}% 近零点后 MAE 仍高，单独汇报。")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"最终对照表已保存：{outpath}")


# ==========================================================================
# 主流程
# ==========================================================================

def main():
    print("=" * 64)
    print("ML v3 — Stage 4 出版级对照图表")
    print("=" * 64)
    setup_cn_journal_style()  # 配置中文核心期刊样式
    diag, base, train, evalm, v2 = load_all_metrics()  # 读全部指标

    # 用闭包绑定数据，使各 panel 函数签名统一为 func(ax)
    panel_funcs = [
        lambda ax: panel_a_tafh_mae(ax, base, train, v2),
        lambda ax: panel_b_channel_r2(ax, evalm, v2),
        lambda ax: panel_c_channel_mae(ax, evalm, v2),
        lambda ax: panel_d_extrapolation(ax, train, evalm),
    ]
    panel_names = ["a_TAFh_MAE_allmethods", "b_channel_R2",
                   "c_channel_MAE", "d_extrapolation_domain"]

    save_composite_with_panels("fig_v2_vs_v3", panel_funcs, panel_names,
                               (diag, base, train, evalm, v2))

    write_final_table(base, train, evalm, v2,
                      os.path.join(OUT_V3, "table_final.md"))

    print("\n[完成] Stage 4 出版级图表生成完成")


if __name__ == "__main__":
    main()
