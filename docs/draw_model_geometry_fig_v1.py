# -*- coding: utf-8 -*-
"""绘制论文用"模型尺寸设计示意图"（对齐 slope_frame_ssi_full_v1.py 无量纲几何设计）。

输出：docs/论文章节_模型尺寸设计_v1_figs/模型尺寸设计示意图.{png,svg,pdf}
运行：python docs/draw_model_geometry_fig_v1.py
图中一切长度以坡高 h 为单位（典型取值 A_max=5, C_max=4, c=2, b=2, d/h=1, i=45°）。
"""
import os

import matplotlib
matplotlib.use('Agg')  # 无界面后端（脚本出图）
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Polygon

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文标注字体
plt.rcParams['axes.unicode_minus'] = False  # 修复负号方块
plt.rcParams['font.size'] = 11  # 全图基准字号

# ---------------- 几何取值（以 h=1 为单位，对应脚本默认参数） ----------------
A_MAX, C_MAX, CLEAR, BASE = 5.0, 4.0, 2.0, 3.0  # 观测窗/净空/坡脚面以下深度参数
H = 1.0  # 坡高（作图单位）
LA = (A_MAX + CLEAR) * H  # 坡顶平台长 7
LB = H  # 坡面水平投影长（i=45°）
LC = (C_MAX + CLEAR) * H  # 坡脚平台长 6
L = LA + LB + LC  # 总长 14
Y_TOE = BASE * H  # 坡脚地表高 3（坡脚面以下深度恒定）
Y_CREST = Y_TOE + H  # 坡顶地表高 4
X_CREST, X_TOE = LA, LA + LB  # 坡肩/坡脚 x

T1, T2, TN = 0.5, 0.5, 1.0  # 土层厚度示例（h 单位；每层显式给定，剩余深度归基岩）
Y_BED = Y_CREST - (T1 + T2 + TN)  # 基岩顶面高 2（=坡顶地表 − Σ土层厚）
C_BEDROCK = '#d5d0c8'  # 基岩填充色
C_SOIL = ('#f5ecd7', '#ecdcb4', '#e2cd96')  # 土层 1/2/N 填充色（自浅到深）
C_DIM = '#333333'  # 尺寸线颜色
C_WIN = '#c0392b'  # 观测窗标注色
C_VAB = '#2166ac'  # 人工边界颜色


def dim_h(ax, x1, x2, y, text, color=C_DIM, ext=None, fs=11):  # 水平尺寸线（双箭头+界址线+标注）
    ax.annotate('', xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.0, shrinkA=0, shrinkB=0))
    if ext is not None:  # 画界址延长线（从 ext 高度引到尺寸线）
        for x in (x1, x2):
            ax.plot([x, x], [ext, y], color=color, lw=0.5, ls=(0, (2, 2)))
    ax.text((x1 + x2) / 2.0, y + 0.12, text, ha='center', va='bottom', color=color, fontsize=fs)


def dim_v(ax, y1, y2, x, text, color=C_DIM, ext=None, fs=11):  # 竖直尺寸线（双箭头+界址线+标注）
    ax.annotate('', xy=(x, y1), xytext=(x, y2),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.0, shrinkA=0, shrinkB=0))
    if ext is not None:
        for y in (y1, y2):
            ax.plot([ext, x], [y, y], color=color, lw=0.5, ls=(0, (2, 2)))
    ax.text(x + 0.15, (y1 + y2) / 2.0, text, ha='left', va='center', color=color, fontsize=fs, rotation=90)


def main():  # 组装整图并导出三种格式
    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    # ---------------- 地层区域 ----------------
    surface = [(0, Y_CREST), (X_CREST, Y_CREST), (X_TOE, Y_TOE), (L, Y_TOE)]  # 地表折线（坡顶-坡肩-坡脚-右端）
    ax.add_patch(Polygon([(0, 0), (L, 0), (L, Y_BED), (0, Y_BED)], closed=True,
                         facecolor=C_BEDROCK, edgecolor='#9a938a', hatch='//', lw=0))  # 基岩带（hatch 需 edgecolor 才可见）
    y1 = Y_CREST - T1  # 土层1底界面高程（自坡顶地表向下）
    y2 = Y_CREST - T1 - T2  # 土层2底界面高程（=坡脚地表高，本示例恰好齐平）
    x1 = X_CREST + (Y_CREST - y1) / (Y_CREST - Y_TOE) * (X_TOE - X_CREST)  # 界面 y1 与坡面交点 x
    x2 = X_CREST + (Y_CREST - y2) / (Y_CREST - Y_TOE) * (X_TOE - X_CREST)  # 界面 y2 与坡面交点 x
    ax.add_patch(Polygon([(0, y1), (x1, y1), (X_CREST, Y_CREST), (0, Y_CREST)], closed=True,
                         facecolor=C_SOIL[0], edgecolor='none', lw=0))  # 土层1（固定厚 t1，坡面出露）
    ax.add_patch(Polygon([(0, y2), (x2, y2), (x1, y1), (0, y1)], closed=True,
                         facecolor=C_SOIL[1], edgecolor='none', lw=0))  # 土层2（固定厚 t2，坡面出露）
    ax.add_patch(Polygon([(0, Y_BED), (L, Y_BED), (L, Y_TOE), (X_TOE, Y_TOE),
                          (x2, y2), (0, y2)], closed=True,
                         facecolor=C_SOIL[2], edgecolor='none', lw=0))  # 土层N（最底土层，厚度由几何闭合）
    ax.plot([0, x1], [y1, y1], color='k', lw=0.6)  # 土层1底界面线
    ax.plot([0, x2], [y2, y2], color='k', lw=0.6)  # 土层2底界面线
    ax.plot([0, L], [Y_BED, Y_BED], color='k', lw=0.8)  # 基岩界面线
    ax.plot(*zip(*surface), color='k', lw=1.6)  # 地表线
    ax.plot([0, 0], [0, Y_CREST], color='k', lw=0.8)  # 左边界
    ax.plot([L, L], [0, Y_TOE], color='k', lw=0.8)  # 右边界
    ax.plot([0, L], [0, 0], color='k', lw=0.8)  # 底边界
    ax.text(3.0, 1.0, '基岩（剩余深度，净空 $\\geq 2h$）', fontsize=12)
    ax.text(2.2, (Y_CREST + y1) / 2.0 - 0.03, '土层 1（厚 $t_1$）', fontsize=10.5, va='center')
    ax.text(2.2, (y1 + y2) / 2.0 - 0.03, '土层 2（厚 $t_2$）', fontsize=10.5, va='center')
    ax.text(2.2, (y2 + Y_BED) / 2.0 + 0.12, '土层 $N$（厚 $t_N$）', fontsize=10.5, va='center')
    ax.text(2.2, (y2 + Y_BED) / 2.0 - 0.22, '（土层数不定、逐层给厚度，$N=0$ 时全坡为基岩）', fontsize=9.5, va='center', color='#666666')

    # ---------------- 坡肩/坡脚 标记 ----------------
    ax.plot([X_CREST], [Y_CREST], 'ko', ms=4)
    ax.plot([X_TOE], [Y_TOE], 'ko', ms=4)
    ax.annotate('坡肩', xy=(X_CREST, Y_CREST), xytext=(X_CREST - 0.55, Y_CREST + 0.28), fontsize=10)
    ax.annotate('坡脚', xy=(X_TOE, Y_TOE), xytext=(X_TOE + 0.18, Y_TOE - 0.42), fontsize=10)

    # ---------------- 坡角（坡面与水平参考线夹角） ----------------
    ax.plot([X_TOE - 1.0, X_TOE], [Y_TOE, Y_TOE], color='k', lw=0.7, ls=(0, (4, 3)))  # 水平参考线
    ax.add_patch(Arc((X_TOE, Y_TOE), 1.1, 1.1, angle=0.0, theta1=135.0, theta2=180.0, color='k', lw=1.0))
    ax.text(X_TOE - 0.95, Y_TOE + 0.16, '$i$', fontsize=13)

    # ---------------- 观测窗（地表加粗红线） ----------------
    ax.plot([CLEAR, X_CREST], [Y_CREST, Y_CREST], color=C_WIN, lw=4, solid_capstyle='butt', alpha=0.85)
    ax.plot([X_TOE, X_TOE + C_MAX], [Y_TOE, Y_TOE], color=C_WIN, lw=4, solid_capstyle='butt', alpha=0.85)

    # ---------------- 顶部尺寸链（第一层：分段） ----------------
    y_dim1 = Y_CREST + 0.75
    dim_h(ax, 0, CLEAR, y_dim1, '$c\\,h$\n净空', ext=Y_CREST)
    dim_h(ax, CLEAR, X_CREST, y_dim1, '$A_{\\max}\\,h$（坡顶观测窗）', color=C_WIN, ext=Y_CREST)
    dim_h(ax, X_CREST, X_TOE, y_dim1, '$h/\\tan i$', ext=Y_TOE)
    dim_h(ax, X_TOE, X_TOE + C_MAX, y_dim1, '$C_{\\max}\\,h$（坡脚观测窗）', color=C_WIN, ext=Y_TOE)
    dim_h(ax, X_TOE + C_MAX, L, y_dim1, '$c\\,h$\n净空', ext=Y_TOE)

    # ---------------- 顶部尺寸链（第二层：总长） ----------------
    y_dim2 = Y_CREST + 1.75
    dim_h(ax, 0, L, y_dim2, '模型总长 $L=(A_{\\max}+C_{\\max}+2c)\\,h + h/\\tan i$（随 $h$、$i$ 浮动）', fs=12)
    for x in (0, L):
        ax.plot([x, x], [y_dim1 + 0.45, y_dim2], color=C_DIM, lw=0.5, ls=(0, (2, 2)))

    # ---------------- 右侧竖向尺寸链 ----------------
    x_dim = L + 0.55
    dim_v(ax, 0, Y_TOE, x_dim, '$s\\,h$（坡脚面以下，恒定）', ext=L)
    dim_v(ax, Y_TOE, Y_CREST, x_dim, '$h$（坡高）', ext=X_TOE)

    # ---------------- 左侧土层总厚 ----------------
    dim_v(ax, Y_BED, Y_CREST, -0.55, '$\\Sigma t_i$（土层总厚）', ext=0)

    # ---------------- 人工边界（左右底三侧） ----------------
    vab_kw = dict(color=C_VAB, lw=3.2, ls=(0, (4, 2)), alpha=0.9, solid_capstyle='butt')
    ax.plot([0, 0], [0, Y_CREST], **vab_kw)
    ax.plot([L, L], [0, Y_TOE], **vab_kw)
    ax.plot([0, L], [0, 0], **vab_kw)
    ax.text(L * 0.72, -0.42, '粘弹性人工边界＋分层自由场输入（左、右、底三侧）',
            color=C_VAB, fontsize=11, ha='center')

    # ---------------- 入射波箭头 ----------------
    ax.add_patch(FancyArrowPatch((L * 0.42 - 0.5, -1.25), (L * 0.42 + 0.1, -0.15),
                                 arrowstyle='-|>', mutation_scale=18, color='k', lw=1.4))
    ax.text(L * 0.42 + 0.22, -0.95, 'SV 波入射（入射角 $\\theta$）', fontsize=11)

    # ---------------- 画布 ----------------
    ax.set_xlim(-1.35, L + 1.45)
    ax.set_ylim(-1.6, Y_CREST + 2.35)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.3)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '论文章节_模型尺寸设计_v1_figs')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(os.path.join(out_dir, '模型尺寸设计示意图.%s' % ext), dpi=300, bbox_inches='tight')
    print('已输出: %s (png/svg/pdf)' % out_dir)


if __name__ == '__main__':
    main()
