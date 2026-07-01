# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""生成学术方法文档插图（射线法 vs 频域全局矩阵法对比，全部为真实计算结果）。

在本文件夹（Modeling/Multi/docs/）下运行：
    python make_doc_figures.py
依赖：numpy、matplotlib（普通 Python 即可，不需要 Abaqus）。
输出 5 张图到 ./figs/，供 freefield_fd_method_academic_v1.md / .docx 引用：
  fig1_schematic.png   方法思想示意（射线叠加 vs 频域柱解）
  fig2_halfspace.png   半空间退化验证：fd vs 解析射线解（θ=15°）
  fig3_transfer.png    三层柱传递函数：fd(=精确) vs SH Haskell vs 射线引擎
  fig4_softnode.png    软表层侧边界节点自由场时程：射线 vs fd
  fig5_damping.png     瑞利阻尼 ξ(ω) 与恒定 Q + 阻尼对柱传递函数的影响
"""
import os, sys, math, types, importlib  # 导入标准库

# ---- 桩模块屏蔽 Abaqus 依赖（与 test 脚本同法） ----
for _n, _attr in (('abaqus', 'mdb'), ('abaqusConstants', None),
                  ('regionToolset', 'Region'), ('caeModules', None), ('mesh', None)):
    _m = types.ModuleType(_n)  # 创建桩模块
    if _attr == 'mdb':  # abaqus 需要 mdb 占位
        _m.mdb = types.SimpleNamespace()  # 设置 mdb 占位
    elif _attr == 'Region':  # regionToolset 需要 Region 占位
        _m.Region = object  # 设置 Region 占位
    sys.modules[_n] = _m  # 注册桩模块

_HERE = os.path.dirname(os.path.abspath(__file__))  # 本脚本所在目录（docs）
_OVERRIDE = os.environ.get('V6_PATH')  # 可选：v6 脚本目录覆盖（特殊环境用）
if _OVERRIDE:  # 提供覆盖目录时优先
    sys.path.insert(0, _OVERRIDE)  # 插入搜索路径
sys.path.append(os.path.dirname(_HERE))  # 加入上级目录（Multi，含 v6 脚本）

import numpy as np  # 导入数值库
import matplotlib  # 导入绘图库
matplotlib.use('Agg')  # 非交互后端（仅存图）
import matplotlib.pyplot as plt  # 导入绘图接口
from matplotlib.patches import Rectangle, FancyArrowPatch  # 导入图形元素
ml = importlib.import_module('VAB_oblique_TAF_multilayer_v6')  # 导入 v6 多层脚本

OUT = os.path.join(_HERE, 'figs')  # 图件输出目录
os.makedirs(OUT, exist_ok=True)  # 确保输出目录存在
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'figure.dpi': 200})  # 统一绘图风格

GCFG = {'H_minus_h': 200.0, 'i': 45.0, 'h_over_H': 0.5, 'total_L': 1800.0,  # 论文几何参数
        'left_flat': 1000.0, 'bedrock_thickness': 200.0}  # 上平台长度与基岩厚度
NU, RHO = 0.3, 2500.0  # 泊松比与密度（论文恒定值）
BEDROCK = {'elastic_modulus': 26e9, 'poisson_ratio': NU, 'density': RHO}  # 基岩（Vs=2000）
SURF = {'name': 'surface', 'velocity_ratio': 5.0, 'poisson_ratio': NU, 'density': RHO, 'thickness': 150.0}  # 软表层
OVER = {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': NU, 'density': RHO}  # 覆盖层


def ricker(fc=4.0, dt=1e-3, T=2.0, t0=0.3):  # 生成 Ricker 加速度记录
    """返回 (t, acc)：中心频率 fc 的 Ricker 子波加速度时程。"""
    t = np.arange(0.0, T + dt * 0.5, dt)  # 时间轴
    a = (1 - 2 * (math.pi * fc * (t - t0)) ** 2) * np.exp(-(math.pi * fc * (t - t0)) ** 2)  # Ricker 公式
    return t, a  # 返回时间轴与加速度


def make_ctx(layers, angle=15.0, fc=4.0, damping_on=False, include_damping=True):  # 构造 FreeFieldCtx
    """按层配置构造 v6 FreeFieldCtx（含 v6 新增 acc/damp_terms/ffcfg 字段）。"""
    mcfg = {'angle': angle, 'bedrock': dict(BEDROCK), 'layers': layers}  # 材料配置
    site, fixed = ml.build_site(mcfg, GCFG)  # 构建场地
    geom = ml.make_geometry(total_L=GCFG['total_L'], H_minus_h=GCFG['H_minus_h'], i=GCFG['i'],  # 几何
                            h_over_H=GCFG['h_over_H'], left_flat=GCFG['left_flat'],
                            bedrock_thickness=GCFG['bedrock_thickness'], fixed_thicknesses=fixed)
    strat = ml._build_stratigraphy(site, geom, ymin=0.0)  # 分层带
    a1 = math.radians(angle if abs(angle) > 1e-12 else 1e-10)  # 入射角弧度
    mb = ml._compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 基岩参数
    p = math.sin(a1) / mb['cs']  # 水平慢度
    b1 = ml._safe_arcsin(mb['cp'] * math.sin(a1) / mb['cs'])  # 基岩 P 角
    t, acc = ricker(fc=fc)  # 输入 Ricker
    dt = t[1] - t[0]  # 步长
    vel, _ = ml._integrate_acc_to_velocity(acc.copy(), dt, t)  # 积分速度（射线法路径用）
    dis = np.zeros_like(vel); dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 积分位移
    damping = None  # 默认无阻尼
    if damping_on:  # 启用阻尼时构造解析后配置
        damping = {'enable': True, 'method': 'rayleigh', 'qs_factor': 0.05,
                   'q_bedrock': 999.0, 'fc': fc, 'f1_factor': 0.5, 'f2_factor': 2.5}
    damp_terms = ml._band_damping_terms(strat, damping)  # 各带瑞利系数表
    ffcfg = dict(ml.freefield_cfg); ffcfg['include_damping'] = include_damping  # 引擎配置
    ml._REFL_COEFF_CACHE.clear(); ml._FD_SOLVER_CACHE.clear()  # 清空缓存
    ctx = ml.FreeFieldCtx(site=site, geom=geom, strat=strat, ymax_l=geom.H_upper, ymax_r=geom.H_lower,
                          ymin=0.0, alpha=a1, beta_p=b1, p_horiz=p, GG=mb['GG'], lam=mb['lam'],
                          cs=mb['cs'], cp=mb['cp'], VEL=np.column_stack((t, vel)),
                          DIS=np.column_stack((t, dis)), dt=dt, time_arr=t, max_reflect_order=3,
                          acc=acc, damp_terms=damp_terms, ffcfg=ffcfg)  # 组装上下文
    return ctx, geom  # 返回上下文与几何


def sh_transfer(freqs, layers_top_down, halfspace):  # 独立 SH Haskell 递推（图3 对照）
    """layers_top_down: [(rho, cs, h), ...] 自地表向下；halfspace: (rho, cs)；返回 |传递函数|。"""
    out = np.zeros(len(freqs))  # 初始化输出
    for i, f in enumerate(freqs):  # 逐频递推
        w = 2.0 * math.pi * f  # 圆频率
        A, B = 1.0 + 0j, 1.0 + 0j  # 地表层上/下行幅值（自由面 A=B）
        stack = list(layers_top_down) + [halfspace + (None,)]  # 层栈（末项半空间）
        for m in range(len(layers_top_down)):  # 自上向下
            rho1, c1, h1 = stack[m]  # 当前层
            rho2, c2 = stack[m + 1][0], stack[m + 1][1]  # 下一层
            alp = (rho1 * c1) / (rho2 * c2)  # 阻抗比
            k1 = w / c1  # 波数
            e_p, e_m = np.exp(1j * k1 * h1), np.exp(-1j * k1 * h1)  # 相位因子
            A, B = (0.5 * A * (1 + alp) * e_p + 0.5 * B * (1 - alp) * e_m,  # 递推 A
                    0.5 * A * (1 - alp) * e_p + 0.5 * B * (1 + alp) * e_m)  # 递推 B
        out[i] = abs(2.0 / A)  # 地表/入射幅值比
    return out  # 返回传递函数模


def fig1():  # 图1：方法思想示意
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))  # 双面板
    for ax, title in zip(axes, ['(a) Ray superposition (old, v3-v5)',
                                '(b) Frequency-domain global matrix (new, v6)']):  # 两个标题
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')  # 画布设置
        ax.set_title(title, fontsize=10)  # 设标题
        ax.add_patch(Rectangle((1, 1), 8, 3.4, fc='#d9c7a0', ec='k', lw=0.8))  # 基岩
        ax.add_patch(Rectangle((1, 4.4), 8, 3.0, fc='#c4d8e8', ec='k', lw=0.8))  # 覆盖层
        ax.add_patch(Rectangle((1, 7.4), 8, 1.8, fc='#e8d5d5', ec='k', lw=0.8))  # 表层
        ax.text(9.15, 2.6, 'Bedrock\n$V_R$', fontsize=8, va='center')  # 基岩标注
        ax.text(9.15, 5.9, 'Layer 2\n$V_{s2}$', fontsize=8, va='center')  # 覆盖层标注
        ax.text(9.15, 8.3, 'Layer 1\n$V_{s1}$', fontsize=8, va='center')  # 表层标注
        ax.plot([1, 9], [9.2, 9.2], 'k-', lw=1.2)  # 自由面线
        ax.text(5, 9.5, 'free surface', ha='center', fontsize=8)  # 自由面标注
    ax = axes[0]  # 左面板：射线叠加
    ax.add_patch(FancyArrowPatch((2.2, 1.2), (3.0, 4.4), arrowstyle='-|>', mutation_scale=10, color='b'))  # 入射波
    pts = [(3.0, 4.4), (3.6, 7.4), (4.0, 9.2), (4.6, 7.4), (5.0, 9.2), (5.6, 7.4), (6.2, 9.2)]  # 折线点列
    for i in range(len(pts) - 1):  # 逐段画多次反射
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1], arrowstyle='-|>', mutation_scale=8,
                                     color='b', alpha=max(0.25, 1 - 0.18 * i)))  # 渐淡表示衰减级数
    ax.add_patch(FancyArrowPatch((3.6, 7.4), (4.4, 4.4), arrowstyle='-|>', mutation_scale=8,
                                 color='g', linestyle='--', alpha=0.8))  # 被忽略的转换波
    ax.text(6.4, 8.1, 'truncated at\norder $N_r$', fontsize=8, color='b')  # 截断标注
    ax.text(4.5, 5.2, '$R_{sp}=T_{sp}=0$\n(no SV-P conv.)', fontsize=8, color='g')  # 转换波置零标注
    ax.text(1.4, 2.0, 'incident SV', fontsize=8, color='b', rotation=72)  # 入射标注
    ax = axes[1]  # 右面板：频域全局矩阵
    for ylo, yhi, names in ((1, 4.4, ['$S_u^{(0)}$(known)', '$P_d^{(0)}, S_d^{(0)}$']),
                            (4.4, 7.4, ['$P_u^{(1)}, S_u^{(1)}$', '$P_d^{(1)}, S_d^{(1)}$']),
                            (7.4, 9.2, ['$P_u^{(2)}, S_u^{(2)}$', '$P_d^{(2)}, S_d^{(2)}$'])):  # 各层波幅标注
        ym = (ylo + yhi) / 2  # 层中部
        ax.add_patch(FancyArrowPatch((3.0, ym - 0.55), (3.6, ym + 0.55), arrowstyle='-|>',
                                     mutation_scale=10, color='b'))  # 上行波箭头
        ax.add_patch(FancyArrowPatch((4.6, ym + 0.55), (5.2, ym - 0.55), arrowstyle='-|>',
                                     mutation_scale=10, color='r'))  # 下行波箭头
        ax.text(5.5, ym + 0.32, names[0], fontsize=8, color='b')  # 上行波标注
        ax.text(5.5, ym - 0.55, names[1], fontsize=8, color='r')  # 下行波标注
    ax.text(1.2, 9.55, r'$\sigma_{yy}=\sigma_{xy}=0$', fontsize=8)  # 自由面条件
    for Y in (4.4, 7.4):  # 两个界面
        ax.text(1.2, Y + 0.12, r'$u_x,u_y,\sigma_{yy},\sigma_{xy}$ continuous', fontsize=7.5)  # 连续条件
    ax.text(2.0, 0.45, r'solve $\mathbf{A}(\omega)\,\mathbf{x}(\omega)=\mathbf{b}(\omega)$ per frequency',
            fontsize=9)  # 方程标注
    fig.tight_layout()  # 紧凑布局
    fig.savefig(os.path.join(OUT, 'fig1_schematic.png'), bbox_inches='tight')  # 保存
    plt.close(fig)  # 关闭画布
    print('fig1 done')  # 进度提示


def fig2():  # 图2：半空间退化验证
    ctx, geom = make_ctx([], angle=15.0, include_damping=False)  # 均质半空间（弹性）
    gv = ml._make_delay_cache(ctx.VEL, ctx.dt)  # 射线法速度缓存
    gd = ml._make_delay_cache(ctx.DIS, ctx.dt)  # 射线法位移缓存
    bnd, x, y = 'b', 900.0, 0.0  # 底边界代表节点
    ymax = ml._surface_y_at(900.0, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 柱地表高度
    ray = ml._compute_freefield_at_node(bnd, x, y, ymax, ctx, gv, gd)  # 射线法（半空间=解析解）
    fd = ml._fd_freefield_at_node(bnd, x, y, ymax, ctx)  # fd 引擎
    N = len(ctx.time_arr); t = ctx.time_arr  # 比较窗口
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)  # 两子图
    for ax, key, lab in zip(axes, ['dotux', 'sigmay'],
                            ['Horizontal velocity $\\dot{u}_x$ (m/s)',
                             'Boundary traction term $\\sigma_y$ (Pa)']):  # 速度与面力
        ax.plot(t, ray[key][:N], 'k-', lw=1.4, label='Ray method (= exact half-space solution)')  # 解析解
        ax.plot(t, fd[key][:N], 'r--', lw=1.1, label='FD global matrix (v6)')  # fd 解
        err = np.max(np.abs(fd[key][:N] - ray[key][:N])) / np.max(np.abs(ray[key][:N]))  # 相对误差
        ax.text(0.985, 0.06, 'max rel. err = %.2e' % err, transform=ax.transAxes, ha='right', fontsize=8)  # 标注
        ax.set_ylabel(lab, fontsize=8.5); ax.grid(alpha=0.3)  # 轴标签与网格
    axes[0].legend(fontsize=8, loc='upper right')  # 图例
    axes[0].set_title('Half-space degenerate case, bottom-boundary node (x=900 m), '
                      r'SV $\theta_s=15°$, elastic', fontsize=9.5)  # 标题
    axes[1].set_xlabel('Time (s)')  # 横轴
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_halfspace.png'), bbox_inches='tight')  # 保存
    plt.close(fig); print('fig2 done')  # 关闭并提示


def fig3():  # 图3：三层柱传递函数对比
    ctx, geom = make_ctx([SURF, OVER], angle=0.0, include_damping=False)  # 三层弹性、垂直入射
    freqs = np.linspace(0.3, 12.0, 480); omega = 2.0 * math.pi * freqs  # 频带
    col = ml._build_column(ctx.strat, geom.H_upper, ctx.p_horiz, 0.0)  # 上平台柱
    sol = ml._fd_solve_column(col, ctx.p_horiz, omega, ctx.damp_terms, False)  # fd 求解
    tf_fd = np.abs(ml._fd_eval_column(sol, omega, ctx.p_horiz, geom.H_upper)['ux'])  # fd 传递函数
    tf_sh = sh_transfer(freqs, [(RHO, 400.0, 150.0), (RHO, 800.0, 250.0)], (RHO, 2000.0))  # SH 递推
    gv = ml._make_delay_cache(ctx.VEL, ctx.dt); gd = ml._make_delay_cache(ctx.DIS, ctx.dt)  # 射线缓存
    ray = ml._compute_freefield_at_node('l', 0.0, geom.H_upper, geom.H_upper, ctx, gv, gd)  # 射线地表响应
    nfft = 1 << 15  # FFT 长度
    U_ray = np.fft.rfft(ray['ux'], n=nfft)  # 射线响应谱
    U_in = np.fft.rfft(ctx.DIS[:, 1], n=nfft)  # 输入位移谱
    fr = np.fft.rfftfreq(nfft, ctx.dt)  # 频率轴
    band = (np.abs(U_in) > 0.02 * np.max(np.abs(U_in))) & (fr > 0.3) & (fr < 12.0)  # 有效频带
    fig, ax = plt.subplots(figsize=(7.2, 3.8))  # 单图
    ax.plot(freqs, tf_fd, 'r-', lw=1.6, label='FD global matrix (v6, exact)')  # fd 曲线
    ax.plot(freqs[::24], tf_sh[::24], 'ko', ms=4, mfc='none',
            label='Independent Thomson-Haskell (SH recursion)')  # SH 散点
    ax.plot(fr[band], np.abs(U_ray[band]) / np.abs(U_in[band]), 'b--', lw=1.3,
            label='Ray superposition engine (v5)')  # 射线曲线
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel(r'Surface transfer $|u_x^{surf}/u_0^{inc}|$')  # 轴标签
    ax.set_title('Three-layer column ($V_{s1}$=400, $V_{s2}$=800, $V_R$=2000 m/s), '
                 r'vertical incidence, elastic', fontsize=9.5)  # 标题
    ax.grid(alpha=0.3); ax.legend(fontsize=8)  # 网格与图例
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_transfer.png'), bbox_inches='tight')  # 保存
    plt.close(fig); print('fig3 done')  # 关闭并提示


def fig4():  # 图4：软表层侧边界节点时程对比
    ctx, geom = make_ctx([SURF, OVER], angle=15.0, include_damping=False)  # 三层弹性、15° 入射
    gv = ml._make_delay_cache(ctx.VEL, ctx.dt); gd = ml._make_delay_cache(ctx.DIS, ctx.dt)  # 射线缓存
    bnd, x, y, ymax = 'l', 0.0, 520.0, geom.H_upper  # 节点位于软表层内（450~600 m）
    ray = ml._compute_freefield_at_node(bnd, x, y, ymax, ctx, gv, gd)  # 射线法结果
    fd = ml._fd_freefield_at_node(bnd, x, y, ymax, ctx)  # fd 结果
    N = len(ctx.time_arr); t = ctx.time_arr  # 比较窗口
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)  # 两子图
    for ax, key, lab in zip(axes, ['ux', 'sigmax'],
                            ['Horizontal displacement $u_x$ (m)',
                             'Boundary traction term $\\sigma_x$ (Pa)']):  # 位移与面力
        ax.plot(t, fd[key][:N], 'r-', lw=1.4, label='FD global matrix (v6, exact)')  # fd 曲线
        ax.plot(t, ray[key][:N], 'b--', lw=1.2, label='Ray superposition (v5)')  # 射线曲线
        ratio = np.max(np.abs(ray[key][:N])) / np.max(np.abs(fd[key][:N]))  # 峰值比
        ax.text(0.985, 0.06, 'peak ratio (ray/FD) = %.2f' % ratio, transform=ax.transAxes,
                ha='right', fontsize=8)  # 标注峰值比
        ax.set_ylabel(lab, fontsize=8.5); ax.grid(alpha=0.3)  # 轴标签与网格
    axes[0].legend(fontsize=8, loc='upper right')  # 图例
    axes[0].set_title('Free-field at a lateral-boundary node inside the SOFT surface layer '
                      r'(y=520 m, $V_{s1}$=400 m/s), $\theta_s=15°$', fontsize=9.5)  # 标题
    axes[1].set_xlabel('Time (s)')  # 横轴
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_softnode.png'), bbox_inches='tight')  # 保存
    plt.close(fig); print('fig4 done')  # 关闭并提示


def fig5():  # 图5：阻尼一致化
    fc = 4.0; Q = 20.0; xi_t = 1.0 / (2 * Q)  # 软层目标阻尼比
    a_ray, b_ray = ml._rayleigh_coeffs(xi_t, {'method': 'rayleigh', 'f1_factor': 0.5,
                                              'f2_factor': 2.5}, fc)  # 双频拟合系数
    f = np.linspace(0.3, 15, 400); w = 2 * math.pi * f  # 频带
    xi = a_ray / (2 * w) + b_ray * w / 2  # 瑞利 ξ(ω)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))  # 双面板
    ax = axes[0]  # 面板 a
    ax.plot(f, xi * 100, 'b-', lw=1.5, label=r'Rayleigh $\xi(\omega)=\alpha/2\omega+\beta\omega/2$')  # ξ曲线
    ax.axhline(xi_t * 100, color='k', ls='--', lw=1.0, label='Constant-Q target ($Q_s$=20)')  # 目标线
    ax.axvspan(0.5 * fc, 2.5 * fc, color='orange', alpha=0.15, label='fitting band $[0.5f_c, 2.5f_c]$')  # 拟合带
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel(r'Damping ratio $\xi$ (%)')  # 轴标签
    ax.set_title('(a) Rayleigh damping vs constant Q', fontsize=9.5)  # 标题
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.set_ylim(0, 8)  # 图例/网格/范围
    ctx, geom = make_ctx([SURF, OVER], angle=0.0, damping_on=True, include_damping=True)  # 三层含阻尼
    freqs = np.linspace(0.3, 12.0, 480); omega = 2 * math.pi * freqs  # 频带
    col = ml._build_column(ctx.strat, geom.H_upper, ctx.p_horiz, 0.0)  # 上平台柱
    tf_e = np.abs(ml._fd_eval_column(ml._fd_solve_column(col, ctx.p_horiz, omega, ctx.damp_terms, False),
                                     omega, ctx.p_horiz, geom.H_upper)['ux'])  # 弹性传递函数
    tf_d = np.abs(ml._fd_eval_column(ml._fd_solve_column(col, ctx.p_horiz, omega, ctx.damp_terms, True),
                                     omega, ctx.p_horiz, geom.H_upper)['ux'])  # 含阻尼传递函数
    ax = axes[1]  # 面板 b
    ax.plot(freqs, tf_e, 'k--', lw=1.2, label='elastic free field (v5 assumption)')  # 弹性曲线
    ax.plot(freqs, tf_d, 'r-', lw=1.5, label='damping-consistent free field (v6)')  # 阻尼曲线
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel(r'$|u_x^{surf}/u_0^{inc}|$')  # 轴标签
    ax.set_title('(b) Effect on 3-layer column transfer', fontsize=9.5)  # 标题
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)  # 图例与网格
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5_damping.png'), bbox_inches='tight')  # 保存
    plt.close(fig); print('fig5 done')  # 关闭并提示


if __name__ == '__main__':  # 直接运行入口
    fig1(); fig2(); fig3(); fig4(); fig5()  # 依次生成 5 张图
    print('ALL FIGURES DONE ->', OUT)  # 完成提示
