# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""探针脚本3：验证改写为射线法后的 v7 引擎可在纯 Python 下运行且输出干净。

做法：给 Abaqus 模块打桩→导入 v7→手工组装 FreeFieldCtx→对一批地表节点调用
_compute_freefield_at_node，检查地表速度时程是否为干净 Ricker、PGV 随覆盖层厚度是否光滑。
"""

import os  # 导入操作系统接口
import sys  # 导入系统模块
import types  # 导入类型模块
import math  # 导入数学模块
import numpy as np  # 导入数值计算库
import matplotlib  # 导入绘图库
matplotlib.use('Agg')  # 无界面后端
import matplotlib.pyplot as plt  # 导入绘图接口


def _install_abaqus_stubs():  # 安装 Abaqus 桩模块
    for name in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):  # 遍历模块名
        sys.modules[name] = types.ModuleType(name)  # 注册空桩
    sys.modules['abaqus'].mdb = types.SimpleNamespace()  # 提供 mdb
    sys.modules['regionToolset'].Region = object  # 提供 Region


def _load_v7():  # 加载 v7 模块
    here = os.path.dirname(os.path.abspath(__file__))  # 当前目录
    src = os.path.normpath(os.path.join(here, '..', '..', 'Modeling', 'Multi', 'VAB_oblique_TAF_double_v4.py'))  # 射线法引擎源文件（原 v7 已重命名为 v4）
    import importlib.util  # 导入加载工具
    spec = importlib.util.spec_from_file_location('v7mod', src)  # 模块规范
    mod = importlib.util.module_from_spec(spec)  # 创建模块
    spec.loader.exec_module(mod)  # 执行加载
    return mod  # 返回模块


def main():  # 主流程
    _install_abaqus_stubs()  # 打桩
    v7 = _load_v7()  # 加载（同时验证语法/导入是否通过）
    print('导入成功，模块加载通过')  # 导入即语法检查

    # ===== 材料、角度、全局射线系数（复刻 VAB_oblique 输入段逻辑）=====
    mat_b = v7._compute_material_params(2000.0, 0.3, 2500.0)  # 基岩
    mat_o = v7._compute_material_params(1600.0, 0.3, 2500.0)  # 覆盖层
    cs1, cp1, cs2, cp2 = mat_b['cs'], mat_b['cp'], mat_o['cs'], mat_o['cp']  # 波速
    alpha1 = math.radians(15.0)  # 入射角

    i12 = v7._compute_interface_sv_coeff(alpha1, mat_b, mat_o)  # 界面 1->2
    alpha2 = i12['alpha2']  # 透射角
    beta1 = v7._safe_arcsin(cp1 * math.sin(alpha1) / cs1)  # 基岩 P 角
    beta2 = v7._safe_arcsin(cp2 * math.sin(alpha2) / cs2) if abs(math.sin(alpha2)) > 0 else 1e-10  # 覆盖层 P 角
    fsv = v7._compute_free_surface_sv_coeff(alpha2, cp2, cs2)  # 自由面 SV
    fp = v7._compute_free_surface_p_coeff(beta2, cp2, cs2)  # 自由面 P
    i21 = v7._compute_interface_sv_coeff(alpha2, mat_o, mat_b)  # 界面 2->1
    cycle_sv = fsv['A1'] * i21['Rss']  # SV 混响幅值因子
    cycle_p = fp['B2'] * i21['Rss']  # P 混响幅值因子
    oc = v7.MAX_REFLECT_ORDER  # 阶数
    scs = sum(cycle_sv ** k for k in range(oc + 1))  # SV 级数和
    scp = sum(cycle_p ** k for k in range(oc + 1))  # P 级数和
    Rss_eff = i12['Rss'] + i12['Tss'] * fsv['A1'] * i21['Tss'] * scs  # 等效反射
    Rsp_eff = i12['Rsp'] + i12['Tss'] * fsv['A2'] * i21['Tss'] * scp  # 等效转换
    print('Rss_eff=%.4f Rsp_eff=%.4f alpha2=%.4f beta2=%.4f' % (Rss_eff, Rsp_eff, alpha2, beta2))  # 打印系数

    # ===== 读 6Hz Ricker，积分得 VEL/DIS =====
    here = os.path.dirname(os.path.abspath(__file__))  # 当前目录
    wave = os.path.normpath(os.path.join(here, '..', '..', 'Wave', 'Impulse', 'ricker_wavelet_6Hz.txt'))  # 波文件
    ACC = np.loadtxt(wave)  # 读取
    t_arr, acc = ACC[:, 0], ACC[:, 1]  # 时间与加速度
    dt = t_arr[1] - t_arr[0]  # 步长
    vel, _ = v7._integrate_acc_to_velocity(acc, dt, t_arr)  # 速度（基线校正）
    dis = np.zeros_like(vel); dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 位移
    VEL = np.column_stack((t_arr, vel)); DIS = np.column_stack((t_arr, dis))  # 时程

    # ===== 构造几何（H-h=200,i=45,h/H=0.5,基岩厚200）=====
    geom = v7.make_geometry(total_L=1800.0, H_minus_h=200.0, i=45.0, h_over_H=0.5,
                            left_flat=1000.0, bedrock_thickness=200.0)  # 调用几何构造
    ymin = 0.0  # 底边
    ymax_l = geom.H_upper  # 左侧（坡顶）地表高
    ymax_r = geom.H_lower  # 右侧（坡脚）地表高

    ctx = v7.FreeFieldCtx(  # 组装上下文
        mat_bedrock=mat_b, mat_overlying=mat_o, geom=geom,
        ymax_l=ymax_l, ymax_r=ymax_r, ymin=ymin,
        alpha=alpha1, beta_p=beta1, alpha2=alpha2, beta2=beta2,
        A1=Rss_eff, A2=Rsp_eff, cycle_sv=cycle_sv, cycle_p=cycle_p,
        GG=mat_b['GG'], lam=mat_b['lam'], cs=cs1, cp=cp1, cs2=cs2, cp2=cp2,
        VEL=VEL, DIS=DIS, dt=dt, time_arr=t_arr, max_reflect_order=oc)

    get_vel = v7._make_delay_cache(VEL, dt)  # 速度延迟缓存
    get_dis = v7._make_delay_cache(DIS, dt)  # 位移延迟缓存

    # ===== 左边界一个坡顶地表节点的时程 =====
    ff = v7._compute_freefield_at_node('l', 0.0, geom.H_upper, geom.H_upper, ctx, get_vel, get_dis)  # 地表节点
    No = len(t_arr)  # 原始长度
    print('左坡顶地表节点: PGV_x=%.4f m/s, 时程总长=%d' % (np.max(np.abs(ff['dotux'][:No])), len(ff['time'])))  # PGV

    # ===== 底边节点沿 x 扫描，看 PGV 是否光滑（覆盖层厚随 x 变化）=====
    xs = np.linspace(0.0, 1800.0, 120)  # 底边采样 x
    pgv_b = []  # 存底边 PGV
    for xx in xs:  # 遍历
        ymax_col = v7._surface_y_at(xx, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 该柱地表高
        ffb = v7._compute_freefield_at_node('b', xx, ymin, ymax_col, ctx, get_vel, get_dis)  # 底边节点
        pgv_b.append(np.max(np.abs(ffb['dotux'][:No])))  # 记录 PGV
    pgv_b = np.array(pgv_b)  # 转数组
    jump = np.max(np.abs(np.diff(pgv_b))) / np.median(pgv_b)  # 相邻最大相对跳变
    print('底边 PGV_x(x) 扫描: 中位=%.4f, 相邻最大相对跳变=%.2f%%' % (np.median(pgv_b), jump * 100))  # 统计

    # ===== 出图 =====
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))  # 双子图
    axs[0].plot(ff['time'][:No], ff['dotux'][:No])  # 地表速度时程
    axs[0].set_title('Surface dotux (ray, left top)'); axs[0].set_xlabel('t (s)')  # 标注
    axs[1].plot(xs, pgv_b)  # 底边 PGV-x
    axs[1].set_title('Bottom PGV_x vs x (ray)'); axs[1].set_xlabel('x (m)')  # 标注
    fig.tight_layout()  # 布局
    out = os.path.join(here, 'probe_ray_v1.png')  # 输出路径
    fig.savefig(out, dpi=130)  # 保存
    print('图已保存:', out)  # 打印


if __name__ == '__main__':  # 入口
    main()  # 运行
