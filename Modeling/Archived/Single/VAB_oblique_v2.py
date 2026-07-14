# -*- coding: utf-8 -*-
"""
VAB_oblique_v10 - 二维土体地震动斜向输入粘弹性人工边界与等效节点力施加
反编译自 VAB_oblique_v10.pyc (Python 2.7, Abaqus)
原始编译路径: c:/Users/Steve Yang/abaqus_plugins/VAB_oblique_v10/VAB_oblique_v10.py
编译时间: 2025-08-02 05:24:08
"""
from abaqus import *
from abaqusConstants import *
from abaqus import mdb
from abaqusConstants import ON
from regionToolset import Region
import numpy as np
import math


def VAB_oblique(angle, cs, vv, density):
    """
    主函数：为二维模型施加粘弹性人工边界（弹簧-阻尼器）和地震动斜向输入等效节点力

    参数:
        angle  (float): SV波入射角度（度），0为垂直入射
        cs     (float): 剪切波速 (m/s)
        vv     (float): 泊松比
        density(float): 密度 (kg/m³)
    """
    # ============ 基本参数 ============
    VELtxt = 'VEL.txt'           # 速度时程文件（两列：时间, 速度）
    step_name = 'step-earthquake'  # 地震分析步名称

    # ============ 获取装配体和Part ============
    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(adaptiveMeshConstraints=OFF)

    part = mdb.models['Model-1'].parts['Part-1']
    nodes = part.nodes

    # ============ 提取边界节点 ============
    x_list = [node.coordinates[0] for node in nodes]
    y_list = [node.coordinates[1] for node in nodes]
    xmin = min(x_list)
    xmax = max(x_list)
    ymin = min(y_list)
    tol = 1e-6

    # 左边界、右边界、底边界节点
    l_nodes_list = [node for node in nodes if abs(node.coordinates[0] - xmin) < tol]
    r_nodes_list = [node for node in nodes if abs(node.coordinates[0] - xmax) < tol]
    b_nodes_list = [node for node in nodes if abs(node.coordinates[1] - ymin) < tol]

    l_nodes = part.nodes.sequenceFromLabels([node.label for node in l_nodes_list])
    r_nodes = part.nodes.sequenceFromLabels([node.label for node in r_nodes_list])
    b_nodes = part.nodes.sequenceFromLabels([node.label for node in b_nodes_list])

    part.Set(name='l_nodes', nodes=l_nodes)
    part.Set(name='r_nodes', nodes=r_nodes)
    part.Set(name='b_nodes', nodes=b_nodes)

    # ============ 材料参数计算 ============
    GG = density * cs ** 2                    # 剪切模量
    EE = 2 * GG * (1 + vv)                   # 弹性模量
    lam = 2 * GG * vv / (1 - 2 * vv)         # 拉梅常数 λ
    cp = math.sqrt((lam + 2 * GG) / density)  # 纵波波速

    # ============ 获取模型尺寸 ============
    part = mdb.models['Model-1'].parts['Part-1']
    l_nodes = part.sets['l_nodes'].nodes
    l_ymax_node = max(l_nodes, key=lambda node: node.coordinates[1])
    xmin = l_ymax_node.coordinates[0]
    ymax_l = l_ymax_node.coordinates[1]

    b_nodes = part.sets['b_nodes'].nodes
    ymin = b_nodes[0].coordinates[1]

    r_nodes = part.sets['r_nodes'].nodes
    r_ymax_node = max(r_nodes, key=lambda node: node.coordinates[1])
    xmax = r_ymax_node.coordinates[0]
    ymax_r = r_ymax_node.coordinates[1]

    ymax = max(ymax_l, ymax_r)

    # ============ 计算节点影响长度 ============
    def get_node_influence(part, set_name, sort_axis='y', ascending=False):
        """
        获取边界节点的影响长度（半距离），返回 [n, 4] 数组：节点号、x、y、影响长度
        :param part: Abaqus Part对象
        :param set_name: 节点集名称，如 'l_nodes'
        :param sort_axis: 排序依据，'x' 或 'y'
        :param ascending: True为升序，False为降序
        :return: node_data: n行4列数组
        """
        nodes = part.sets[set_name].nodes
        node_data = []
        for node in nodes:
            node_data.append([node.label, node.coordinates[0], node.coordinates[1]])

        node_data = np.array(node_data)
        axis = 1 if sort_axis == 'x' else 2
        node_data = node_data[node_data[:, axis].argsort()]
        if not ascending:
            node_data = node_data[::-1]

        n = node_data.shape[0]
        influence = np.zeros(n)
        for i in range(n):
            if i == 0:
                influence[i] = abs(node_data[i, axis] - node_data[i + 1, axis]) / 2
            elif i == n - 1:
                influence[i] = abs(node_data[i, axis] - node_data[i - 1, axis]) / 2
            else:
                influence[i] = abs(node_data[i - 1, axis] - node_data[i + 1, axis]) / 2

        node_data = np.hstack((node_data, influence.reshape(-1, 1)))
        return node_data

    part = mdb.models['Model-1'].parts['Part-1']
    node_data_l = get_node_influence(part, 'l_nodes', sort_axis='y', ascending=False)
    node_data_r = get_node_influence(part, 'r_nodes', sort_axis='y', ascending=False)
    node_data_b = get_node_influence(part, 'b_nodes', sort_axis='x', ascending=True)

    # ============ 粘弹性人工边界参数（刘晶波公式） ============
    kn = GG / 2 / ymax       # 法向弹簧刚度系数
    cn = density * cp         # 法向阻尼系数
    kt = GG / 4 / ymax       # 切向弹簧刚度系数
    ct = density * cs         # 切向阻尼系数

    def add_spring_damper(node_data):
        """将弹簧刚度和阻尼系数乘以影响长度，追加到 node_data"""
        influence = node_data[:, 3]
        kns = kn * influence
        cns = cn * influence
        kts = kt * influence
        cts = ct * influence
        return np.hstack((node_data,
                           kns.reshape(-1, 1),
                           cns.reshape(-1, 1),
                           kts.reshape(-1, 1),
                           cts.reshape(-1, 1)))

    node_data_l = add_spring_damper(node_data_l)
    node_data_r = add_spring_damper(node_data_r)
    node_data_b = add_spring_damper(node_data_b)

    # ============ 在Abaqus中添加弹簧-阻尼器到地面 ============
    model = mdb.models['Model-1']
    assembly = model.rootAssembly
    instance = assembly.instances['Part-1-1']

    def add_spring_dashpot(node_data, prefix, dof_n, dof_t):
        """为每个边界节点添加法向和切向弹簧-阻尼器"""
        for row in node_data:
            node_label = int(row[0])
            kn = row[4]
            cn = row[5]
            kt = row[6]
            ct = row[7]
            node_array = instance.nodes.sequenceFromLabels([node_label])
            if len(node_array) == 0:
                print('Node {} does not exist'.format(node_label))
                continue
            region = Region(nodes=node_array)
            assembly.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_normal'.format(prefix, node_label),
                region=region, orientation=None, dof=dof_n,
                springBehavior=ON, springStiffness=kn,
                dashpotBehavior=ON, dashpotCoefficient=cn)
            assembly.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_tangent'.format(prefix, node_label),
                region=region, orientation=None, dof=dof_t,
                springBehavior=ON, springStiffness=kt,
                dashpotBehavior=ON, dashpotCoefficient=ct)

    # 左/右边界: dof_n=1(x方向), dof_t=2(y方向)
    # 底边界:    dof_n=2(y方向), dof_t=1(x方向)
    add_spring_dashpot(node_data_l, prefix='l', dof_n=1, dof_t=2)
    add_spring_dashpot(node_data_r, prefix='r', dof_n=1, dof_t=2)
    add_spring_dashpot(node_data_b, prefix='b', dof_n=2, dof_t=1)

    # ============ 入射角与反射系数计算 ============
    if angle == 0:
        angle = 1e-10  # 避免除零
    else:
        angle = round(angle, 4)

    alpha = np.radians(angle)                    # SV波入射角(弧度)
    alpha_critical = np.arcsin(cs / cp)          # 临界角
    if alpha >= alpha_critical:
        raise ValueError('The incident angle is greater than or equal to the critical angle.')

    beta_p = np.arcsin(cp * np.sin(alpha) / cs)  # 反射P波角度

    # 反射系数 A1 (SV波反射系数), A2 (P波反射系数)
    numerator_A1 = cs ** 2 * np.sin(2 * alpha) * np.sin(2 * beta_p) - cp ** 2 * np.cos(2 * alpha) ** 2
    denominator_A1 = cs ** 2 * np.sin(2 * alpha) * np.sin(2 * beta_p) + cp ** 2 * np.cos(2 * alpha) ** 2
    A1 = numerator_A1 / denominator_A1

    numerator_A2 = 2 * cp * cs * np.sin(2 * alpha) * np.cos(2 * alpha)
    A2 = numerator_A2 / denominator_A1

    # ============ 读取速度时程并积分得到位移 ============
    VEL = np.loadtxt(VELtxt)
    time = VEL[:, 0]
    vel = VEL[:, 1]
    dt = VEL[1, 0] - VEL[0, 0]

    dis = np.zeros_like(vel)
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 梯形积分
    DIS = np.column_stack((time, dis))

    max_time = VEL[-1, 0]
    Ly = ymax - ymin  # 模型高度
    Lx = xmax - xmin  # 模型宽度

    # ============ 计算各节点的波到达延迟时间 ============
    def calc_node_delay(node_data, boundary, alpha, beta_p, cs, cp, Ly, Lx):
        """
        计算节点延迟时间
        :param node_data: 节点数据，n行4列（label, x, y, influence）
        :param boundary: 'l'（左）、'r'（右）、'b'（底）
        :param alpha: 入射角（弧度）
        :param beta_p: P波反射角（弧度）
        :param cs: 剪切波速
        :param cp: 纵波波速
        :param Ly: 高度
        :param Lx: 宽度
        :return: det，n行4列（label, t1, t2, t3）
        """
        n = node_data.shape[0]
        det = np.zeros((n, 4))
        det[:, 0] = node_data[:, 0]

        for i in range(n):
            x0 = node_data[i, 1]
            y0 = node_data[i, 2]

            if boundary == 'l':
                # 入射SV波到达时间
                t1 = y0 * np.cos(alpha) / cs
                # 反射SV波到达时间
                t2 = (2 * Ly - y0) * np.cos(alpha) / cs
                # 反射P波到达时间
                t3 = ((Ly - y0) / (cp * np.cos(beta_p))
                      + (Ly - (Ly - y0) * np.tan(alpha) * np.tan(beta_p)) * np.cos(alpha) / cs)
                det[i, 1] = t1
                det[i, 2] = t2
                det[i, 3] = t3

            elif boundary == 'r':
                t7 = y0 * np.cos(alpha) / cs + Lx * np.sin(alpha) / cs
                t8 = (2 * Ly - y0) * np.cos(alpha) / cs + Lx * np.sin(alpha) / cs
                t9 = ((Ly - y0) / (cp * np.cos(beta_p))
                      + (Ly - (Ly - y0) * np.tan(alpha) * np.tan(beta_p)) * np.cos(alpha) / cs
                      + Lx * np.sin(alpha) / cs)
                det[i, 1] = t7
                det[i, 2] = t8
                det[i, 3] = t9

            elif boundary == 'b':
                t4 = x0 * np.sin(alpha) / cs
                t5 = (2 * Ly + x0 * np.tan(alpha)) * np.cos(alpha) / cs
                t6 = (Ly / (cp * np.cos(beta_p))
                      + (Ly * np.cos(alpha) + x0 * np.sin(alpha)
                         - Ly * np.tan(beta_p) * np.sin(alpha)) / cs)
                det[i, 1] = t4
                det[i, 2] = t5
                det[i, 3] = t6

            else:
                raise ValueError("boundary must be 'l', 'r', or 'b'")

        return det

    det_l = calc_node_delay(node_data_l, 'l', alpha, beta_p, cs, cp, Ly, Lx)
    det_r = calc_node_delay(node_data_r, 'r', alpha, beta_p, cs, cp, Ly, Lx)
    det_b = calc_node_delay(node_data_b, 'b', alpha, beta_p, cs, cp, Ly, Lx)

    # 如果最大延迟超过输入时程长度，则补零延长
    detmax = max(np.max(det_l[:, 1:]), np.max(det_r[:, 1:]), np.max(det_b[:, 1:]))
    if max_time < detmax:
        n_add = int(np.ceil((detmax - max_time) / dt))
        new_times = VEL[-1, 0] + dt * np.arange(1, n_add + 1)
        new_vel = np.zeros((n_add, 2))
        new_vel[:, 0] = new_times
        VEL = np.vstack([VEL, new_vel])
        DIS = np.vstack([DIS, new_vel])

    # ============ 延迟时间对齐到时间步 ============
    def round_delay(det, dt):
        det[:, 1:4] = np.round(det[:, 1:4] / dt) * dt
        return det

    det_l = round_delay(det_l, dt)
    det_r = round_delay(det_r, dt)
    det_b = round_delay(det_b, dt)

    # ============ 信号延迟工具函数 ============
    def delay_signal(u0, delay_t, dt):
        """将信号延迟 delay_t 时间"""
        n_delay = int(np.round(delay_t / dt))
        N = u0.shape[0]
        new_len = N + n_delay
        delayed = np.zeros((new_len, 2))
        delayed[:, 0] = np.arange(new_len) * dt
        delayed[n_delay:, 1] = u0[:, 1]
        return delayed

    def pad_to(arr, length, dt):
        """将数组补零到指定长度"""
        if arr.shape[0] < length:
            pad = np.zeros((length - arr.shape[0], 2))
            pad[:, 0] = np.arange(arr.shape[0], length) * dt
            arr = np.vstack([arr, pad])
        return arr

    # ============ 计算自由场位移和速度 ============
    # 用字典存储中间结果，替代原代码的 globals()
    field_data = {}

    def calc_freefield_u_and_dotu_general(node_data, det, timeseries, dt,
                                           alpha, beta_p, A1, A2,
                                           suffix1, suffix2, prefix):
        """
        对各边界（左、右、底）计算自由场 ux/uy 或 dotux/dotuy 时程
        """
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            idx = np.where(det[:, 0] == node_id)[0][0]
            tA = det[idx, 1]
            tB = det[idx, 2]
            tC = det[idx, 3]

            u0_tA = delay_signal(timeseries, tA, dt)
            u0_tB = delay_signal(timeseries, tB, dt)
            u0_tC = delay_signal(timeseries, tC, dt)

            max_len = max(u0_tA.shape[0], u0_tB.shape[0], u0_tC.shape[0])
            u0_tA = pad_to(u0_tA, max_len, dt)
            u0_tB = pad_to(u0_tB, max_len, dt)
            u0_tC = pad_to(u0_tC, max_len, dt)

            # 自由场位移/速度叠加（入射SV + 反射SV + 反射P）
            ux = (u0_tA[:, 1] * np.cos(alpha)
                  - A1 * u0_tB[:, 1] * np.cos(alpha)
                  + A2 * u0_tC[:, 1] * np.sin(beta_p))
            uy = (-u0_tA[:, 1] * np.sin(alpha)
                  - A1 * u0_tB[:, 1] * np.sin(alpha)
                  - A2 * u0_tC[:, 1] * np.cos(beta_p))

            ux_arr = np.zeros((max_len, 2))
            uy_arr = np.zeros((max_len, 2))
            ux_arr[:, 0] = u0_tA[:, 0]
            uy_arr[:, 0] = u0_tA[:, 0]
            ux_arr[:, 1] = ux
            uy_arr[:, 1] = uy

            field_data['{}-{}-{}'.format(node_id, prefix, suffix1)] = ux_arr
            field_data['{}-{}-{}'.format(node_id, prefix, suffix2)] = uy_arr

    # 计算位移自由场
    calc_freefield_u_and_dotu_general(node_data_l, det_l, DIS, dt, alpha, beta_p, A1, A2, 'ux', 'uy', 'l')
    calc_freefield_u_and_dotu_general(node_data_r, det_r, DIS, dt, alpha, beta_p, A1, A2, 'ux', 'uy', 'r')
    calc_freefield_u_and_dotu_general(node_data_b, det_b, DIS, dt, alpha, beta_p, A1, A2, 'ux', 'uy', 'b')
    # 计算速度自由场
    calc_freefield_u_and_dotu_general(node_data_l, det_l, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'l')
    calc_freefield_u_and_dotu_general(node_data_r, det_r, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'r')
    calc_freefield_u_and_dotu_general(node_data_b, det_b, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'b')

    # ============ 计算自由场应力 ============
    def calc_freefield_sigma_general(node_data, det, VEL, dt,
                                      alpha, beta_p, A1, A2,
                                      GG, cs, lam, cp, prefix):
        """
        对各边界（左、右、底）计算自由场应力 sigmax/sigmay 时程
        """
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            idx = np.where(det[:, 0] == node_id)[0][0]
            tA = det[idx, 1]
            tB = det[idx, 2]
            tC = det[idx, 3]

            v0_tA = delay_signal(VEL, tA, dt)
            v0_tB = delay_signal(VEL, tB, dt)
            v0_tC = delay_signal(VEL, tC, dt)

            max_len = max(v0_tA.shape[0], v0_tB.shape[0], v0_tC.shape[0])
            v0_tA = pad_to(v0_tA, max_len, dt)
            v0_tB = pad_to(v0_tB, max_len, dt)
            v0_tC = pad_to(v0_tC, max_len, dt)

            sin2a = np.sin(2 * alpha)
            cos2a = np.cos(2 * alpha)
            sinbp = np.sin(beta_p)
            sin2bp = np.sin(beta_p) ** 2
            sin2bp_2 = np.sin(2 * beta_p)
            cosbp = np.cos(beta_p)
            cosbp2 = cosbp ** 2

            if prefix == 'l':
                sigmax = (GG / cs * sin2a * (v0_tA[:, 1] - A1 * v0_tB[:, 1])
                          + A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC[:, 1])
                sigmay = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          - A2 * GG * sin2bp_2 / cp * v0_tC[:, 1])
            elif prefix == 'r':
                sigmax = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          - A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC[:, 1])
                sigmay = (GG / cs * cos2a * (-v0_tA[:, 1] - A1 * v0_tB[:, 1])
                          + A2 * GG * sin2bp_2 / cp * v0_tC[:, 1])
            elif prefix == 'b':
                sigmax = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          - A2 * GG * sin2bp_2 / cp * v0_tC[:, 1])
                sigmay = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          + A2 * (lam + 2 * GG * cosbp2) / cp * v0_tC[:, 1])
            else:
                raise ValueError("prefix must be 'l', 'r' or 'b'")

            sigmax_arr = np.zeros((max_len, 2))
            sigmay_arr = np.zeros((max_len, 2))
            sigmax_arr[:, 0] = v0_tA[:, 0]
            sigmay_arr[:, 0] = v0_tA[:, 0]
            sigmax_arr[:, 1] = sigmax
            sigmay_arr[:, 1] = sigmay

            field_data['{}-{}-sigmax'.format(node_id, prefix)] = sigmax_arr
            field_data['{}-{}-sigmay'.format(node_id, prefix)] = sigmay_arr

    calc_freefield_sigma_general(node_data_l, det_l, VEL, dt, alpha, beta_p, A1, A2, GG, cs, lam, cp, 'l')
    calc_freefield_sigma_general(node_data_r, det_r, VEL, dt, alpha, beta_p, A1, A2, GG, cs, lam, cp, 'r')
    calc_freefield_sigma_general(node_data_b, det_b, VEL, dt, alpha, beta_p, A1, A2, GG, cs, lam, cp, 'b')

    # ============ 计算等效节点力 ============
    def calc_equiv_node_force_general(node_data, prefix):
        """
        计算等效节点力 fx, fy
        node_data: [节点号, x, y, A, kn, cn, kt, ct]
        prefix: 'l'（左侧）、'r'（右侧）、'b'（底边）
        """
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            A = node_data[i, 3]       # 影响长度
            kn = node_data[i, 4]
            cn = node_data[i, 5]
            kt = node_data[i, 6]
            ct = node_data[i, 7]

            ux_arr = field_data['{}-{}-ux'.format(node_id, prefix)]
            dotux_arr = field_data['{}-{}-dotux'.format(node_id, prefix)]
            sigmax_arr = field_data['{}-{}-sigmax'.format(node_id, prefix)]
            uy_arr = field_data['{}-{}-uy'.format(node_id, prefix)]
            dotuy_arr = field_data['{}-{}-dotuy'.format(node_id, prefix)]
            sigmay_arr = field_data['{}-{}-sigmay'.format(node_id, prefix)]

            min_len = min(ux_arr.shape[0], dotux_arr.shape[0], sigmax_arr.shape[0],
                          uy_arr.shape[0], dotuy_arr.shape[0], sigmay_arr.shape[0])

            ux = ux_arr[:min_len, 1]
            dotux = dotux_arr[:min_len, 1]
            sigmax = sigmax_arr[:min_len, 1]
            uy = uy_arr[:min_len, 1]
            dotuy = dotuy_arr[:min_len, 1]
            sigmay = sigmay_arr[:min_len, 1]
            time = ux_arr[:min_len, 0]

            # 等效节点力 = 弹簧力 + 阻尼力 + 应力贡献
            if prefix in ('l', 'r'):
                fx = kn * ux + cn * dotux + A * sigmax
                fy = kt * uy + ct * dotuy + A * sigmay
            elif prefix == 'b':
                fx = kt * ux + ct * dotux + A * sigmax
                fy = kn * uy + cn * dotuy + A * sigmay
            else:
                raise ValueError("prefix must be 'l', 'r' or 'b'")

            fx_arr = np.zeros((min_len, 2))
            fy_arr = np.zeros((min_len, 2))
            fx_arr[:, 0] = time
            fy_arr[:, 0] = time
            fx_arr[:, 1] = fx
            fy_arr[:, 1] = fy

            field_data['{}-{}-fx'.format(node_id, prefix)] = fx_arr
            field_data['{}-{}-fy'.format(node_id, prefix)] = fy_arr

    calc_equiv_node_force_general(node_data_l, 'l')
    calc_equiv_node_force_general(node_data_r, 'r')
    calc_equiv_node_force_general(node_data_b, 'b')

    # ============ 创建幅值曲线 (Amplitude) ============
    def batch_add_node_force_amplitude(node_data, prefix):
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            fx_arr = field_data['{}-{}-fx'.format(node_id, prefix)]
            fy_arr = field_data['{}-{}-fy'.format(node_id, prefix)]

            ampli_fx = tuple(tuple(row) for row in fx_arr)
            ampli_fy = tuple(tuple(row) for row in fy_arr)

            name_amp_fx = 'AMP-{}-{}-fx'.format(node_id, prefix)
            name_amp_fy = 'AMP-{}-{}-fy'.format(node_id, prefix)

            mdb.models['Model-1'].TabularAmplitude(
                data=ampli_fx, name=name_amp_fx,
                smooth=SOLVER_DEFAULT, timeSpan=STEP)
            mdb.models['Model-1'].TabularAmplitude(
                data=ampli_fy, name=name_amp_fy,
                smooth=SOLVER_DEFAULT, timeSpan=STEP)

    batch_add_node_force_amplitude(node_data_l, 'l')
    batch_add_node_force_amplitude(node_data_r, 'r')
    batch_add_node_force_amplitude(node_data_b, 'b')

    # ============ 施加集中力载荷 ============
    def batch_add_node_force(node_data, prefix, step_name):
        a = mdb.models['Model-1'].rootAssembly
        instance_name = 'Part-1-1'
        n = a.instances[instance_name].nodes

        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            name_amp_fx = 'AMP-{}-{}-fx'.format(node_id, prefix)
            name_amp_fy = 'AMP-{}-{}-fy'.format(node_id, prefix)
            name_load_fx = 'load-{}-{}-fx'.format(node_id, prefix)
            name_load_fy = 'load-{}-{}-fy'.format(node_id, prefix)

            node_array = n.sequenceFromLabels([node_id])
            if len(node_array) == 0:
                print('Node {} does not exist in instance {}'.format(node_id, instance_name))
                continue

            region = [node_array]
            mdb.models['Model-1'].ConcentratedForce(
                name=name_load_fx, createStepName=step_name,
                region=region, cf1=1.0, amplitude=name_amp_fx,
                distributionType=UNIFORM, field='', localCsys=None)
            mdb.models['Model-1'].ConcentratedForce(
                name=name_load_fy, createStepName=step_name,
                region=region, cf2=1.0, amplitude=name_amp_fy,
                distributionType=UNIFORM, field='', localCsys=None)

    batch_add_node_force(node_data_l, 'l', step_name)
    batch_add_node_force(node_data_r, 'r', step_name)
    batch_add_node_force(node_data_b, 'b', step_name)

if __name__ == '__main__':
    VAB_oblique(angle=15, cs=1754, vv=0.3, density=2500)
