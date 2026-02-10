# -*- coding: utf-8 -*-

import numpy as np
from abaqus import *
from abaqusConstants import *
import part
import regionToolset
import mesh

class VAB_oblique:
    def __init__(self):
        pass

    def get_node_influence(self, p, set_name, sort_axis, ascending=True):
        """
        计算节点集合中每个节点的有效影响长度(2D)或面积。
        :param p: Abaqus Part 对象
        :param set_name: 节点集名称 (String)
        :param sort_axis: 排序轴 'x' 或 'y'
        :param ascending: 是否升序 (Boolean)
        :return: node_data 数组 [NodeLabel, X, Y, InfluenceLength]
        """
        nodes = p.sets[set_name].nodes
        # 提取节点信息：(Label, x, y, z)
        # 假设是2D或3D平面问题，主要关注x, y
        data = []
        for n in nodes:
            coords = n.coordinates
            data.append([n.label, coords[0], coords[1], 0.0]) # 最后一列预留给影响长度

        # 转换为numpy数组以便处理
        data = np.array(data)
        
        # 排序
        axis_idx = 1 if sort_axis == 'x' else 2
        sorted_indices = np.argsort(data[:, axis_idx])
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        
        data = data[sorted_indices]
        
        # 计算影响长度 (Tributary Length)
        # L_i = 0.5 * (dist(i, i-1) + dist(i, i+1))
        n_nodes = len(data)
        for i in range(n_nodes):
            length = 0.0
            coord_curr = data[i, 1:3]
            
            # 与前一个节点的距离一半
            if i > 0:
                coord_prev = data[i-1, 1:3]
                dist_prev = np.linalg.norm(coord_curr - coord_prev)
                length += 0.5 * dist_prev
            
            # 与后一个节点的距离一半
            if i < n_nodes - 1:
                coord_next = data[i+1, 1:3]
                dist_next = np.linalg.norm(coord_curr - coord_next)
                length += 0.5 * dist_next
                
            data[i, 3] = length
            
        return data

    def calc_node_delay(self, node_data, boundary, alpha, beta_p, cs, cp, Ly, Lx):
        """
        计算每个节点的波到达延迟时间。
        :param node_data: get_node_influence 返回的数组
        :param boundary: 'l' (左), 'r' (右), 'b' (底)
        :param alpha: 入射角 (SV波入射角, 弧度)
        :param beta_p: 反射/折射 P波角 (弧度)
        :param cs: 剪切波速
        :param cp: 纵波速
        :param Ly: 模型高度
        :param Lx: 模型宽度
        :return: 带有延迟时间的数组 (最后一列添加 delay)
        """
        # 假设波从底部左侧入射，基于平面波假定
        # 延迟 t = (x * sin(alpha) + y * cos(alpha)) / cs (简化版，具体视波型而定)
        # 对于斜入射SV波，水平视波速 Cx = cs / sin(alpha)
        
        # 注意：这里复刻的是常见的行波效应计算
        # 实际pyd中可能有针对反射波的复杂逻辑，这里采用标准几何延迟
        
        updated_data = []
        ref_x = 0.0
        ref_y = 0.0 # 参考点
        
        # 视波速
        c_app_x = cs / np.sin(alpha) if abs(alpha) > 1e-6 else 1e10
        # 垂直方向传播速度需考虑投影
        
        for row in node_data:
            x, y = row[1], row[2]
            
            # 计算延迟 (以原点(0,0)为参考的波阵面到达时间)
            # t = x * sin(alpha)/cs + y * cos(alpha)/cs
            # 这是一个标准的平面波前方程
            delay = (x * np.sin(alpha) + y * np.cos(alpha)) / cs
            
            # 将延迟添加到行末
            new_row = list(row)
            new_row.append(delay)
            updated_data.append(new_row)
            
        return np.array(updated_data)

    def add_spring_dashpot(self, model_name, part_name, node_data, rho, cs, cp, thickness=1.0):
        """
        添加弹簧阻尼器到边界节点。
        KB边界: 只有阻尼器 (Spring K=0)
        Cn = rho * cp * A (法向)
        Ct = rho * cs * A (切向)
        """
        m = mdb.models[model_name]
        p = m.parts[part_name]
        
        # 创建工程特性 (Engineering Features)
        for row in node_data:
            node_label = int(row[0])
            area = row[3] * thickness # 影响长度 * 厚度
            
            # 计算阻尼系数
            c_normal = rho * cp * area
            c_tangent = rho * cs * area
            
            # 获取节点对象 (这里需要高效处理，逐个查找较慢，通常建议用Set)
            # 为演示逻辑，采用逐个处理
            region = p.sets['Node-{}'.format(node_label)] # 假设已为每个节点建Set，或者直接用Region
            # 实际操作中通常直接通过节点索引创建Region
            # region = regionToolset.Region(nodes=p.nodes[node_label-1:node_label])
            
            # 这里简化逻辑：用户需确保已有Set或自行修改为Region构造
            # 注意：pyd内部可能有一个 batch_create 的过程
            
            # 法向 (X方向或Y方向，取决于边界)
            # 这里假设是一个通用函数，实际使用需区分边界方向
            pass # 具体Abaqus API调用代码略长，核心是计算出 c_normal 和 c_tangent

    def calc_freefield_u_and_dotu_general(self, t, u_input, alpha, boundary):
        """
        计算自由场位移和速度。
        对于斜入射，自由场不仅仅是输入波的简单时移，还包含自由表面的反射放大。
        """
        # 简单近似：自由表面幅值加倍 (SH波或垂直入射)
        # 对于SV斜入射，需利用Zoeppritz方程计算系数
        # 这里仅做框架复刻
        return u_input * 2.0 # 占位符逻辑

    # -----------------------------------------------------------
    # 以下是根据 pyd 字符串提取出的辅助函数签名，进行逻辑填充
    # -----------------------------------------------------------
    
    def round_delay(self, delay, dt):
        """将延迟时间取整为时间步长的倍数"""
        return round(delay / dt) * dt

    def delay_signal(self, signal, delay, dt):
        """
        对信号进行时移
        :param signal: 原始时程数组
        :param delay: 延迟时间
        :param dt: 时间间隔
        """
        n_steps = int(round(delay / dt))
        if n_steps == 0:
            return signal
        
        shifted = np.zeros_like(signal)
        if n_steps > 0:
            shifted[n_steps:] = signal[:-n_steps]
        else:
            # 负延迟（超前），通常不应发生，或者向左移
            shifted[:n_steps] = signal[-n_steps:]
            
        return shifted

# 使用示例 (伪代码)
if __name__ == "__main__":
    # 1. 初始化
    vab = VAB_oblique()
    
    # 2. 获取参数
    # part = mdb.models['Model-1'].parts['Part-1']
    # l_nodes = vab.get_node_influence(part, 'Left_Edge_Nodes', 'y')
    
    # 3. 计算延迟
    # l_nodes_delay = vab.calc_node_delay(l_nodes, 'l', alpha=0.2, beta_p=0.1, cs=200, cp=400, Ly=50, Lx=100)
    
    # 4. 施加边界和荷载...
    print("VAB Oblique Logic Loaded.")