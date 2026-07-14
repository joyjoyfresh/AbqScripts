# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
import part
import assembly

# =========================自定义参数==========================
# 你的模型名称
model_name = '1rf'
# 多个目标坐标
target_coords_list = [(0.0,0.0,0.0), (5.0,5.0,0.0), (5.0,0.0,0.0),(55.0, 0.0, 0.0),(60.0, 0.0, 0.0),(60.0, 5.0, 0.0)]
# 节点集名称
node_set_name = 'A'
# 坐标容差
tol = 1e-3
# ============================================================

# 访问模型和装配体
myModel = mdb.models[model_name]
root_asm = myModel.rootAssembly

# 存储所有匹配到的有效节点（跨实例去重）
node_label_dict = {}

# 自动遍历根装配体中所有实例
for inst_name in list(root_asm.instances.keys()):
    matched_nodes = []
    # 遍历当前实例的所有节点
    for node in root_asm.instances[inst_name].nodes:
        # 遍历所有目标坐标，匹配则记录
        for target in target_coords_list:
            # 计算欧氏距离，容差内匹配
            dist = ((node.coordinates[0] - target[0])**2 +
                    (node.coordinates[1] - target[1])**2 +
                    (node.coordinates[2] - target[2])**2)**0.5
            # 匹配成功 + 节点未被添加过 → 加入列表
            if dist < tol and node not in matched_nodes:
                matched_nodes.append(node.label)
    node_label_dict[inst_name]=matched_nodes

# 转成Abaqus要求的双层元组格式
node_labels = tuple((k, tuple(v)) for k, v in node_label_dict.items())

# 创建节点集（无实例名依赖）
root_asm.SetFromNodeLabels(nodeLabels=node_labels,name=node_set_name)