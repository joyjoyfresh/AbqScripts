# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
import part
import assembly

# ========================= 自定义参数（需根据模型修改）===========================
model_name = '1rf'                                                  # 你的模型名称
inst_name = 'ground-1'                                              # 装配体实例名
target_coords_list = [(0.0,0.0,0.0), (5.0,5.0,0.0), (5.0,0.0,0.0)]  # 多个目标坐标
node_set_name = 'C'                                                 # 批量节点集名称
tol = 1e-3                                                          # 坐标容差
# ==============================================================================

# 访问模型和装配体实例
myModel = mdb.models[model_name]
myInst = myModel.rootAssembly.instances[inst_name]

matched_nodes = []
# 遍历所有目标坐标，逐个匹配节点
for target in target_coords_list:
    for node in myInst.nodes:
        dist = ((node.coordinates[0] - target[0])**2 +
                (node.coordinates[1] - target[1])**2 +
                (node.coordinates[2] - target[2])**2)**0.5
        if dist < tol and node not in matched_nodes:  # 避免重复添加节点
            matched_nodes.append(node)
# 提取匹配节点的标签
node_labels = tuple([n.label for n in matched_nodes])

# 创建装配体级别节点集
myModel.rootAssembly.SetFromNodeLabels(
    nodeLabels=((inst_name, node_labels),),  # 固定双层括号，不要改！
    name=node_set_name
)