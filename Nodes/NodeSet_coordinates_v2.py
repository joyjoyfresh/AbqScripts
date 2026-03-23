# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
from odbAccess import openOdb
from textRepr import *
from caeModules import *
import csv
import regionToolset

# ====================== 自定义参数==========================
# 模型名称
model_name = '1rf'
# 节点集名称
node_set_name = 'A'
# ==========================================================

# 访问模型和节点集
NodeSet = mdb.models[model_name].rootAssembly.sets[node_set_name]

# 导出节点标签和坐标到CSV文件
with open('coordinates.csv','w') as f:
    f.write("NodeLabel, X, Y, Z\n")
    for node in NodeSet.nodes:
        csv_line = "{}, {}, {}, {}\n".format(node.label, node.coordinates[0], node.coordinates[1], node.coordinates[2])
        f.write(csv_line)