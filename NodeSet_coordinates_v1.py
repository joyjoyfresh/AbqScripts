# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
from odbAccess import openOdb
from textRepr import *
from caeModules import *
import csv
import regionToolset

# ====================== 自定义参数（需根据你的模型修改）==========================
model_name = '1rf'                                 # 模型名称
node_set_name = 'A'                                # 节点集名称
# ==============================================================================

NodeSet = mdb.models[model_name].rootAssembly.sets[node_set_name]

with open('coordinates.csv','w') as f:
    f.write("NodeLabel, X, Y, Z\n")
    for node in NodeSet.nodes:
        csv_line = "{}, {}, {}, {}\n".format(node.label, node.coordinates[0], node.coordinates[1], node.coordinates[2])
        f.write(csv_line)