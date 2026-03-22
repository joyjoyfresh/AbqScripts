# -*- coding: utf-8 -*-
import sys

# 打印sys.path的所有路径
print("当前sys.path的内容：")
for idx, path in enumerate(sys.path):
    print(f"{idx+1}. {path}")

# 也可以直接打印整个列表（适合快速查看）
#print(sys.path)