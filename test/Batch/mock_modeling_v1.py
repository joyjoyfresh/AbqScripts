# -*- coding: utf-8 -*-
"""模拟 Abaqus 建模与运算的 Mock 脚本 v1。

用于在无需进入 Abaqus 环境且无需实际计算的前提下，模拟生成后处理所需的元数据与地表响应数据，
从而使整个 Autorun 批处理与后处理收集 pipeline 可以在普通 Python 环境下成功走通。
"""

import os  # 导入系统接口模块
import json  # 导入 JSON 模块
import csv  # 导入 CSV 模块


def main():  # 模拟建模的核心主控逻辑
    """读取配置并生成模拟数据。"""
    print(">>> 启动 Mock 建模脚本...")  # 提示用户
    
    # 1. 尝试读取 Autorun 注入的配置文件 case_config.json
    config_path = 'case_config.json'  # 配置路径
    if not os.path.isfile(config_path):  # 不存在
        print("错误: 当前目录下未检测到 case_config.json")  # 报错
        return
        
    with open(config_path, 'r') as f:  # 打开配置
        case_cfg = json.load(f)  # 加载
        
    # 获取注入的几何参数
    geo_cfg = case_cfg.get('geometry', {})  # 几何配置
    x_crest = geo_cfg.get('x_crest', 1000.0)  # 坡顶棱
    x_toe = geo_cfg.get('x_toe', 1200.0)  # 坡脚棱
    h_val = geo_cfg.get('h', 200.0)  # 高度 h
    H_val = geo_cfg.get('H', 400.0)  # 高度 H
    
    # 2. 模拟生成 case_meta.json 并注入相应字段
    meta = {  # 元数据结构
        "model_type": "hybrid_slope_mock",  # 模型类型
        "model_script": "slope_frame_ssi_full_mock_v1.py",  # 建模脚本
        "incident_angle": 15.0,  # 入射角
        "mesh_size": 2.0,  # 网格尺寸
        "geometry": {  # 几何
            "x_crest": x_crest,  # 坡顶
            "x_toe": x_toe,  # 坡脚
            "h": h_val,  # 下部高度
            "H": H_val,  # 总高度
            "i": 30.0,  # 坡度角
            "total_L": 1800.0,  # 总长
            "left_flat": 1000.0,  # 上平台
            "h_over_H": 0.5,  # 高度比
            "bedrock_thickness": 200.0,  # 基岩厚度
            "w_slope": 200.0,  # 坡长
            "H_minus_h": H_val - h_val  # 坡高
        },
        "derived": {  # 派生参数
            "n_finite_layers": 2,  # 有限层数
            "n_layers_total": 3,  # 总层数
            "vs_bedrock": 2000.0,  # 基岩速度
            "vs_surface": 1600.0,  # 表层速度
            "vs_cover": 800.0,  # 覆盖速度
            "vr_over_vs2": 2.5,  # 剪切波速比
            "vs1_over_vs2": 2.0,  # 层间波速比
            "slope_height": 200.0,  # a0 高度
            "a0_base": 0.5  # a0 基数
        },
        "layers": [  # 覆盖层列表
            {"name": "surface", "thickness": 150.0, "cs": 1600.0, "density": 2500.0, "vv": 0.3},
            {"name": "overlying", "thickness": None, "cs": 800.0, "density": 2500.0, "vv": 0.3}
        ],
        "ff_normalization": {  # 归一化参考
            "factor_h": 2.0,  # 水平因子
            "factor_v": 1.0  # 垂直因子
        },
        "ff_theory": {  # 理论台阶值
            "left": {"taf_h": 1.5, "taf_v": 0.2},  # 左侧
            "right": {"taf_h": 1.2, "taf_v": 0.1},  # 右侧
            "fc_used": 4.0  # 计算主频
        }
    }
    
    with open('case_meta.json', 'w') as f:  # 打开文件
        json.dump(meta, f, indent=2)  # 写入 json
    print(">>> 已生成 case_meta.json")  # 提示
    
    # 3. 模拟生成地表响应 CSV：surface_response_ricker_wavelet_4Hz.csv
    # 沿地表从 x=0 到 x=1800 产生 19 个节点的数据
    header = ['x', 'y', 'PGA_h', 'PGA_v', 'AF_h', 'TAF_h', 'AF_v', 'TAF_v', 'V_over_H', 'ff_side']  # 表头
    rows = []  # 数据行
    for i in range(19):  # 循环
        x = i * 100.0  # 步长 100m
        y = 600.0 if x <= x_crest else (600.0 - (x - x_crest) if x < x_toe else 400.0)  # y 坐标
        
        # 沿地表在坡棱段模拟响应的放大
        pga_h = 1.6 + (0.6 * (x - x_crest) / (x_toe - x_crest) if x_crest < x < x_toe else (0.3 if x <= x_crest else 0.0))  # 水平峰值
        pga_v = 0.4 + (0.5 * (x - x_crest) / (x_toe - x_crest) if x_crest < x < x_toe else (0.1 if x <= x_crest else 0.0))  # 垂直峰值
        
        af_h = pga_h / 2.0  # 水平放大
        af_v = pga_v / 2.0  # 垂直放大
        taf_h = af_h / 1.5 if x <= x_toe else af_h / 1.2  # 水平地形放大
        taf_v = af_v / 0.2 if x <= x_toe else af_v / 0.1  # 垂直地形放大
        side = 'left' if x <= x_toe else 'right'  # 同侧分区
        
        rows.append([x, y, pga_h, pga_v, af_h, taf_h, af_v, taf_v, pga_v / pga_h, side])  # 组合行
        
    csv_path = 'surface_response_ricker_wavelet_4Hz.csv'  # 目标路径
    with open(csv_path, 'w') as f_out:  # 打开
        writer = csv.writer(f_out)  # 写入器
        writer.writerow(header)  # 写表头
        for r in rows:  # 写行
            writer.writerow(r)  # 写入
    print(">>> 已生成 %s" % csv_path)  # 提示
    
    # 4. 模拟生成地表摘要 surface_summary.json，主要提供 AR_max 和 suspect 字段
    summary = {  # 摘要结构
        "schema_version": 1,  # 版本号
        "records": [
            {
                "record": "ricker_wavelet_4Hz",  # 地震记录名
                "n_nodes": 19,  # 节点数
                "AR_max": 2.2,  # 峰值放大
                "suspect": False  # 是否异常
            }
        ]
    }
    with open('surface_summary.json', 'w') as f:  # 打开文件
        json.dump(summary, f, indent=2)  # 写入 json
    print(">>> 已生成 surface_summary.json")  # 提示
    print(">>> Mock 建模全部完成！")  # 提示完成


if __name__ == '__main__':  # 入口
    main()  # 运行
