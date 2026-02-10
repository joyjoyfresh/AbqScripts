# -*- coding: utf-8 -*-
"""
粘弹性边界条件脚本 (适用于Abaqus 2021)

功能：
1. 自动创建粘弹性边界条件，包括弹簧-阻尼器系统的参数定义与配置
2. 将地震输入数据等效转换为模型边界节点力
3. 完整的模型初始化、边界条件施加、荷载转换及分析步设置流程
4. 参数化设计，允许用户通过变量调整相关参数
5. 输出必要的计算结果信息，包括边界节点力时程数据及相关状态报告

作者：Abaqus Scripting Team
日期：2026-02-10
"""

import numpy as np
from abaqus import *
from abaqusConstants import *
import part
import regionToolset
import mesh
import visualization
import os

class ViscousSpringBoundary:
    """粘弹性边界条件类"""
    
    def __init__(self, model_name="Model-1", part_name="Part-1"):
        """
        初始化粘弹性边界条件类
        
        Args:
            model_name (str): 模型名称
            part_name (str): 零件名称
        """
        self.model_name = model_name
        self.part_name = part_name
        self.model = None
        self.part = None
        self.node_data = {}
        self.earthquake_data = None
        self.results = {}
    
    def initialize_model(self, params):
        """
        初始化模型
        
        Args:
            params (dict): 模型参数
        """
        try:
            # 获取或创建模型
            if self.model_name in mdb.models:
                self.model = mdb.models[self.model_name]
                print(f"使用现有模型: {self.model_name}")
            else:
                self.model = mdb.Model(name=self.model_name)
                print(f"创建新模型: {self.model_name}")
            
            # 获取或创建零件
            if self.part_name in self.model.parts:
                self.part = self.model.parts[self.part_name]
                print(f"使用现有零件: {self.part_name}")
            else:
                # 创建默认零件（用户可根据需要修改）
                self.part = self.model.Part(name=self.part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
                print(f"创建新零件: {self.part_name}")
            
            # 创建分析步
            self.create_analysis_steps(params)
            
            return True
        except Exception as e:
            print(f"初始化模型失败: {str(e)}")
            return False
    
    def create_analysis_steps(self, params):
        """
        创建分析步
        
        Args:
            params (dict): 分析步参数
        """
        # 创建初始分析步
        self.model.StaticStep(name="Initial", previous="Initial", description="Initial step")
        
        # 创建动力分析步
        self.model.DynamicStep(
            name="Dynamic",
            previous="Initial",
            response=STEADY_STATE,
            timePeriod=params["analysis_time"],
            maxNumInc=params["max_time_increments"],
            initialInc=params["initial_time_increment"],
            minInc=params["min_time_increment"],
            maxInc=params["max_time_increment"]
        )
        print("创建分析步完成")
    
    def load_earthquake_data(self, filepath, dt=None):
        """
        加载地震波数据
        
        Args:
            filepath (str): 地震波文件路径
            dt (float): 时间步长（如果文件中没有）
        """
        try:
            # 读取地震波文件
            data = np.loadtxt(filepath)
            
            # 检查数据格式
            if data.ndim == 1:
                # 只有加速度数据，需要创建时间数组
                if dt is None:
                    raise ValueError("文件只有加速度数据时，必须提供dt参数")
                time = np.arange(0, len(data) * dt, dt)
                accel = data
            else:
                # 第一列是时间，第二列是加速度
                time = data[:, 0]
                accel = data[:, 1]
            
            self.earthquake_data = {
                "time": time,
                "accel": accel,
                "dt": time[1] - time[0] if len(time) > 1 else dt,
                "duration": time[-1] if len(time) > 0 else 0
            }
            
            print(f"加载地震波数据成功: {filepath}")
            print(f"地震波时长: {self.earthquake_data['duration']}秒")
            print(f"时间步长: {self.earthquake_data['dt']}秒")
            print(f"数据点数量: {len(time)}")
            
            return True
        except Exception as e:
            print(f"加载地震波数据失败: {str(e)}")
            return False
    
    def calculate_node_influence(self, node_set_name, sort_axis='x', ascending=True):
        """
        计算节点影响长度/面积
        
        Args:
            node_set_name (str): 节点集名称
            sort_axis (str): 排序轴
            ascending (bool): 是否升序
        """
        try:
            nodes = self.part.sets[node_set_name].nodes
            
            # 提取节点信息
            data = []
            for n in nodes:
                coords = n.coordinates
                data.append([n.label, coords[0], coords[1], coords[2], 0.0])  # 最后一列预留给影响长度
            
            # 转换为numpy数组
            data = np.array(data)
            
            # 排序
            axis_idx = 1 if sort_axis == 'x' else 2 if sort_axis == 'y' else 3
            sorted_indices = np.argsort(data[:, axis_idx])
            if not ascending:
                sorted_indices = sorted_indices[::-1]
            
            data = data[sorted_indices]
            
            # 计算影响长度
            n_nodes = len(data)
            for i in range(n_nodes):
                length = 0.0
                coord_curr = data[i, 1:4]
                
                # 与前一个节点的距离一半
                if i > 0:
                    coord_prev = data[i-1, 1:4]
                    dist_prev = np.linalg.norm(coord_curr - coord_prev)
                    length += 0.5 * dist_prev
                
                # 与后一个节点的距离一半
                if i < n_nodes - 1:
                    coord_next = data[i+1, 1:4]
                    dist_next = np.linalg.norm(coord_curr - coord_next)
                    length += 0.5 * dist_next
                
                data[i, 4] = length
            
            self.node_data[node_set_name] = data
            print(f"计算节点影响长度完成: {node_set_name}")
            return data
        except Exception as e:
            print(f"计算节点影响长度失败: {str(e)}")
            return None
    
    def add_viscous_spring_boundary(self, node_set_name, params):
        """
        添加粘弹性边界条件
        
        Args:
            node_set_name (str): 节点集名称
            params (dict): 边界条件参数
        """
        try:
            if node_set_name not in self.node_data:
                self.calculate_node_influence(node_set_name)
            
            node_data = self.node_data[node_set_name]
            rho = params["material_density"]
            cs = params["shear_wave_velocity"]
            cp = params["compressional_wave_velocity"]
            thickness = params.get("thickness", 1.0)
            
            # 计算弹簧和阻尼器参数
            for row in node_data:
                node_label = int(row[0])
                influence_length = row[4]
                area = influence_length * thickness
                
                # 计算阻尼系数
                c_normal = rho * cp * area
                c_tangent = rho * cs * area
                
                # 计算弹簧系数
                k_normal = params.get("spring_stiffness_normal", 0.0)
                k_tangent = params.get("spring_stiffness_tangent", 0.0)
                
                # 创建弹簧阻尼器
                # 注意：这里需要根据实际边界方向调整力的方向
                # 以下代码为示例，实际使用时需要根据具体情况修改
                
                # 获取节点
                node = self.part.nodes[node_label - 1]
                region = regionToolset.Region(nodes=[node])
                
                # 添加阻尼器（法向）
                self.model.DasDamper(
                    name=f"Damper_Normal_{node_label}",
                    region=region,
                    u1=0.0, u2=0.0, u3=0.0,
                    ur1=0.0, ur2=0.0, ur3=0.0,
                    amplitude=UNSET
                )
                
                # 添加阻尼器（切向）
                self.model.DasDamper(
                    name=f"Damper_Tangent_{node_label}",
                    region=region,
                    u1=0.0, u2=0.0, u3=0.0,
                    ur1=0.0, ur2=0.0, ur3=0.0,
                    amplitude=UNSET
                )
                
                # 添加弹簧（如果需要）
                if k_normal > 0:
                    self.model.Spring(
                        name=f"Spring_Normal_{node_label}",
                        region=region,
                        u1=0.0, u2=0.0, u3=0.0,
                        ur1=0.0, ur2=0.0, ur3=0.0,
                        amplitude=UNSET
                    )
                
                if k_tangent > 0:
                    self.model.Spring(
                        name=f"Spring_Tangent_{node_label}",
                        region=region,
                        u1=0.0, u2=0.0, u3=0.0,
                        ur1=0.0, ur2=0.0, ur3=0.0,
                        amplitude=UNSET
                    )
            
            print(f"添加粘弹性边界条件完成: {node_set_name}")
            return True
        except Exception as e:
            print(f"添加粘弹性边界条件失败: {str(e)}")
            return False
    
    def convert_earthquake_to_forces(self, node_set_name, params):
        """
        将地震波转换为边界节点力
        
        Args:
            node_set_name (str): 节点集名称
            params (dict): 转换参数
        """
        try:
            if self.earthquake_data is None:
                raise ValueError("请先加载地震波数据")
            
            if node_set_name not in self.node_data:
                self.calculate_node_influence(node_set_name)
            
            node_data = self.node_data[node_set_name]
            rho = params["material_density"]
            thickness = params.get("thickness", 1.0)
            
            # 计算节点力
            forces = []
            for row in node_data:
                node_label = int(row[0])
                influence_length = row[4]
                area = influence_length * thickness
                
                # 计算节点质量
                mass = rho * area * params.get("element_size", 1.0)
                
                # 计算节点力时程
                node_forces = mass * self.earthquake_data["accel"]
                forces.append({
                    "node_label": node_label,
                    "forces": node_forces
                })
            
            self.results["boundary_forces"] = forces
            print(f"地震波转换为边界节点力完成: {node_set_name}")
            return forces
        except Exception as e:
            print(f"地震波转换为边界节点力失败: {str(e)}")
            return None
    
    def create_analysis_steps(self, params):
        """
        创建分析步
        
        Args:
            params (dict): 分析步参数
        """
        # 创建初始分析步
        self.model.StaticStep(name="Initial", previous="Initial", description="Initial step")
        
        # 创建动力分析步
        self.model.DynamicStep(
            name="Dynamic",
            previous="Initial",
            response=STEADY_STATE,
            timePeriod=params["analysis_time"],
            maxNumInc=params["max_time_increments"],
            initialInc=params["initial_time_increment"],
            minInc=params["min_time_increment"],
            maxInc=params["max_time_increment"]
        )
        print("创建分析步完成")
    
    def export_results(self, output_dir):
        """
        导出结果
        
        Args:
            output_dir (str): 输出目录
        """
        try:
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 导出边界节点力时程数据
            if "boundary_forces" in self.results:
                forces = self.results["boundary_forces"]
                for force_data in forces:
                    node_label = force_data["node_label"]
                    node_forces = force_data["forces"]
                    
                    # 创建输出文件
                    output_file = os.path.join(output_dir, f"node_{node_label}_forces.txt")
                    
                    # 准备数据
                    data = np.column_stack((self.earthquake_data["time"], node_forces))
                    
                    # 写入文件
                    np.savetxt(output_file, data, header="Time (s)    Force (N)", fmt="%.6f %.6f")
                    print(f"导出节点力时程数据: {output_file}")
            
            # 导出状态报告
            report_file = os.path.join(output_dir, "status_report.txt")
            with open(report_file, "w") as f:
                f.write("粘弹性边界条件分析报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"模型名称: {self.model_name}\n")
                f.write(f"零件名称: {self.part_name}\n")
                if self.earthquake_data:
                    f.write(f"地震波时长: {self.earthquake_data['duration']}秒\n")
                    f.write(f"时间步长: {self.earthquake_data['dt']}秒\n")
                f.write(f"节点集数量: {len(self.node_data)}\n")
                f.write(f"边界节点力文件数量: {len(self.results.get('boundary_forces', []))}\n")
                f.write("\n分析完成！\n")
            
            print(f"导出状态报告: {report_file}")
            return True
        except Exception as e:
            print(f"导出结果失败: {str(e)}")
            return False

def main():
    """
    主函数
    """
    # 参数配置
    params = {
        # 模型参数
        "model_name": "ViscousSpringModel",
        "part_name": "ViscousSpringPart",
        
        # 材料参数
        "material_density": 2500.0,  # kg/m^3
        "shear_wave_velocity": 300.0,  # m/s
        "compressional_wave_velocity": 600.0,  # m/s
        "thickness": 1.0,  # m
        
        # 边界条件参数
        "spring_stiffness_normal": 0.0,  # N/m
        "spring_stiffness_tangent": 0.0,  # N/m
        
        # 分析控制参数
        "analysis_time": 30.0,  # 分析时长（秒）
        "max_time_increments": 10000,  # 最大时间增量数
        "initial_time_increment": 0.001,  # 初始时间增量（秒）
        "min_time_increment": 1e-6,  # 最小时间增量（秒）
        "max_time_increment": 0.01,  # 最大时间增量（秒）
        
        # 其他参数
        "element_size": 1.0,  # 单元尺寸（m）
    }
    
    # 地震波文件路径
    earthquake_file = "path/to/earthquake.txt"  # 用户需要修改为实际路径
    
    # 输出目录
    output_dir = "output"  # 用户需要修改为实际路径
    
    # 节点集名称
    boundary_node_sets = ["LeftBoundary", "RightBoundary", "BottomBoundary"]  # 用户需要修改为实际节点集名称
    
    # 创建粘弹性边界条件对象
    vab = ViscousSpringBoundary(
        model_name=params["model_name"],
        part_name=params["part_name"]
    )
    
    # 初始化模型
    if not vab.initialize_model(params):
        print("初始化模型失败，退出程序")
        return
    
    # 加载地震波数据
    if not vab.load_earthquake_data(earthquake_file):
        print("加载地震波数据失败，退出程序")
        return
    
    # 处理每个边界节点集
    for node_set in boundary_node_sets:
        print(f"\n处理边界节点集: {node_set}")
        
        # 计算节点影响长度
        vab.calculate_node_influence(node_set)
        
        # 添加粘弹性边界条件
        vab.add_viscous_spring_boundary(node_set, params)
        
        # 转换地震波为边界节点力
        vab.convert_earthquake_to_forces(node_set, params)
    
    # 导出结果
    if not vab.export_results(output_dir):
        print("导出结果失败，退出程序")
        return
    
    print("\n粘弹性边界条件分析完成！")

if __name__ == "__main__":
    main()
