#encoding:utf-8
# -*- coding: mbcs -*-
from multiprocessing import cpu_count
from abaqus import *
from abaqusConstants import *
from odbAccess import *
import __main__
import csv
import os
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import math

# 提取反力数据
def get_rf(odb_name,coor):
    path = 'Job-' + odb_name + '.odb'
    odb = openOdb(path=path)
    # 选择第一个分析步
    step1 = odb.steps['Step-1']
    # 选择最后一帧
    lastFrame = step1.frames[-1]
    # 选定对应集合
    Set_nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets[odb_name]
    # 建立节点编号,距波源距离,相对面积集合
    array_label = [0]*len(Set_nodes.nodes)
    array_distence = [0]*len(Set_nodes.nodes)
    array_RF = [0]*len(Set_nodes.nodes)
    array_location = [0]*len(Set_nodes.nodes)
    # 指定输出变量
    RF = lastFrame.fieldOutputs['RF']
    RF = RF.getSubset(region=Set_nodes)
    # 全局波源坐标（后续可外部定义）
    x0 = 25
    y0 = 0
    # 遍历节点计算距离和位置
    for i in range(len(Set_nodes.nodes)):
        array_label[i] = Set_nodes.nodes[i].label
        array_distence[i] = ((Set_nodes.nodes[i].coordinates[0]-x0)**2 + 
                             (Set_nodes.nodes[i].coordinates[1]-y0)**2)**0.5
        array_location[i] = Set_nodes.nodes[i].coordinates[1]
    # 提取反力数据
    for i in range(len(RF.values)):
        array_RF[i] = abs(RF.values[i].data[coor])
    odb.close()
    return (array_label, array_distence, array_RF, array_location)

# 定义节点弹簧阻尼器（接地）
def spring(node_number,Kn,Cn,Kt,Ct,JUGG,name_spring,A0):
    a = mdb.models['Model-1'].rootAssembly
    n1 = a.instances['Part-1-1'].nodes
    region = [n1.sequenceFromLabels([node_number])]  # 节点区域
    name_n = name_spring + '-n'
    name_t = name_spring + '-t'
    # 施加阻尼器和弹簧-法线方向为1
    if JUGG == 1:
        mdb.models['Model-1'].rootAssembly.engineeringFeatures.SpringDashpotToGround(
            name=name_n, region=region, orientation=None, dof=1,
            springBehavior=ON, springStiffness=A0*Kn, 
            dashpotBehavior=ON, dashpotCoefficient=A0*Cn)
        mdb.models['Model-1'].rootAssembly.engineeringFeatures.SpringDashpotToGround(
            name=name_t, region=region, orientation=None, dof=2,
            springBehavior=ON, springStiffness=A0*Kt, 
            dashpotBehavior=ON, dashpotCoefficient=A0*Ct)
    # 判定法线方向为2
    if JUGG == 2:
        mdb.models['Model-1'].rootAssembly.engineeringFeatures.SpringDashpotToGround(
            name=name_n, region=region, orientation=None, dof=2,
            springBehavior=ON, springStiffness=A0*Kn, 
            dashpotBehavior=ON, dashpotCoefficient=A0*Cn)
        mdb.models['Model-1'].rootAssembly.engineeringFeatures.SpringDashpotToGround(
            name=name_t, region=region, orientation=None, dof=1,
            springBehavior=ON, springStiffness=A0*Kt, 
            dashpotBehavior=ON, dashpotCoefficient=A0*Ct)

# 定义等效节点集中荷载
def lord(node_number,f,JUGG_lord,name_lord,step_name,ampli,name_ampli):
    a = mdb.models['Model-1'].rootAssembly
    n1 = a.instances['Part-1-1'].nodes
    # 设置幅值曲线
    mdb.models['Model-1'].TabularAmplitude(name=name_ampli, timeSpan=STEP, 
                                           smooth=SOLVER_DEFAULT, data=ampli)
    # 判断力的方向-沿dof1方向
    if JUGG_lord == 1:
        region = [n1.sequenceFromLabels([node_number])]
        mdb.models['Model-1'].ConcentratedForce(name=name_lord, createStepName=step_name, 
                                                region=region, cf1=f,amplitude=name_ampli, 
                                                distributionType=UNIFORM, field='', localCsys=None)
    # 沿dof2方向
    if JUGG_lord == 2:
        region = [n1.sequenceFromLabels([node_number])]
        mdb.models['Model-1'].ConcentratedForce(name=name_lord, createStepName=step_name, 
                                                region=region, cf2=f,amplitude=name_ampli, 
                                                distributionType=UNIFORM, field='', localCsys=None)

# 读取位移和速度文件数据
def node_u_v():
    index = 0
    # 读取位移文件
    with open('E:/ABAQUS/TEMP/NIANTANXING/DISP.txt' ) as f:
        reader1 = f.readline()
        while reader1:
            reader1 = f.readline()
            index = index + 1
    # 初始化数组
    array_u = [[0]*2 for i in range(index)]
    array_v = [[0]*2 for i in range(index)]
    # 读取位移数据
    with open('E:/ABAQUS/TEMP/NIANTANXING/DISP.txt') as f:
        reader1 = f.readline()
        location = 0
        while reader1:
            b = reader1.split(',')
            array_u[location][0] = float(b[0])
            array_u[location][1] = float(b[1])
            reader1 = f.readline()
            location = location + 1
    # 读取速度数据
    with open('E:/ABAQUS/TEMP/NIANTANXING/v.txt') as f:
        reader1 = f.readline()
        location = 0
        while reader1:
            b = reader1.split(',')
            array_v[location][0] = float(b[0])
            array_v[location][1] = float(b[1])
            reader1 = f.readline()
            location = location + 1
    return(array_u, array_v)

# 线性插值函数计算节点处位移和速度延迟
def solve_u_v(x,t_jugg,array_u_v):
    result = 0
    time_solve = x - t_jugg
    if(time_solve > 0):
        for i in range(1,len(array_u_v)):
            if (array_u_v[i-1][0] <= time_solve <= array_u_v[i][0]):
                result = array_u_v[i-1][1] + (time_solve-array_u_v[i-1][0]) * \
                (array_u_v[i][1]-array_u_v[i-1][1])/(array_u_v[i][0]-array_u_v[i-1][0])
                break
    return(result)

# 计算入射波和反射波延迟函数（位移+速度）
def node_dis_veto(node_h,input,H,Cs):
    partition = 400
    time_add = 0
    time_max = input[len(input)-1][0]
    # 入射波延迟
    T_RS = node_h / Cs
    # 反射波延迟
    T_FS = (2*H - node_h) / Cs
    # 判定时间相位最大值
    if T_RS >= T_FS:
        time_add = T_RS
    else:
        time_add = T_FS
    time_real = time_max + time_add
    partition_number = time_real / partition
    array_time = [0+i*partition_number for i in range(partition+1)]
    array_save_rsb_y = [0]*len(array_time)
    array_save_fsb_y = [0]*len(array_time)
    # 二维输出列表
    array_save_output = [[0]*2 for j in range(len(array_save_rsb_y))]
    array_save_output_u_y = [[0]*2 for j in range(len(array_save_rsb_y))]
    index = 0
    # 计算各点插值
    for i in array_time:
        array_save_rsb_y[index] = solve_u_v(i, T_RS, input)
        array_save_fsb_y[index] = solve_u_v(i, T_FS, input)
        index += 1
    # 合成入射波+反射波
    for i in range(len(array_save_fsb_y)):
        array_save_output[i][0] = array_time[i]
        array_save_output[i][1] = array_save_rsb_y[i] + array_save_fsb_y[i]
        array_save_output_u_y[i][0] = array_time[i]
        array_save_output_u_y[i][1] = -(array_save_rsb_y[i] - array_save_fsb_y[i])/Cs
    # 补充时间步结束点
    step_time = 1  # 与全局分析步时间一致
    array_save_output_2 = array_save_output + [[step_time,0]]
    array_save_output_u_y_2 = array_save_output_u_y + [[step_time,0]]
    return(array_save_output_2, array_save_output_u_y_2)

# 计算弹簧刚度和阻尼器系数
def k_c(G,R,Cs,Cp,P):
    nt = 0.5
    nn = 1
    # 切向弹簧刚度
    Kt = nt * G / R
    # 法向阻尼器阻尼
    Ct = P * Cs
    # 法向弹簧刚度
    Kn = nn * G / R
    # 切向阻尼器阻尼
    Cn = P * Cp
    return(Kn, Cn, Kt, Ct)

# 复制基础模型
def copy_model():
    mdb.Model(name='Model-X0', objectToCopy=mdb.models['Model-1'])
    mdb.Model(name='Model-X1', objectToCopy=mdb.models['Model-1'])
    mdb.Model(name='Model-Z0', objectToCopy=mdb.models['Model-1'])

# 定义各模型边界条件和荷载
def boundry_conditions():
    # Model-X0 边界条件
    a = mdb.models['Model-X0'].rootAssembly
    region = a.instances['Part-1-1'].surfaces['X0']
    mdb.models['Model-X0'].Pressure(name='Load-1',createStepName='Step-1',
                                    region=region, distributionType=UNIFORM, 
                                    field='', magnitude=1.0, amplitude=UNSET)
    region = a.instances['Part-1-1'].sets['X-0']
    mdb.models['Model-X0'].DisplacementBC(name='BC-1',createStepName='Initial',
                                          region=region, u1=SET, u2=UNSET, ur3=UNSET, 
                                          amplitude=UNSET, distributionType=UNIFORM, 
                                          fieldName='', localCsys=None)
    # Model-X1 边界条件
    a = mdb.models['Model-X1'].rootAssembly
    region = a.instances['Part-1-1'].surfaces['X1']
    mdb.models['Model-X1'].Pressure(name='Load-1',createStepName='Step-1',
                                    region=region, distributionType=UNIFORM, 
                                    field='', magnitude=1.0, amplitude=UNSET)
    region = a.instances['Part-1-1'].sets['X-1']
    mdb.models['Model-X1'].DisplacementBC(name='BC-1',createStepName='Initial',
                                          region=region, u1=SET, u2=UNSET, ur3=UNSET, 
                                          amplitude=UNSET, distributionType=UNIFORM, 
                                          fieldName='', localCsys=None)
    # Model-Z0 边界条件
    a = mdb.models['Model-Z0'].rootAssembly
    region = a.instances['Part-1-1'].surfaces['Z0']
    mdb.models['Model-Z0'].Pressure(name='Load-1',createStepName='Step-1',
                                    region=region, distributionType=UNIFORM, 
                                    field='', magnitude=1.0, amplitude=UNSET)
    region = a.instances['Part-1-1'].sets['Z-0']
    mdb.models['Model-Z0'].DisplacementBC(name='BC-1',createStepName='Initial',
                                          region=region, u1=UNSET, u2=SET, ur3=UNSET, 
                                          amplitude=UNSET, distributionType=UNIFORM, 
                                          fieldName='', localCsys=None)

# 创建分析作业
def Creat_Job():
    # 创建X0 作业
    mdb.Job(name='Job-X0', model='Model-X0', description='', type=ANALYSIS, 
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, 
            echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
            userSubroutine='', scratch='', resultsFormat=ODB, 
            multiprocessingMode=DEFAULT, numCpus=5, numDomains=5, numGPUs=0)
    # 创建X1 作业
    mdb.Job(name='Job-X1', model='Model-X1', description='', type=ANALYSIS, 
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, 
            echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
            userSubroutine='', scratch='', resultsFormat=ODB, 
            multiprocessingMode=DEFAULT, numCpus=5, numDomains=5, numGPUs=0)
    # 创建Z0 作业
    mdb.Job(name='Job-Z0', model='Model-Z0', description='', type=ANALYSIS, 
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, 
            echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
            userSubroutine='', scratch='', resultsFormat=ODB, 
            multiprocessingMode=DEFAULT, numCpus=5, numDomains=5, numGPUs=0)

# 提交并运行作业
def Run_JOB():
    mdb.jobs['Job-X0'].submit()
    mdb.jobs['Job-X0'].waitForCompletion()
    mdb.jobs['Job-X1'].submit()
    mdb.jobs['Job-X1'].waitForCompletion()
    mdb.jobs['Job-Z0'].submit()
    mdb.jobs['Job-Z0'].waitForCompletion()

# ====================== 脚本主执行入口 ======================
if __name__ == '__main__':
    # 底面中点坐标（波源）
    x0 = 25
    y0 = 0
    H = 50
    y_down = 0
    # 分析步时间长度
    step_time = 1
    # 材料参数
    E = 2e8          # 弹性模量
    U = 0.25         # 泊松比
    P = 2000         # 密度
    G = 0.5*E/(1+U)  # 剪切模量
    # 波速计算
    Cs = (G/P)**0.5  # 剪切波波速
    Cp = ((U+2*G)/P)**0.5  # 纵波波速
    # 等效节点荷载分析步名称
    step_name = 'Step-1'

    # 模型初始化流程
    copy_model()       # 复制基础模型
    boundry_conditions()  # 定义边界条件
    Creat_Job()        # 创建分析作业
    Run_JOB()          # 运行作业并获取ODB结果

    # 读取各边界反力结果
    Result_X0 = get_rf('X0',0)
    Result_X1 = get_rf('X1',0)
    Result_Z0 = get_rf('Z0',1)
    # 读取地震波位移和速度数据
    input = node_u_v()

    # ====================== X0 边界处理 ======================
    for i in range(len(Result_X0[0])):
        A0 = Result_X0[2][i]
        R = Result_X0[1][i]
        # 计算弹簧阻尼参数
        k_c_1 = k_c(G,R,Cs,Cp,P)
        Kn,Cn,Kt,Ct = k_c_1[0],k_c_1[1],k_c_1[2],k_c_1[3]
        JUGG = 1  # 法线方向
        name_spring = 'X0-'+str(i+1)
        # 设置人工边界弹簧阻尼
        spring(Result_X0[0][i],Kn,Cn,Kt,Ct,JUGG,name_spring,A0)
        # 计算波延迟位移和速度
        node_h = Result_X0[3][i]
        displacement_result = node_dis_veto(node_h,input[0],H,Cs)
        v_result = node_dis_veto(node_h,input[1],H,Cs)
        u_node = displacement_result[0]
        v_node = v_result[0]
        u_y = v_result[1]
        # 计算节点荷载
        f_y_bx = [[0]*2 for j in range(len(u_node))]
        f_y_by = [[0]*2 for j in range(len(u_node))]
        for j in range(len(u_node)):
            f_y_bx[j][0] = u_node[j][0]
            f_y_bx[j][1] = A0*(Kn*u_node[j][1]+Cn*v_node[j][1])
            f_y_by[j][1] = -A0*P*u_y[j][1]*Cs**2
            f_y_by[j][0] = u_node[j][0]
            if u_y[j][1]==0:
                f_y_by[j][1] = 0
        # 施加节点荷载
        f = 1
        name_ampli_x = 'X0-x-'+str(i+1)
        name_ampli_y = 'X0-y-'+str(i+1)
        JUGG_lord = 1
        lord(Result_X0[0][i],f,JUGG_lord,name_ampli_x,step_name,f_y_bx,name_ampli_x)
        JUGG_lord = 2
        lord(Result_X0[0][i],f,JUGG_lord,name_ampli_y,step_name,f_y_by,name_ampli_y)
    print('left-spring-completed')
    print('left-sigma-completed')

    # ====================== X1 边界处理 ======================
    for i in range(len(Result_X1[0])):
        A0 = Result_X1[2][i]
        R = Result_X1[1][i]
        # 计算弹簧阻尼参数
        k_c_1 = k_c(G,R,Cs,Cp,P)
        Kn,Cn,Kt,Ct = k_c_1[0],k_c_1[1],k_c_1[2],k_c_1[3]
        JUGG = 1  # 法线方向
        name_spring = 'X1-'+str(i+1)
        # 设置人工边界弹簧阻尼
        spring(Result_X1[0][i],Kn,Cn,Kt,Ct,JUGG,name_spring,A0)
        # 计算波延迟位移和速度
        node_h = Result_X1[3][i]
        displacement_result = node_dis_veto(node_h,input[0],H,Cs)
        v_result = node_dis_veto(node_h,input[1],H,Cs)
        u_node = displacement_result[0]
        v_node = v_result[0]
        u_y = v_result[1]
        # 计算节点荷载
        f_y_bx = [[0]*2 for j in range(len(u_node))]
        f_y_by = [[0]*2 for j in range(len(u_node))]
        for j in range(len(u_node)):
            f_y_bx[j][0] = u_node[j][0]
            f_y_bx[j][1] = A0*(Kn*u_node[j][1]+Cn*v_node[j][1])
            f_y_by[j][0] = u_node[j][0]
            f_y_by[j][1] = A0*P*Cs**2*u_y[j][1]
        # 施加节点荷载
        f = 1
        name_ampli_x = 'X1-x-'+str(i+1)
        name_ampli_y = 'X1-y-'+str(i+1)
        JUGG_lord = 1
        lord(Result_X1[0][i],f,JUGG_lord,name_ampli_x,step_name,f_y_bx,name_ampli_x)
        JUGG_lord = 2
        lord(Result_X1[0][i],f,JUGG_lord,name_ampli_y,step_name,f_y_by,name_ampli_y)
    print('right-spring-completed')
    print('right-sigma-completed')

    # ====================== Z0 边界处理 ======================
    for i in range(len(Result_Z0[0])):
        A0 = Result_Z0[2][i]
        R = 4*H  # 固定距离
        # 计算弹簧阻尼参数
        k_c_1 = k_c(G,R,Cs,Cp,P)
        Kn,Cn,Kt,Ct = k_c_1[0],k_c_1[1],k_c_1[2],k_c_1[3]
        JUGG = 2  # 法线方向
        name_spring = 'Z0-'+str(i+1)
        # 设置人工边界弹簧阻尼
        spring(Result_Z0[0][i],Kn,Cn,Kt,Ct,JUGG,name_spring,A0)
        # 读取基础位移和速度
        u_down = input[0]
        v_down = input[1]
        # 计算节点荷载
        f_y_bx = [[0]*2 for j in range(len(u_down)+1)]
        f_y_by = [[0]*2 for j in range(len(u_down)+1)]
        for j in range(len(f_y_bx)-1):
            f_y_bx[j][0] = u_down[j][0]
            f_y_bx[j][1] = A0*(Kt*u_down[j][1]+(Ct+P*Cs)*v_down[j][1])
        # 补充结束点
        f_y_bx[len(f_y_bx)-1][0] = step_time
        f_y_bx[len(f_y_bx)-1][1] = 0
        # 施加节点荷载
        JUGG_lord = 1
        f = 1
        name_ampli_x = 'Z0-x-'+str(i+1)
        lord(Result_Z0[0][i],f,JUGG_lord,name_ampli_x,step_name,f_y_bx,name_ampli_x)
    print('down-spring-completed')
    print('down-sigma-completed')

    # 修正分析步参数
    mdb.models['Model-1'].steps[step_name].setValues(timePeriod=step_time,
                                                     initialInc=0.0001, 
                                                     minInc=6e-05,maxNumInc=10000)
    # 修正场输出频率
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(frequency=1)