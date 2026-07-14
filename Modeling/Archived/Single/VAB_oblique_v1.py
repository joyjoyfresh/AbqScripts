# -*- coding: utf-8 -*-
"""
Abaqus 斜向冲击/加载分析脚本
VAB_oblique - Virtual Abaqus Bow / Oblique loading analysis
"""
from abaqus import *
import abaqusConstants as C
from abaqusConstants import ON
from caeModules import *


def VAB_oblique(angle, cs, vv, density):
    """
    斜向加载/冲击分析主函数

    参数:
        angle (float): 斜向加载角度（度），0为垂直，90为水平
        cs (float): 压缩强度（MPa）
        vv (float): 泊松比
        density (float): 材料密度（kg/m³）
    """
    # ============ 1. 创建模型 ============
    try:
        myModel = mdb.models['Model-1']
    except:
        myModel = mdb.Model(name='Model-1')

    # ============ 2. 定义材料属性 ============
    myMaterial = myModel.Material(name='Concrete')

    # 弹性属性（基于泊松比和压缩强度估算弹性模量）
    # 混凝土经验公式：E ≈ 4700 * √fc' (fc'为压缩强度MPa)
    elastic_modulus = 4700 * (cs ** 0.5) * 1e6  # 转换为Pa

    myMaterial.Elastic(
        table=((elastic_modulus, vv),)
    )

    myMaterial.Density(
        table=((density,),)
    )

    # 塑性属性（简化模型，可根据需要调整）
    myMaterial.Concrete(
        table=(
            (0.0, 0.0),                      # 初始点
            (0.4 * cs / elastic_modulus, 0.5),  # 开裂应变
            (cs / elastic_modulus, 1.0),       # 峰值应力
        ),
        table1=(  # 拉伸软化
            (0.0001, 0.6),
            (0.0002, 0.2),
            (0.0003, 0.1),
        ),
        TYPE=C.STRAIN
    )

    # ============ 3. 创建部件 ============
    # 创建长方体（可根据实际几何修改）
    length = 2.0  # 长度
    width = 0.5   # 宽度
    height = 0.5  # 高度

    mySketch = myModel.ConstrainedSketch(
        name='sketch',
        sheetSize=5.0
    )

    mySketch.rectangle(
        point1=(0.0, 0.0),
        point2=(length, height)
    )

    myPart = myModel.Part(
        name='Part-1',
        dimensionality=C.THREE_D,
        type=C.DEFORMABLE_BODY
    )

    myPart.BaseSolidExtrude(
        sketch=mySketch,
        depth=width
    )

    # ============ 4. 创建截面并赋给部件 ============
    mySection = myModel.HomogeneousSolidSection(
        name='Section-1',
        material='Concrete',
        thickness=None
    )

    region = myPart.Set(
        name='Set-1',
        cells=myPart.cells
    )

    myPart.SectionAssignment(
        region=region,
        sectionName='Section-1'
    )

    # ============ 5. 创建装配 ============
    myAssembly = myModel.rootAssembly
    myAssembly.Instance(name='Part-1-1', part=myPart, dependent=ON)

    # ============ 6. 创建分析步 ============
    myModel.StaticStep(
        name='Step-1',
        previous='Initial',
        description='斜向加载分析步',
        timePeriod=1.0,
        maxNumInc=1000,
        initialInc=0.01,
        minInc=1e-05,
        maxInc=0.1
    )

    # ============ 7. 创建载荷 ============
    # 计算斜向载荷分量
    angle_rad = angle * 3.14159265 / 180.0
    total_force = 100000  # 总力（可根据需要调整）
    fx = total_force * (angle_rad)  # x方向分量
    fy = total_force * (1 - angle_rad)  # y方向分量

    # 定义载荷区域（右端面）
    right_face = myPart.instances['Part-1-1'].faces.findAt(
        ((length, 0, 0),)
    )
    region = myAssembly.Surface(
        side1Faces=right_face,
        name='LoadSurface'
    )

    # 应用斜向载荷
    myModel.Pressure(
        name='Load-1',
        createStepName='Step-1',
        region=region,
        distributionType=C.UNIFORM,
        magnitude=total_force / (height * width),
        field='',
        amplitude=C.UNSET
    )

    # ============ 8. 创建边界条件 ============
    # 左端固定
    left_face = myPart.instances['Part-1-1'].faces.findAt(
        ((0, 0, 0),)
    )
    region = myAssembly.Set(
        faces=left_face,
        name='FixedSet'
    )

    myModel.EncastreBC(
        name='BC-1',
        createStepName='Initial',
        region=region,
        localCsys=None
    )

    # ============ 9. 网格划分 ============
    myPart.seedPart(size=0.05, deviationFactor=0.1)
    myPart.generateMesh()

    # ============ 10. 输出请求 ============
    myModel.FieldOutputRequest(
        name='F-Output-1',
        createStepName='Step-1',
        variables=('S', 'E', 'PE', 'PEEQ', 'U', 'RF'),
        frequency=1
    )

    # ============ 11. 创建并提交作业 ============
    try:
        job = mdb.Job(
            name='VAB_oblique',
            model='Model-1',
            description=f'S斜向加载分析 - 角度:{angle}°, 强度:{cs}MPa'
        )
        job.submit()
        job.waitForCompletion()
        print('分析完成！')
    except:
        print('作业提交失败，请检查设置')


if __name__ == '__main__':
    # 测试参数（与原脚本一致）
    angle = 15
    cs = 1754
    vv = 0.3
    density = 2500.0

    VAB_oblique(angle, cs, vv, density)
