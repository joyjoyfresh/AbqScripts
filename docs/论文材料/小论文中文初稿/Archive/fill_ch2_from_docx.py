# 将 小论文中文初稿0902.docx 的第二章内容替换进 小论文中文初稿0902.md 的第二章
# 公式编号按修正后顺延（原docx中式(4)重号），交叉引用同步更新
import io

md_path = r'C:\Users\12462\Documents\Code\AbqScripts\docs\论文材料\小论文中文初稿\小论文中文初稿0902.md'

with io.open(md_path, encoding='utf-8') as f:
    lines = f.read().split('\n')

# 定位第二章边界：## 2 起，## 3 前
ch2_start = next(i for i, l in enumerate(lines) if l.startswith('## 2 '))
ch3_start = next(i for i, l in enumerate(lines) if l.startswith('## 3 '))

new_ch2 = '''## 2 数值模型与结果校核

### 2.1 成层坡地地震动输入方法

#### 2.1.1 黏弹性人工边界

本文采用Abaqus/Standard 2021<sup>[31]</sup>建立二维有限元模型，模型仅截取无限场地中的局部区域。为减小外行波在截断边界处的非物理反射，本文在模型左右两侧及底部采用刘晶波等<sup>[32]</sup>提出的黏弹性人工边界，地表保持自由。按照该方法，边界节点 $l$ 处的法向、切向弹簧刚度及阻尼系数分别为：

$$
\\begin{bmatrix}K_{\\mathrm N}\\\\K_{\\mathrm T}\\end{bmatrix}
=\\frac{1}{2r(1+A)}
\\begin{bmatrix}\\lambda+2G\\\\G\\end{bmatrix}. \\tag{1}
$$

$$
\\begin{bmatrix}C_{\\mathrm N}\\\\C_{\\mathrm T}\\end{bmatrix}
=B\\rho
\\begin{bmatrix}c_{\\mathrm p}\\\\c_{\\mathrm s}\\end{bmatrix}. \\tag{2}
$$

式中，$K_{\\mathrm N}$、$K_{\\mathrm T}$ 和 $C_{\\mathrm N}$、$C_{\\mathrm T}$ 分别为法向、切向弹簧及阻尼参数；$\\lambda$、$G$、$\\rho$、$c_{\\mathrm p}$ 和 $c_{\\mathrm s}$ 按边界节点所在介质取值；$r=\\sqrt{(L/2)^2+(H/2)^2}$ 为计算域包络矩形的半对角线。本文取修正系数 $A=0.8$、$B=1.1$。

#### 2.1.2 成层自由场与等效节点力输入

仅设置吸收边界只能减小外行波在截断边界处的反射，不能引入原无限场地中的斜入射波。参考Bielak等<sup>[33]</sup>关于背景自由场与局部残余场的分解，将有限元总波场写为自由场与散射场之和：

$$
\\mathbf u=\\mathbf u^{\\mathrm f}+\\mathbf u^{\\mathrm s}. \\tag{3}
$$

其中，$\\mathbf u^{\\mathrm f}$ 为不存在坡地局部不规则性时的自由场，$\\mathbf u^{\\mathrm s}$ 为坡地及材料界面引起的散射场。黏弹性人工边界用于吸收向计算域外传播的散射波，而自由场则通过附加等效节点力引入有限元模型。人工边界节点 $l$ 处的等效节点力可写为：

$$
\\mathbf F_l^{\\mathrm{eq}}(t)=A_l\\left[\\mathbf K_l\\mathbf u_l^{\\mathrm f}(t)+\\mathbf C_l\\dot{\\mathbf u}_l^{\\mathrm f}(t)+\\boldsymbol\\sigma_l^{\\mathrm f}(t)\\mathbf n_l\\right]. \\tag{4}
$$

式中，$A_l$ 为二维单位厚度下节点的代表边界长度，$\\mathbf n_l$ 为边界外法向，$\\mathbf K_l$ 和 $\\mathbf C_l$ 为式（1）和式（2）在全局坐标下的参数矩阵；$\\mathbf u_l^{\\mathrm f}$、$\\dot{\\mathbf u}_l^{\\mathrm f}$ 和 $\\boldsymbol\\sigma_l^{\\mathrm f}$ 分别为节点自由场位移、速度和应力。弹簧项和阻尼项补偿人工边界对自由场运动的作用，应力项恢复原无限介质作用于截断面的自由场牵引。因此，地震动输入问题归结为求取各边界节点的自由场位移、速度和应力。

本文参考张季等<sup>[34]</sup>采用的“频域求解成层自由场—生成边界等效节点力”总体流程：由加速度时程作傅里叶变换并换算为入射位移谱，随后逐频求解单位SV波斜入射下的水平成层自由场，恢复各节点的实际自由场，最后组装节点力谱并逆变换至时域。与张季以层间界面位移为未知量组装精确动力刚度矩阵不同，本文直接以覆盖层内上、下行P波和SV波以及基岩中的反射P波、SV波复幅值为未知量：

$$
\\mathbf a(\\omega)=\\begin{bmatrix}A_{\\mathrm{P,c}}^{\\uparrow}&A_{\\mathrm{P,c}}^{\\downarrow}&A_{\\mathrm{SV,c}}^{\\uparrow}&A_{\\mathrm{SV,c}}^{\\downarrow}&A_{\\mathrm{P,b}}^{\\downarrow}&A_{\\mathrm{SV,b}}^{\\downarrow}\\end{bmatrix}^{\\mathrm T}. \\tag{5}
$$

式中，下标c、b分别表示覆盖层和基岩；基岩上行SV波为已知单位入射波，不列入未知向量。这样可在求解后直接恢复任意深度处的位移与应力响应，无需由界面位移进一步重构层内波场，因而更便于计算不同位置人工边界节点的自由场量及等效节点力。

为求解这些未知量，以 $U_x$、$U_y$、$\\Sigma_{yy}$ 和 $\\Sigma_{xy}$ 表示频域位移及应力复幅值，覆盖层—基岩界面 $y=y_{\\mathrm I}$ 满足4个完全黏结条件：

$$
\\begin{cases}
U_x^{\\mathrm c}=U_x^{\\mathrm b},\\\\
U_y^{\\mathrm c}=U_y^{\\mathrm b},\\\\
\\Sigma_{yy}^{\\mathrm c}=\\Sigma_{yy}^{\\mathrm b},\\\\
\\Sigma_{xy}^{\\mathrm c}=\\Sigma_{xy}^{\\mathrm b}.
\\end{cases}\\qquad y=y_{\\mathrm I} \\tag{6}
$$

覆盖层自由表面 $y=y_{\\mathrm s}$ 满足2个零牵引条件：

$$
\\begin{cases}
\\Sigma_{yy}^{\\mathrm c}=0,\\\\
\\Sigma_{xy}^{\\mathrm c}=0.
\\end{cases}\\qquad y=y_{\\mathrm s} \\tag{7}
$$

将各P-SV波分量的位移和应力表达式代入式（6）和式（7），6个边界条件对应6个未知波幅，逐频组装为

$$
\\underset{6\\times6}{\\mathbf A(\\omega)}\\,
\\underset{6\\times1}{\\mathbf a(\\omega)}
=\\underset{6\\times1}{\\mathbf b(\\omega)}. \\tag{8}
$$

式中，$\\mathbf A(\\omega)$ 由材料参数、覆盖层厚度、频率和入射角确定，各波分量的竖向传播关系包含在其系数中；$\\mathbf b(\\omega)$ 为已知单位上行SV波在界面处产生的位移和应力项。自由场按线弹性无阻尼介质求解。逐频求解式（8）即可获得各P-SV波分量的复幅值，进而直接计算任意深度处的单位自由场位移、速度和应力，无需先求界面位移再恢复层内响应。所得复数解已包含界面反射、透射、P-SV波型转换及层内多次传播。

得到频域等效节点力

$$
\\widehat{\\mathbf F}_l^{\\mathrm{eq}}(\\omega)=A_l\\left[\\left(\\mathbf K_l+\\mathrm i\\omega\\mathbf C_l\\right)\\widehat{\\mathbf u}_l^{\\mathrm f}(\\omega)+\\widehat{\\boldsymbol\\sigma}_l^{\\mathrm f}(\\omega)\\mathbf n_l\\right]. \\tag{9}
$$

最后，对节点力谱作逆傅里叶变换，得到各节点水平和竖向等效力时程，并作为集中力施加于Abaqus。左、右及底部边界采用同一上平台水平成层自由场，但均按节点实际坐标、外法向和代表边界长度分别计算。至此，输入时程经“位移谱—单位自由场—节点自由场—节点力谱—节点力时程”完成三侧人工边界输入。

### 2.2 数值模型与研究方案

#### 2.2.1 模型几何、材料及工况

研究对象为SV波斜入射下的二维“上土下岩”坡地，采用平面应变、小变形线弹性假定。SV 波入射角固定为15°。

坡高固定为 $h=100\\ \\mathrm{m}$，坡角记为 $i$。取 $x$ 轴水平向右、$y$ 轴竖直向上，坡顶与坡脚的水平坐标分别为 $x_{\\mathrm c}$ 和 $x_{\\mathrm t}$。上、下平台观测窗分别延伸 $4h$ 和 $3h$，左右两侧各保留 $1h$ 边界净空，坡脚以下保留 $3h$ 基底深度。模型底边取 $y=0$，坡脚和坡顶地表高程分别为 $3h$ 和 $4h$；不同坡角下的地表位置采用式（10）映射至共同坐标。

为把不同坡角下的折线地表映射到共同空间坐标，本文采用三段归一化坐标 $s$。上平台以坡高归一化水平距离，坡面以坡顶至坡脚的沿段进程归一化，下平台继续以坡高为尺度：

$$
s=\\begin{cases}
(x-x_{\\mathrm c})/h, & x\\le x_{\\mathrm c},\\\\[3pt]
(x-x_{\\mathrm c})/(x_{\\mathrm t}-x_{\\mathrm c}), & x_{\\mathrm c}<x<x_{\\mathrm t},\\\\[3pt]
1+(x-x_{\\mathrm t})/h, & x\\ge x_{\\mathrm t}.
\\end{cases} \\tag{10}
$$

观测范围为 $s\\in[-4,4]$，其中 $s=0$、0.5和1分别对应坡顶、坡面中部和坡脚；左右各 $1h$ 的边界净空不纳入统计。模型布置见图1。

> 【图1占位符】二维成层坡地数值模型

基岩剪切波速为 $V_{\\mathrm{s,b}}=2000\\ \\mathrm{m/s}$，密度为 $2500\\ \\mathrm{kg/m^3}$，泊松比为0.30；覆盖层密度固定为基岩的0.85倍，即 $2125\\ \\mathrm{kg/m^3}$，泊松比为0.35，剪切波速由波速比 $s=V_{\\mathrm{s,c}}/V_{\\mathrm{s,b}}$ 控制。材料的剪切模量、弹性模量和压缩波速均由波速、密度与泊松比一致换算，而非彼此独立指定。

材料、几何与输入参数见表1。L组由坡角 $i$、厚度比 $d/h$ 和波速比 $r_v$ 的 $4\\times4\\times4$ 全因子组合构成，共64例；H组含4个同坡角均质基岩坡。坡高、密度比、入射角及输入信号均保持不变。

表1 工况参数表

| 参数 | H 组（均质坡，4 例） | L 组（成层坡，64 例） |
|:---:|:---:|:---:|
| 坡角 $i$ (°) | 15、30、45、60 | 15、30、45、60 |
| 覆盖层厚度 $h_{\\mathrm c}$ (m) | — | 20、60、100、140 |
| 覆盖层厚度比 $d=h_{\\mathrm c}/h$ | — | 0.2、0.6、1.0、1.4 |
| 覆盖层剪切波速 $V_{\\mathrm{s,c}}$ (m/s) | — | 600、900、1200、1500 |
| 覆盖层波速比 $s=V_{\\mathrm{s,c}}/V_{\\mathrm{s,b}}$ | — | 0.3、0.45、0.6、0.75 |

为便于表述与检索，工况编号采用“组别字母—参数串”规则命名。组别字母标明工况类别：H、L分别为均质坡（homogeneous）、成层坡（layered）。参数串中的字母 $i$、$h$、$a$、$d$、$s$ 依次表示坡角、坡高、SV 波入射角、厚度比 $h_{\\mathrm c}/h$ 与波速比 $V_{\\mathrm{s,c}}/V_{\\mathrm{s,b}}$。均质坡 H 组标注 $i$、$h$、$a$；成层坡 L 组及 T、V 组标注 $i$、$d$、$s$（坡高与入射角为全体工况常量，仅在 H 组标出）。例如，H-i15h100a15 表示坡角 15°、坡高 100 m、入射角 15° 的均质坡；L-i60d1.4s0.3 表示坡角 60°、厚度比 1.4、波速比 0.3 的成层坡。

#### 2.2.2 宽频输入与地表响应识别方法

G1b在0.5～10 Hz内的归一化谱幅中位数为0.921，5%分位数为0.533，在分析频带内未出现连续的近零谱谷。因此，该频带可用于复数相除及相位分析。计算时先检查频点有效性，再进行复数相除、相位展开和群时延求导，以减小分母弱谱点对结果的影响。

> 【图3占位符】P061宽频系统识别与复频响提取：（a）宽频多正弦输入时程；（b）输入能量覆盖识别频带；（c）复比值幅值；（d）复比值保留相位信息

为了比较不同地表区段和频带的响应，将上平台A段、坡面B段和下平台C段分别定义为 $-4\\le s\\le 0$、$0<s\\le 1$ 和 $1<s\\le 4$，低、中、高频分别取 $[0.5,3)$、$[3,6)$ 和 $[6,10]$ Hz。

### 2.3 数值结果校核

采用SPECFEM2D<sup>[35]</sup>计算60°均质坡H004和同坡角软厚成层坡P061，输入统一采用4 Hz Ricker脉冲。图2给出了Abaqus与SPECFEM2D计算的地表水平、竖向PGA空间曲线，归一化均方根误差（NRMSE）和峰位差见表2。

> 【图2占位符】Abaqus与SPECFEM2D地表PGA对比：（a）X001：H004水平分量；（b）X001：H004竖向分量；（c）X002：P061水平分量；（d）X002：P061竖向分量

结果显示，两套软件计算的地表PGA空间曲线总体吻合：均质坡H004的水平与竖向PGA曲线NRMSE分别为5.6%和7.3%，软厚成层坡P061分别为3.6%和10.2%；PGA峰值位置的归一化坐标偏差不超过0.07。Abaqus与SPECFEM2D在单元离散、波动输入与边界实现上相互独立，上述量级的偏差主要来自网格离散与散射波细节的差异，不改变坡顶放大、坡面振荡与下平台干涉等主要空间特征。据此认为，本文模型的黏弹性人工边界与等效节点力输入可为后续宽频复频响分析提供可靠的数值基础。'''

# 参考文献补 [35]（2.3节引用SPECFEM2D，md原参考文献止于[34]）
ref34 = '[34] 张季, 谭灿星, 叶国涛, 等. SV波超临界角斜入射时层状地基地震动输入在ABAQUS中的实现[J]. 工程力学, 2021, 38(4): 200-210.'
ref35 = '\n[35] SPECFEM2D Development Team. SPECFEM2D Cartesian user guide (v8.1.0)[CP/OL]. [2026-08-24]. https://github.com/geodynamics/specfem2d.'
content = '\n'.join(lines[:ch2_start]) + '\n' + new_ch2 + '\n\n' + '\n'.join(lines[ch3_start:])
if '[35]' not in content:
    content = content.replace(ref34, ref34 + ref35, 1)

with io.open(md_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('替换完成')
print('第二章起始于原行', ch2_start + 1, '第三章起始于原行', ch3_start + 1)
