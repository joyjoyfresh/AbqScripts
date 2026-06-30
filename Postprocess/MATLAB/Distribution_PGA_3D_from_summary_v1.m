%% 读取 Python 汇总 CSV 并绘制 3D PGA_h 最大值分布（总图 + 单图）  % 定义脚本功能说明
clear;  % 清理工作区变量
clc;  % 清空命令行窗口

scriptDir = fileparts(mfilename('fullpath'));  % 获取当前脚本所在目录
defaultCsvPath = fullfile(scriptDir, 'PGA_h_max_summary.csv');  % 设置默认汇总 CSV 路径
outputGridPath = fullfile(scriptDir, 'PGA_h_max_3D_grid_matlab.png');  % 设置 MATLAB 总图输出路径
outputSinglePrefix = 'PGA_h_max_3D';  % 设置单图输出文件名前缀
targetAngles = [0, 30];  % 设置目标入射角顺序
faceColor = [236, 236, 236] ./ 255;  % 设置背景色
surfaceAlpha = 0.72;  % 设置曲面透明度
markerColor = [1.0, 0.102, 0.102];  % 设置标记点颜色
markerEdgeColor = [0.0, 0.0, 0.0];  % 设置标记点边线颜色

if ~isfile(defaultCsvPath)  % 判断默认汇总 CSV 是否存在
    error('未找到汇总 CSV：%s', defaultCsvPath);  % 抛出缺少文件异常
end  % 结束文件存在性判断

tbl = readtable(defaultCsvPath, 'TextType', 'string');  % 读取汇总 CSV 为表格
requiredCols = ["motion", "motion_display", "marker", "h", "i", "angle", "pga_max"];  % 定义必须字段列表
for k = 1:numel(requiredCols)  % 遍历检查每个必需字段
    if ~ismember(requiredCols(k), string(tbl.Properties.VariableNames))  % 判断当前字段是否存在
        error('CSV 缺少字段：%s', requiredCols(k));  % 抛出缺少字段异常
    end  % 结束字段存在性判断
end  % 结束字段检查循环

motionKeys = ["el_centro", "loma_prieta", "northridge"];  % 定义地震动顺序键
motionDisplay = ["El Centro", "Loma Prieta", "Northridge"];  % 定义地震动显示名称
motionMarkers = ["o", "^", "s"];  % 定义地震动标记符号

figGrid = figure('Color', faceColor, 'Units', 'pixels', 'Position', [80, 60, 1150, 980]);  % 创建总图画布
tiledlayout(figGrid, 3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');  % 创建 3x2 紧凑布局

for r = 1:numel(motionKeys)  % 按地震动遍历每一行
    rowMask = tbl.motion == motionKeys(r);  % 生成当前地震动行掩码
    rowValues = tbl.pga_max(rowMask);  % 提取当前地震动的全部 z 值
    rowValues = rowValues(isfinite(rowValues));  % 过滤非有限值
    zLimRow = localComputeZLim(rowValues);  % 计算当前行统一 z 轴范围
    for c = 1:numel(targetAngles)  % 按入射角遍历每一列
        nexttile;  % 切换到当前子图
        ax = gca;  % 获取当前坐标轴
        hold(ax, 'on');  % 打开当前坐标轴叠加绘图
        panelMask = (tbl.motion == motionKeys(r)) & (abs(tbl.angle - targetAngles(c)) <= 1e-6);  % 生成当前子图筛选掩码
        panelTbl = tbl(panelMask, :);  % 提取当前子图数据
        localDrawPanel(ax, panelTbl, motionMarkers(r), markerColor, markerEdgeColor, surfaceAlpha);  % 绘制当前子图曲面与标记
        localStyleAxes(ax, targetAngles(c), panelTbl.h, panelTbl.i, zLimRow, faceColor);  % 设置当前子图坐标轴样式
        if isempty(panelTbl)  % 判断当前子图是否为空
            text(ax, 0.35, 0.50, 0.50, 'No Data', 'Units', 'normalized', 'Color', [0.25, 0.25, 0.25], 'FontSize', 11);  % 写入无数据提示
        end  % 结束空数据判断
    end  % 结束列循环
end  % 结束行循环

lgdMarkers = gobjects(1, numel(motionKeys));  % 初始化图例句柄数组
for r = 1:numel(motionKeys)  % 遍历地震动生成图例句柄
    lgdMarkers(r) = plot3(nan, nan, nan, motionMarkers(r), 'LineStyle', 'none', 'MarkerSize', 6, 'MarkerFaceColor', markerColor, 'MarkerEdgeColor', markerEdgeColor, 'DisplayName', motionDisplay(r));  % 构造虚拟图例点
end  % 结束图例句柄创建循环
lgd = legend(lgdMarkers, motionDisplay, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'on');  % 创建底部共享图例
set(lgd, 'FontSize', 11);  % 设置图例字体大小
exportgraphics(figGrid, outputGridPath, 'Resolution', 350);  % 导出总图为 PNG
close(figGrid);  % 关闭总图对象释放内存

singleCount = 0;  % 初始化单图计数器
for r = 1:numel(motionKeys)  % 按地震动遍历单图行
    rowMask = tbl.motion == motionKeys(r);  % 生成当前地震动行掩码
    rowValues = tbl.pga_max(rowMask);  % 提取当前地震动的全部 z 值
    rowValues = rowValues(isfinite(rowValues));  % 过滤非有限值
    zLimRow = localComputeZLim(rowValues);  % 计算当前行统一 z 轴范围
    for c = 1:numel(targetAngles)  % 按入射角遍历单图列
        figSingle = figure('Color', faceColor, 'Units', 'pixels', 'Position', [180, 120, 760, 560]);  % 创建单图画布
        ax = axes(figSingle);  % 创建单图坐标轴
        hold(ax, 'on');  % 打开单图坐标轴叠加绘图
        panelMask = (tbl.motion == motionKeys(r)) & (abs(tbl.angle - targetAngles(c)) <= 1e-6);  % 生成当前单图筛选掩码
        panelTbl = tbl(panelMask, :);  % 提取当前单图数据
        localDrawPanel(ax, panelTbl, motionMarkers(r), markerColor, markerEdgeColor, surfaceAlpha);  % 绘制当前单图曲面与标记
        localStyleAxes(ax, targetAngles(c), panelTbl.h, panelTbl.i, zLimRow, faceColor);  % 设置当前单图坐标轴样式
        if isempty(panelTbl)  % 判断当前单图是否为空
            text(ax, 0.38, 0.52, 0.50, 'No Data', 'Units', 'normalized', 'Color', [0.25, 0.25, 0.25], 'FontSize', 11);  % 写入无数据提示
        end  % 结束空数据判断
        hLegendPoint = plot3(ax, nan, nan, nan, motionMarkers(r), 'LineStyle', 'none', 'MarkerSize', 6, 'MarkerFaceColor', markerColor, 'MarkerEdgeColor', markerEdgeColor, 'DisplayName', motionDisplay(r));  % 创建单图图例虚拟句柄
        lgdSingle = legend(ax, hLegendPoint, motionDisplay(r), 'Location', 'southoutside', 'NumColumns', 1, 'Box', 'on');  % 创建单图底部图例
        set(lgdSingle, 'FontSize', 10);  % 设置单图图例字体大小
        angleText = localFormatNumber(targetAngles(c));  % 格式化角度文本用于命名
        singleName = sprintf('%s-%s-angle%s-matlab.png', outputSinglePrefix, motionKeys(r), angleText);  % 组装单图文件名
        singlePath = fullfile(scriptDir, singleName);  % 组装单图输出完整路径
        exportgraphics(figSingle, singlePath, 'Resolution', 350);  % 导出单图为 PNG
        close(figSingle);  % 关闭单图对象释放内存
        singleCount = singleCount + 1;  % 累加单图数量
        fprintf('单图输出: %s\n', singlePath);  % 输出当前单图路径
    end  % 结束单图列循环
end  % 结束单图行循环

fprintf('汇总 CSV: %s\n', defaultCsvPath);  % 输出输入 CSV 路径
fprintf('总图输出: %s\n', outputGridPath);  % 输出总图路径
fprintf('单图数量: %d\n', singleCount);  % 输出单图数量

function localDrawPanel(ax, panelTbl, markerSymbol, markerColor, markerEdgeColor, surfaceAlpha)  % 定义子图绘制函数
if isempty(panelTbl)  % 判断当前子图是否为空
    return;  % 直接返回不绘制
end  % 结束空数据判断
x = panelTbl.h;  % 读取 h 列为 x 数据
y = panelTbl.i;  % 读取 i 列为 y 数据
z = panelTbl.pga_max;  % 读取 pga_max 列为 z 数据
validMask = isfinite(x) & isfinite(y) & isfinite(z);  % 构造有效数据掩码
x = x(validMask);  % 过滤后的 x 数据
y = y(validMask);  % 过滤后的 y 数据
z = z(validMask);  % 过滤后的 z 数据
if numel(x) == 0  % 判断过滤后是否无有效点
    return;  % 直接返回不绘制
end  % 结束有效点判断
if numel(x) >= 3  % 判断点数是否足够生成三角曲面
    tri = delaunay(x, y);  % 计算二维三角剖分
    hSurf = trisurf(tri, x, y, z, 'Parent', ax);  % 绘制三角曲面
    set(hSurf, 'FaceColor', 'interp', 'EdgeColor', 'k', 'LineWidth', 0.7, 'FaceAlpha', surfaceAlpha);  % 设置曲面样式
    colormap(ax, flipud(bwrmap(256)));  % 设置蓝白红色图映射
end  % 结束曲面绘制判断
scatter3(ax, x, y, z, 24, 'Marker', markerSymbol, 'MarkerFaceColor', markerColor, 'MarkerEdgeColor', markerEdgeColor, 'LineWidth', 0.35);  % 绘制红色观测点
end  % 结束子图绘制函数

function localStyleAxes(ax, angleValue, hValues, iValues, zLimRow, faceColor)  % 定义坐标轴样式函数
title(ax, sprintf('\\theta_s = %s^\\circ', localFormatNumber(angleValue)), 'FontSize', 11);  % 设置子图标题
xlabel(ax, 'h (m)', 'FontSize', 10);  % 设置 x 轴标签
ylabel(ax, 'i (deg)', 'FontSize', 10);  % 设置 y 轴标签
zlabel(ax, 'PGA_{h,max} (g)', 'FontSize', 10);  % 设置 z 轴标签
if isempty(hValues)  % 判断 h 数据是否为空
    xLim = [0, 1];  % 使用兜底 x 轴范围
else  % 当前 h 数据非空
    xMin = min([0; hValues]);  % 计算 x 轴下界并尽量包含 0
    xMax = max(hValues);  % 计算 x 轴上界
    if abs(xMax - xMin) < 1e-9  % 判断 x 轴范围是否退化
        xMin = xMin - 1;  % 扩展 x 轴下界
        xMax = xMax + 1;  % 扩展 x 轴上界
    end  % 结束 x 轴退化判断
    xLim = [xMin, xMax];  % 组合 x 轴范围
end  % 结束 h 数据空值判断
if isempty(iValues)  % 判断 i 数据是否为空
    yLim = [0, 1];  % 使用兜底 y 轴范围
else  % 当前 i 数据非空
    yMin = min(iValues);  % 计算 y 轴下界
    yMax = max(iValues);  % 计算 y 轴上界
    if abs(yMax - yMin) < 1e-9  % 判断 y 轴范围是否退化
        yMin = yMin - 1;  % 扩展 y 轴下界
        yMax = yMax + 1;  % 扩展 y 轴上界
    end  % 结束 y 轴退化判断
    yLim = [yMin, yMax];  % 组合 y 轴范围
end  % 结束 i 数据空值判断
xlim(ax, xLim);  % 设置 x 轴范围
ylim(ax, yLim);  % 设置 y 轴范围
set(ax, 'YDir', 'reverse');  % 设置 y 轴方向反转
zlim(ax, zLimRow);  % 设置 z 轴范围
xticks(ax, localChooseTicks(hValues, 3));  % 设置 x 轴刻度
yticks(ax, localChooseTicks(iValues, 3));  % 设置 y 轴刻度
zticks(ax, linspace(zLimRow(1), zLimRow(2), 3));  % 设置 z 轴三段刻度
view(ax, -126, 23);  % 设置视角方位角与俯仰角
grid(ax, 'on');  % 打开网格
ax.GridAlpha = 0.65;  % 设置网格透明度
ax.GridColor = [0.737, 0.737, 0.737];  % 设置网格颜色
ax.LineWidth = 0.8;  % 设置坐标轴线宽
ax.Color = faceColor;  % 设置子图背景色
pbaspect(ax, [1.45, 1.05, 0.38]);  % 设置三维纵横比
end  % 结束坐标轴样式函数

function zLimRow = localComputeZLim(values)  % 定义 z 轴范围计算函数
if isempty(values)  % 判断输入值是否为空
    zLimRow = [0, 1];  % 返回默认 z 轴范围
    return;  % 结束函数
end  % 结束空值判断
zMin = min(values);  % 计算最小值
zMax = max(values);  % 计算最大值
if abs(zMax - zMin) < 1e-9  % 判断 z 轴范围是否过窄
    zMin = max(0, zMin - 0.1);  % 对常值情况向下扩展
    zMax = zMax + 0.1;  % 对常值情况向上扩展
else  % 当前范围正常
    span = zMax - zMin;  % 计算跨度
    zMin = max(0, zMin - 0.12 * span);  % 按比例扩展下界
    zMax = zMax + 0.20 * span;  % 按比例扩展上界
end  % 结束范围判断
zLimRow = [zMin, zMax];  % 组合 z 轴范围
end  % 结束 z 轴范围计算函数

function ticksOut = localChooseTicks(values, maxTickCount)  % 定义刻度抽样函数
if isempty(values)  % 判断输入是否为空
    ticksOut = [];  % 返回空刻度
    return;  % 结束函数
end  % 结束空值判断
u = unique(values(:));  % 获取去重值
u = sort(u);  % 对去重值升序排序
if numel(u) <= maxTickCount  % 判断唯一值数量是否不超过上限
    ticksOut = u(:)';  % 直接返回全部值作为刻度
    return;  % 结束函数
end  % 结束数量判断
idx = round(linspace(1, numel(u), maxTickCount));  % 按索引均匀抽样
idx = unique(idx);  % 去重抽样索引
ticksOut = u(idx);  % 返回抽样后刻度
ticksOut = ticksOut(:)';  % 统一输出为行向量
end  % 结束刻度抽样函数

function txt = localFormatNumber(v)  % 定义数字格式化函数
if abs(v - round(v)) < 1e-12  % 判断是否为整数值
    txt = sprintf('%d', round(v));  % 按整数格式输出文本
else  % 当前为非整数值
    txt = sprintf('%g', v);  % 按紧凑浮点格式输出文本
end  % 结束整数判断
end  % 结束数字格式化函数

function cmap = bwrmap(n)  % 定义蓝白红色图函数
if nargin < 1  % 判断是否传入色阶数量
    n = 256;  % 设置默认色阶数量
end  % 结束参数判断
half = floor(n / 2);  % 计算前半段长度
r1 = linspace(0, 1, half)';  % 生成前半段红色通道
g1 = linspace(0, 1, half)';  % 生成前半段绿色通道
b1 = ones(half, 1);  % 生成前半段蓝色通道
r2 = ones(n - half, 1);  % 生成后半段红色通道
g2 = linspace(1, 0, n - half)';  % 生成后半段绿色通道
b2 = linspace(1, 0, n - half)';  % 生成后半段蓝色通道
cmap = [r1, g1, b1; r2, g2, b2];  % 拼接得到蓝白红色图
end  % 结束蓝白红色图函数
