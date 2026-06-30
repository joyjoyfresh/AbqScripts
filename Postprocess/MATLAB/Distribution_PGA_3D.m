% Distribution_PGA_3D.m
% 一款 MATLAB 脚本，用于重现之前 Python 版本中生成的炫酷 3D 拟合图效果
% 本版本直接读取 Python 生成的 PGA_max_summary.csv 文件进行出图，删除了所有冗余步骤。

clc; clear; close all;

%% 1. 读取环境与汇总数据
% 请确保 PGA_max_summary.csv 与此脚本在同级目录（或指定正确绝对路径）
summaryFile = 'PGA_max_summary.csv'; 

if ~exist(summaryFile, 'file')
    error('未找到数据汇总文件 %s ，请先运行 Python 脚本生成该 CSV 数据文件。', summaryFile);
end

fprintf('正在读取数据源: %s\n', summaryFile);
opts = detectImportOptions(summaryFile);
TG = readtable(summaryFile, opts);

targetAngles = [0, 30];
motionsList = {'elcentro', 'lomaprieta', 'northridge'};
motionDisplays = {'El Centro', 'Loma Prieta', 'Northridge'};
markersList = {'o', '^', 's'};

%% 2. 生成 3D 图组
fig = figure('Name', 'PGA MAX 3D - Matlab Render', 'Color', 'w', 'Position', [100, 50, 1080, 960]);
tLayout = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
plotIdx = 1;

for r = 1:length(motionsList)
    mKey = motionsList{r};
    
    % 以 motion 作为匹配项提取包含该工况的地表数据
    % 兼容 string 及 cell array 格式读取
    rowData = TG(strcmp(string(TG.motion), string(mKey)), :);
    if isempty(rowData), continue; end
    
    % 行统一度量：计算统一的最顶刻度 (向 0.5 递进看齐)
    maxPgaAll = max(rowData.pga_max);
    zUpper = ceil(maxPgaAll * 2.0) / 2.0;
    if zUpper - maxPgaAll < 0.05
        zUpper = zUpper + 0.5;
    end
    zLimits = [0, zUpper];
    
    for c = 1:length(targetAngles)
        aCurr = targetAngles(c);
        panData = rowData(rowData.angle == aCurr, :);
        
        ax = nexttile(tLayout);
        hold(ax, 'on');
        
        if ~isempty(panData)
            X = panData.h;
            Y = panData.i;
            Z = panData.pga_max;
            
            % 绘制带灰外线的插值三角曲面
            if length(X) >= 3
                tri = delaunay(X, Y);
                trisurf(tri, X, Y, Z, 'Parent', ax, ...
                    'FaceColor', 'interp', 'EdgeColor', '#4B4B4B', ...
                    'LineWidth', 0.5, 'FaceAlpha', 0.95);
            end
            
            % 叠加突出的红色点位
            scatter3(ax, X, Y, Z, 50, 'red', markersList{r}, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
            
            % 设置底轴边界与智能刻度约束
            [minX, maxX] = bounds(X);  if minX == maxX, minX=minX-1; maxX=maxX+1; end
            [minY, maxY] = bounds(Y);  if minY == maxY, minY=minY-1; maxY=maxY+1; end
            
            xlim(ax, [min(0, minX), maxX]);
            ylim(ax, [minY, maxY]);
            
            xticks(ax, choose_ticks(X, 3));
            yticks(ax, choose_ticks(Y, 3));
        end
        
        % 匹配高度刻度
        zlim(ax, zLimits);
        zTicksArr = 0.5:0.5:zLimits(2);
        zticks(ax, zTicksArr);
        
        % 生成去包含多余 .00 的字符串标记格式
        zTickLabs = arrayfun(@(v) sprintf('%g', v), zTicksArr, 'UniformOutput', false);
        zticklabels(ax, zTickLabs);
        
        % 设置三维透视轴向翻转机制（匹配 Python 版本视角）
        set(ax, 'YDir', 'reverse');
        view(ax, -128, 28);
        grid(ax, 'on');
        
        % 设置背景透射性为无边框白
        set(ax, 'Color', 'w', 'GridAlpha', 0.35, 'FontName', 'Times New Roman', 'FontSize', 11);
        ax.BoxStyle = 'full';
        
        % 设置空间盒子宽高比
        ax.PlotBoxAspectRatio = [1.4, 1.1, 0.45];
        
        % 轴标签和标题使用优雅的 LaTeX 公式字体
        xlabel(ax, '$$h(\mathrm{m})$$', 'Interpreter', 'latex', 'FontSize', 12);
        ylabel(ax, '$$i(^\circ)$$', 'Interpreter', 'latex', 'FontSize', 12);
        zlabel(ax, '$$PGA_{\max}\,(\mathrm{g})$$', 'Interpreter', 'latex', 'FontSize', 12);
        title(ax, sprintf('$$\\theta_s = %g^\\circ$$', aCurr), 'Interpreter', 'latex', 'FontSize', 15);
        
        % 上渐变配色应用蓝-白-红
        colormap(ax, bwr_colormap());
    end
end

%% 3. 跨框共享图例
hLines = gobjects(1, 3);
% 借助图一添加不可见散点用于全局图例提取
axLeg = nexttile(tLayout, 1);
hold(axLeg, 'on');
for i=1:length(motionsList)
    hLines(i) = scatter(axLeg, nan, nan, 65, char(markersList{i}), 'red', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.6);
end

% 配置放置在TiledLayout最底端的平铺图例并应用样式
leg = legend(axLeg, hLines, motionDisplays, 'Orientation', 'horizontal', 'NumColumns', 3);
leg.Layout.Tile = 'south';
leg.FontName = 'Times New Roman';
leg.FontSize = 13;
leg.Interpreter = 'latex';

%% 4. 保存高质量图片
outFile = 'PGA_max_3D_grid_matlab.png';
exportgraphics(fig, outFile, 'Resolution', 400); % 无边缘瑕疵的高清原图
fprintf('成功生成精美图片并存放在当下路径: %s\n', outFile);

%% ============= 局部辅助函数 =============

function ticks = choose_ticks(vals, n)
    % 从序列中均匀抽取不超过 n 的准确定位点构成刻度轴
    uVals = unique(vals);
    if length(uVals) <= n
        ticks = uVals;
    else
        idx = round(linspace(1, length(uVals), n));
        ticks = uVals(idx);
    end
end

function cmap = bwr_colormap()
    % 定制的红白蓝双色流平滑调色板
    n = 256;
    nHalf = n/2;
    % 从纯蓝色平滑过渡到纯白色
    r1 = linspace(0, 1, nHalf)';
    g1 = linspace(0, 1, nHalf)';
    b1 = ones(nHalf, 1);
    
    % 从纯白色平滑过渡到亮红色
    r2 = ones(nHalf, 1);
    g2 = linspace(1, 0, nHalf)';
    b2 = linspace(1, 0, nHalf)';
    
    cmap = [r1 g1 b1; r2 g2 b2];
end
