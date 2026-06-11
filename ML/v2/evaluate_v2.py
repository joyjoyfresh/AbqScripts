import os  # 导入操作系统路径处理模块
import json  # 导入 JSON 读写模块
import numpy as np  # 导入数值计算库
import matplotlib.pyplot as plt  # 导入绘图库
from sklearn.metrics import r2_score  # 导入 R² 评价函数

from config_v2 import OUTPUT_DIR, TARGET_CHANNELS  # 导入输出目录和目标通道配置

# 设置绘图风格，使图像更统一美观
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')  # 选择白网格风格或默认风格
plt.rcParams.update({  # 更新全局绘图参数
    'font.size': 11,  # 设置默认字号
    'axes.labelsize': 12,  # 设置坐标轴标签字号
    'axes.titlesize': 13,  # 设置子图标题字号
    'xtick.labelsize': 10,  # 设置横轴刻度字号
    'ytick.labelsize': 10,  # 设置纵轴刻度字号
    'figure.titlesize': 15,  # 设置整图标题字号
    'figure.facecolor': '#fafafa',  # 设置画布背景色
    'axes.facecolor': '#ffffff'  # 设置坐标轴背景色
})  # 结束全局参数更新

def plot_model_comparison(summary):  # 定义模型对比绘图函数
    """
    1. 模型对比：比较 4 种架构在 4 个目标上的 R² 和 MAE
    """
    models = list(summary.keys())  # 获取所有模型名称
    channels = TARGET_CHANNELS  # 获取目标通道列表
    
    # 1.1 R2 柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 创建一行两列子图
    
    x = np.arange(len(channels))  # 生成通道横坐标
    width = 0.18  # 设置柱状图宽度
    
    for idx, model in enumerate(models):  # 遍历每个模型
        r2_vals = [summary[model][f"{ch}_R2"] for ch in channels]  # 提取该模型各通道 R²
        axes[0].bar(x + (idx - len(models)/2 + 0.5) * width, r2_vals, width, label=model.upper())  # 绘制 R² 柱状图
        
    axes[0].set_title("整体 OOF R² 得分对比")  # 设置左图标题
    axes[0].set_xticks(x)  # 设置左图横坐标刻度位置
    axes[0].set_xticklabels(channels)  # 设置左图横坐标标签
    axes[0].set_ylabel("R² 得分")  # 设置左图纵坐标标签
    axes[0].set_ylim(0.0, 1.05)  # 设置左图纵轴范围
    axes[0].legend(frameon=True, facecolor='white')  # 显示左图图例
    
    # 1.2 MAE 柱状图
    for idx, model in enumerate(models):  # 遍历每个模型
        mae_vals = [summary[model][f"{ch}_MAE"] for ch in channels]  # 提取该模型各通道 MAE
        # 如有需要可按目标均值归一化 MAE 以便对比，这里直接绘制绝对值
        axes[1].bar(x + (idx - len(models)/2 + 0.5) * width, mae_vals, width, label=model.upper())  # 绘制 MAE 柱状图
        
    axes[1].set_title("整体 OOF MAE 对比（物理量纲）")  # 设置右图标题
    axes[1].set_xticks(x)  # 设置右图横坐标刻度位置
    axes[1].set_xticklabels(channels)  # 设置右图横坐标标签
    axes[1].set_ylabel("平均绝对误差（MAE）")  # 设置右图纵坐标标签
    axes[1].legend(frameon=True, facecolor='white')  # 显示右图图例
    
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(os.path.join(OUTPUT_DIR, "1_model_comparison.png"), dpi=200, bbox_inches='tight')  # 保存模型对比图
    plt.close()  # 关闭当前图像

def plot_training_curves(models):  # 定义训练曲线绘图函数
    """
    2. 训练曲线：绘制所有模型第 0 折的损失-轮次曲线
    """
    plt.figure(figsize=(10, 6))  # 创建训练曲线画布
    
    colors = {'cnn': '#1f77b4', 'lstm': '#ff7f0e', 'transformer': '#2ca02c', 'deeponet': '#d62728'}  # 定义模型颜色映射
    
    for model in models:  # 遍历所有模型
        hist_path = os.path.join(OUTPUT_DIR, f"{model}_history.json")  # 拼接历史文件路径
        if not os.path.exists(hist_path):  # 检查历史文件是否存在
            continue  # 如果不存在就跳过
        with open(hist_path, 'r') as f:  # 以只读方式打开历史文件
            hist = json.load(f)  # 读取训练历史数据
            
        epochs = range(1, len(hist['train_loss']) + 1)  # 生成轮次序列
        plt.plot(epochs, hist['train_loss'], label=f"{model.upper()} 训练", color=colors[model], linestyle='--')  # 绘制训练损失曲线
        plt.plot(epochs, hist['val_loss'], label=f"{model.upper()} 验证", color=colors[model], linewidth=2)  # 绘制验证损失曲线
        
    plt.title("训练与验证损失（第 0 折）")  # 设置曲线图标题
    plt.xlabel("轮次")  # 设置横坐标标签
    plt.ylabel("均方误差损失（归一化后）")  # 设置纵坐标标签
    plt.yscale('log')  # 设置纵轴为对数尺度
    plt.legend(frameon=True, facecolor='white', ncol=2)  # 显示图例并设置两列
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(OUTPUT_DIR, "2_training_curves.png"), dpi=200, bbox_inches='tight')  # 保存训练曲线图
    plt.close()  # 关闭当前图像

def plot_parity_plot(best_model):  # 定义一致性散点图函数
    """
    3. 一致性图：绘制最佳模型的预测值与真实值散点图
    """
    oof_path = os.path.join(OUTPUT_DIR, f"{best_model}_oof_predictions.npz")  # 拼接 OOF 预测文件路径
    if not os.path.exists(oof_path):  # 检查 OOF 文件是否存在
        return  # 如果不存在则直接返回
        
    data = np.load(oof_path, allow_pickle=True)  # 加载 OOF 数据文件
    preds = data['preds']  # 读取预测结果数组
    targets = data['targets']  # 读取真实值数组
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))  # 创建 2×2 子图
    axes = axes.flatten()  # 将子图展开为一维数组
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 定义通道颜色列表
    
    for c_idx, ch in enumerate(TARGET_CHANNELS):  # 遍历每个目标通道
        y_true = targets[:, c_idx].flatten()  # 展平真实值数组
        y_pred = preds[:, c_idx].flatten()  # 展平预测值数组
        
        # 为了让图更清晰，进行子采样（最多 10000 个点）
        if len(y_true) > 10000:  # 如果点数过多则子采样
            indices = np.random.choice(len(y_true), 10000, replace=False)  # 随机选择采样索引
            y_t = y_true[indices]  # 提取采样后的真实值
            y_p = y_pred[indices]  # 提取采样后的预测值
        else:  # 如果点数不多则直接使用全部数据
            y_t, y_p = y_true, y_pred  # 保留全部点
            
        r2 = r2_score(y_t, y_p)  # 计算该通道的 R²
        
        axes[c_idx].scatter(y_t, y_p, alpha=0.3, s=4, color=colors[c_idx])  # 绘制散点图
        
        # 绘制 1:1 对角线
        min_val = min(y_t.min(), y_p.min())  # 计算对角线起点
        max_val = max(y_t.max(), y_p.max())  # 计算对角线终点
        axes[c_idx].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='1:1 一致线')  # 绘制参考线
        
        axes[c_idx].set_title(f"{ch} (R² = {r2:.4f})")  # 设置子图标题
        axes[c_idx].set_xlabel("真实物理值")  # 设置横坐标标签
        axes[c_idx].set_ylabel("预测物理值")  # 设置纵坐标标签
        axes[c_idx].legend(frameon=True, facecolor='white')  # 显示图例
        
    plt.suptitle(f"最佳模型一致性图（{best_model.upper()}）", y=0.98)  # 设置整图标题
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(OUTPUT_DIR, "3_parity_plot.png"), dpi=200, bbox_inches='tight')  # 保存一致性图
    plt.close()  # 关闭当前图像

def plot_curve_comparisons(best_model):  # 定义曲线对比函数
    """
    4. 曲线对比：展示 6 个随机验证样本的预测值与真实值
    """
    oof_path = os.path.join(OUTPUT_DIR, f"{best_model}_oof_predictions.npz")  # 拼接 OOF 预测文件路径
    if not os.path.exists(oof_path):  # 检查 OOF 文件是否存在
        return  # 如果不存在则直接返回
        
    data = np.load(oof_path, allow_pickle=True)  # 加载 OOF 数据文件
    preds = data['preds']  # 读取预测数组
    targets = data['targets']  # 读取真实数组
    meta = data['meta']  # 读取元数据
    
    # 网格 x 取值（x/h）
    grid_x = np.linspace(0.0, 8.0, 161)  # 生成空间坐标网格
    
    np.random.seed(42)  # 固定随机种子，保证抽样可复现
    indices = np.random.choice(len(preds), 6, replace=False)  # 随机抽取 6 个样本
    
    fig, axes = plt.subplots(6, 4, figsize=(18, 16), sharex='col')  # 创建 6×4 子图
    
    import re  # 导入正则表达式模块
    FOLDER_PATTERN = re.compile(r"fuke-ALL-h([\d\.]+)_i([\d\.]+)_angle([\d\.]+)")  # 定义文件夹名解析规则
    for i, idx in enumerate(indices):  # 遍历抽取到的样本索引
        case_info = meta[idx]  # 读取当前样本的元数据
        match = FOLDER_PATTERN.match(case_info['case'])  # 尝试匹配案例目录名
        if match:  # 如果匹配成功
            h_val, i_val, a_val = match.group(1), match.group(2), match.group(3)  # 提取参数值
            title_str = f"样本：h={h_val}, i={i_val}, a={a_val} | 波形：{case_info['wave']}"  # 生成样本标题
        else:  # 如果匹配失败
            title_str = f"样本：{case_info['case']} | 波形：{case_info['wave']}"  # 直接使用原始信息
        
        for c_idx, ch in enumerate(TARGET_CHANNELS):  # 遍历每个目标通道
            ax = axes[i, c_idx]  # 取出当前子图轴对象
            ax.plot(grid_x, targets[idx, c_idx], 'k-', label='有限元真实值', linewidth=1.5)  # 绘制真实曲线
            ax.plot(grid_x, preds[idx, c_idx], 'r--', label='机器学习预测值', linewidth=1.5)  # 绘制预测曲线
            
            if i == 0:  # 如果是第一行
                ax.set_title(f"{ch}")  # 设置列标题
            if c_idx == 0:  # 如果是第一列
                ax.set_ylabel(f"样本 {i+1}\n数值")  # 设置行标签
            if i == 5:  # 如果是最后一行
                ax.set_xlabel("x/h")  # 设置横坐标标签
                
            if c_idx == 3:  # 仅在最后一列添加图例
                ax.legend(frameon=True, facecolor='white', loc='upper right', prop={'size': 8})  # 显示图例
                
        # 绘制整行标题
        axes[i, 0].text(-0.35, 1.15, title_str, transform=axes[i, 0].transAxes,  # 设置行标题位置
                        fontsize=10, fontweight='bold', bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round'))  # 设置标题样式
        
    plt.suptitle(f"目标曲线拟合叠加示例（{best_model.upper()}）", y=1.01, fontsize=16)  # 设置整图标题
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(OUTPUT_DIR, "4_curve_comparisons.png"), dpi=200, bbox_inches='tight')  # 保存曲线对比图
    plt.close()  # 关闭当前图像

def plot_error_distribution(best_model):  # 定义误差分布分析函数
    """
    5. 误差分布：残差直方图和沿 x/h 的误差趋势
    """
    oof_path = os.path.join(OUTPUT_DIR, f"{best_model}_oof_predictions.npz")  # 拼接 OOF 预测文件路径
    if not os.path.exists(oof_path):  # 检查 OOF 文件是否存在
        return  # 如果不存在则直接返回
        
    data = np.load(oof_path, allow_pickle=True)  # 加载 OOF 数据文件
    preds = data['preds']  # 读取预测数组
    targets = data['targets']  # 读取真实数组
    
    grid_x = np.linspace(0.0, 8.0, 161)  # 生成空间坐标网格
    residuals = preds - targets  # 计算残差数组
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 创建 2×2 子图
    axes = axes.flatten()  # 展开子图数组
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 定义通道颜色列表
    
    # 5.1 整体残差直方图
    for c_idx, ch in enumerate(TARGET_CHANNELS):  # 遍历每个通道
        res_flat = residuals[:, c_idx].flatten()  # 展平该通道残差
        axes[c_idx].hist(res_flat, bins=100, color=colors[c_idx], alpha=0.7, edgecolor='k')  # 绘制直方图
        axes[c_idx].axvline(0, color='r', linestyle='--', linewidth=1.5)  # 绘制零残差参考线
        
        mae = np.mean(np.abs(res_flat))  # 计算平均绝对误差
        std = np.std(res_flat)  # 计算残差标准差
        axes[c_idx].set_title(f"{ch} 残差（MAE: {mae:.4f}, 标准差: {std:.4f}）")  # 设置子图标题
        axes[c_idx].set_xlabel("残差（预测值 - 真实值）")  # 设置横坐标标签
        axes[c_idx].set_ylabel("频数")  # 设置纵坐标标签
        
    plt.suptitle(f"残差分布（{best_model.upper()}）", y=0.98)  # 设置整图标题
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(OUTPUT_DIR, "5.1_residual_histograms.png"), dpi=200, bbox_inches='tight')  # 保存残差直方图
    plt.close()  # 关闭当前图像
    
    # 5.2 沿空间坐标 x/h 的误差趋势
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 再创建 2×2 子图
    axes = axes.flatten()  # 展开子图数组
    
    for c_idx, ch in enumerate(TARGET_CHANNELS):  # 遍历每个通道
        mean_abs_error_xh = np.mean(np.abs(residuals[:, c_idx]), axis=0)  # 计算每个网格点的平均绝对误差
        std_abs_error_xh = np.std(np.abs(residuals[:, c_idx]), axis=0)  # 计算每个网格点的误差离散程度
        
        axes[c_idx].plot(grid_x, mean_abs_error_xh, color=colors[c_idx], linewidth=2, label='平均绝对误差')  # 绘制误差趋势线
        axes[c_idx].fill_between(grid_x, mean_abs_error_xh - 0.5 * std_abs_error_xh,  # 填充误差带下边界
                                 mean_abs_error_xh + 0.5 * std_abs_error_xh,  # 填充误差带上边界
                     color=colors[c_idx], alpha=0.2, label='±0.5 标准差')  # 设置填充样式
        
        axes[c_idx].set_title(f"{ch} 沿空间坐标 x/h 的误差")  # 设置子图标题
        axes[c_idx].set_xlabel("x/h")  # 设置横坐标标签
        axes[c_idx].set_ylabel("绝对误差")  # 设置纵坐标标签
        axes[c_idx].set_ylim(bottom=0)  # 设置纵轴下界为 0
        axes[c_idx].legend(frameon=True, facecolor='white')  # 显示图例
        
    plt.suptitle(f"沿空间坐标 x/h 的 MAE 趋势（{best_model.upper()}）", y=0.98)  # 设置整图标题
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(OUTPUT_DIR, "5.2_error_trend_spatial.png"), dpi=200, bbox_inches='tight')  # 保存空间误差趋势图
    plt.close()  # 关闭当前图像

def run_worst_case_analysis(best_model):  # 定义最差样本分析函数
    """
    6. 最差样本分析：提取并展示 3 个最差案例
    """
    oof_path = os.path.join(OUTPUT_DIR, f"{best_model}_oof_predictions.npz")  # 拼接 OOF 预测文件路径
    if not os.path.exists(oof_path):  # 检查 OOF 文件是否存在
        return  # 如果不存在则直接返回
        
    data = np.load(oof_path, allow_pickle=True)  # 加载 OOF 数据文件
    preds = data['preds']  # 读取预测数组
    targets = data['targets']  # 读取真实数组
    meta = data['meta']  # 读取元数据
    
    # 计算每个样本在所有通道和空间点上的 MSE
    sample_mse = np.mean((preds - targets)**2, axis=(1, 2))  # 计算每个样本的均方误差
    
    # 获取 3 个最差样本的索引
    worst_indices = np.argsort(sample_mse)[-3:][::-1]  # 取出误差最大的 3 个样本
    
    grid_x = np.linspace(0.0, 8.0, 161)  # 生成空间坐标网格
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))  # 创建 3×4 子图
    
    print("\n================== 最差样本分析 ==================")  # 打印分析标题
    for idx_i, idx in enumerate(worst_indices):  # 遍历最差样本索引
        case_info = meta[idx]  # 读取当前样本元数据
        case_mse = sample_mse[idx]  # 读取当前样本误差
        case_name = case_info['case']  # 读取案例名称
        wave = case_info['wave']  # 读取波形名称
        
        print(f"第 {idx_i+1} 名最差样本：文件夹={case_name}，波形={wave}，样本索引={idx}，MSE={case_mse:.6f}")  # 打印样本信息
        
        title_str = f"最差第 {idx_i+1} 名：{case_name} | {wave} | MSE: {case_mse:.4f}"  # 生成行标题
        
        for c_idx, ch in enumerate(TARGET_CHANNELS):  # 遍历每个通道
            ax = axes[idx_i, c_idx]  # 获取当前子图轴对象
            ax.plot(grid_x, targets[idx, c_idx], 'k-', label='有限元真实值', linewidth=1.5)  # 绘制真实曲线
            ax.plot(grid_x, preds[idx, c_idx], 'r--', label='机器学习预测值', linewidth=1.5)  # 绘制预测曲线
            
            if idx_i == 0:  # 如果是第一行
                ax.set_title(f"{ch}")  # 设置列标题
            if c_idx == 0:  # 如果是第一列
                ax.set_ylabel(f"最差第 {idx_i+1} 名\n数值")  # 设置行标签
            if idx_i == 2:  # 如果是最后一行
                ax.set_xlabel("x/h")  # 设置横坐标标签
            if c_idx == 3:  # 如果是最后一列
                ax.legend(frameon=True, facecolor='white', prop={'size': 8})  # 显示图例
                
        # 行标题
        axes[idx_i, 0].text(-0.3, 1.15, title_str, transform=axes[idx_i, 0].transAxes,  # 设置行标题位置
                            fontsize=9, fontweight='bold', bbox=dict(facecolor='red', alpha=0.2, boxstyle='round'))  # 设置标题样式
        
    plt.suptitle(f"前三个最差预测曲线拟合案例（{best_model.upper()}）", y=1.02, fontsize=15)  # 设置整图标题
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(OUTPUT_DIR, "6_worst_case_analysis.png"), dpi=200, bbox_inches='tight')  # 保存最差样本分析图
    plt.close()  # 关闭当前图像

def plot_extrapolation_heatmap(best_model):  # 定义外推热力图函数
    """
    7. 外推分析：按高度 h 和坡角 i 分组绘制平均 MAE 热力图
    """
    oof_path = os.path.join(OUTPUT_DIR, f"{best_model}_oof_predictions.npz")  # 拼接 OOF 预测文件路径
    if not os.path.exists(oof_path):  # 检查 OOF 文件是否存在
        return  # 如果不存在则直接返回
        
    data = np.load(oof_path, allow_pickle=True)  # 加载 OOF 数据文件
    preds = data['preds']  # 读取预测数组
    targets = data['targets']  # 读取真实数组
    meta = data['meta']  # 读取元数据
    
    # 从元数据中解析条件值（h、i、angle）
    h_vals = []  # 存放高度参数
    i_vals = []  # 存放坡角参数
    angle_vals = []  # 存放角度参数
    
    import re  # 导入正则表达式模块
    FOLDER_PATTERN = re.compile(r"fuke-ALL-h([\d\.]+)_i([\d\.]+)_angle([\d\.]+)")  # 定义案例名解析规则
    for case in meta:  # 遍历每个样本的元数据
        case_name = case['case']  # 读取案例名
        match = FOLDER_PATTERN.match(case_name)  # 匹配案例名格式
        if match:  # 如果匹配成功
            h = float(match.group(1))  # 提取 h 参数
            i = float(match.group(2))  # 提取 i 参数
            angle = float(match.group(3))  # 提取 angle 参数
        else:  # 如果匹配失败
            h, i, angle = 0.0, 0.0, 0.0  # 使用默认值
        h_vals.append(h)  # 记录 h 参数
        i_vals.append(i)  # 记录 i 参数
        angle_vals.append(angle)  # 记录 angle 参数
        
    h_vals = np.array(h_vals)  # 转换为数组
    i_vals = np.array(i_vals)  # 转换为数组
    angle_vals = np.array(angle_vals)  # 转换为数组
    
    # 计算每个样本的误差（跨通道和点的平均 MAE）
    sample_mae = np.mean(np.abs(preds - targets), axis=(1, 2))  # 计算每个样本的平均绝对误差
    
    # 获取唯一类别
    unique_h = sorted(list(set(h_vals)))  # 获取去重后的 h 取值
    unique_i = sorted(list(set(i_vals)))  # 获取去重后的 i 取值
    
    # 构建 h 与 i 的热力图网格
    heatmap_grid = np.zeros((len(unique_i), len(unique_h)))  # 初始化热力图矩阵
    
    for row_idx, i in enumerate(unique_i):  # 遍历每个 i 取值
        for col_idx, h in enumerate(unique_h):  # 遍历每个 h 取值
            mask = (h_vals == h) & (i_vals == i)  # 找出对应分组样本
            if mask.sum() > 0:  # 如果该分组存在样本
                heatmap_grid[row_idx, col_idx] = sample_mae[mask].mean()  # 计算该分组平均 MAE
            else:  # 如果该分组没有样本
                heatmap_grid[row_idx, col_idx] = np.nan  # 填充为空值
                
    fig, ax = plt.subplots(figsize=(8, 6))  # 创建热力图画布
    im = ax.imshow(heatmap_grid, cmap='Oranges', origin='lower')  # 绘制热力图
    
    # 坐标轴标签
    ax.set_xticks(np.arange(len(unique_h)))  # 设置横轴刻度位置
    ax.set_xticklabels([int(h) for h in unique_h])  # 设置横轴刻度标签
    ax.set_yticks(np.arange(len(unique_i)))  # 设置纵轴刻度位置
    ax.set_yticklabels([int(i) for i in unique_i])  # 设置纵轴刻度标签
    
    ax.set_xlabel("坡高 h")  # 设置横坐标标签
    ax.set_ylabel("坡角 i (°)")  # 设置纵坐标标签
    ax.set_title("泛化表现：按 (h, i) 分组的平均 MAE")  # 设置热力图标题
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)  # 添加颜色条对象
    cbar.ax.set_ylabel("平均绝对误差（MAE）", rotation=-90, va="bottom")  # 设置颜色条标签
    
    # 遍历网格并添加文本标注
    for row_idx in range(len(unique_i)):  # 遍历每一行
        for col_idx in range(len(unique_h)):  # 遍历每一列
            val = heatmap_grid[row_idx, col_idx]  # 读取当前格点值
            if not np.isnan(val):  # 如果该格点不是空值
                ax.text(col_idx, row_idx, f"{val:.3f}", ha="center", va="center",  # 在格点上显示数值
                        color="black" if val < heatmap_grid[~np.isnan(heatmap_grid)].mean() else "white",  # 根据背景深浅切换字体颜色
                        fontweight='bold')  # 设置文字加粗
                
    plt.tight_layout()  # 自动调整布局
    plt.savefig(os.path.join(OUTPUT_DIR, "7_extrapolation_heatmap.png"), dpi=200, bbox_inches='tight')  # 保存热力图
    plt.close()  # 关闭当前图像

def main():  # 定义主函数
    print("开始评估分析...")  # 打印开始信息
    
    summary_path = os.path.join(OUTPUT_DIR, "summary_metrics.json")  # 拼接汇总指标文件路径
    if not os.path.exists(summary_path):  # 检查汇总文件是否存在
        print(f"错误：未找到 {summary_path}。请先运行 train_v2.py。")  # 打印错误信息
        return  # 如果不存在则结束程序
        
    with open(summary_path, 'r') as f:  # 以只读方式打开汇总指标文件
        summary = json.load(f)  # 读取汇总指标
        
    # 绘制模型对比图
    print("正在生成模型对比图...")  # 打印处理信息
    plot_model_comparison(summary)  # 调用模型对比绘图函数
    
    # 绘制训练曲线图
    print("正在生成训练曲线图...")  # 打印处理信息
    plot_training_curves(list(summary.keys()))  # 调用训练曲线绘图函数
    
    # 根据 4 个通道的平均 R2 找出最佳模型
    best_model = None  # 初始化最佳模型变量
    best_avg_r2 = -float('inf')  # 初始化最佳平均 R²
    
    for model in summary.keys():  # 遍历每个模型
        avg_r2 = np.mean([summary[model][f"{ch}_R2"] for ch in TARGET_CHANNELS])  # 计算该模型平均 R²
        print(f"模型 {model.upper()} 的平均 OOF R2：{avg_r2:.4f}")  # 打印该模型平均 R²
        if avg_r2 > best_avg_r2:  # 如果当前模型更优
            best_avg_r2 = avg_r2  # 更新最佳平均 R²
            best_model = model  # 更新最佳模型名称
            
    print(f"\n识别出的最佳模型：{best_model.upper()}（平均 R2 = {best_avg_r2:.4f}）")  # 打印最佳模型结果
    
    # 生成最佳模型的各类图表
    print("正在生成一致性图...")  # 打印处理信息
    plot_parity_plot(best_model)  # 调用一致性图函数
    
    print("正在生成曲线叠加图...")  # 打印处理信息
    plot_curve_comparisons(best_model)  # 调用曲线叠加函数
    
    print("正在生成残差分析图...")  # 打印处理信息
    plot_error_distribution(best_model)  # 调用残差分析函数
    
    print("正在执行最差样本分析...")  # 打印处理信息
    run_worst_case_analysis(best_model)  # 调用最差样本分析函数
    
    print("正在生成外推分析热力图...")  # 打印处理信息
    plot_extrapolation_heatmap(best_model)  # 调用外推热力图函数
    
    print("\n所有评估图和诊断图已生成并保存到 outputs 文件夹。")  # 打印完成信息

if __name__ == '__main__':  # 判断是否作为主程序执行
    main()  # 调用主函数
