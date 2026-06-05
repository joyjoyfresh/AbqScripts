import os  # 导入操作系统路径工具
import json  # 导入 JSON 读写工具
import logging  # 导入日志工具
import numpy as np  # 导入数值计算库
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.utils.data import DataLoader  # 导入数据加载器
from sklearn.model_selection import GroupKFold  # 导入分组交叉验证器
from sklearn.metrics import r2_score  # 导入 R² 评估函数

from config_v2 import DATA_DIR, OUTPUT_DIR, TRAIN_CONFIG, MODEL_CONFIGS, COND_FEATURES, TARGET_CHANNELS  # 导入配置常量
from utils_v2 import set_seed, setup_logger, DataScaler  # 导入随机种子、日志和归一化工具
from data.dataset_v2 import load_raw_dataset, SeisMLDataset  # 导入数据读取和数据集封装函数
from models.cnn_encoder import CNNModel  # 导入 CNN 模型
from models.lstm_encoder import LSTMModel  # 导入 LSTM 模型
from models.transformer_encoder import TransformerModel  # 导入 Transformer 模型
from models.deeponet import DeepONetModel  # 导入 DeepONet 模型

def get_model(model_name):  # 定义模型构建函数
    cfg = MODEL_CONFIGS[model_name]  # 读取指定模型的配置
    if model_name == 'cnn':  # 判断是否为 CNN 模型
        return CNNModel(wave_emb_dim=cfg['wave_emb_dim'], dropout=cfg['dropout'])  # 构建 CNN 模型
    elif model_name == 'lstm':  # 判断是否为 LSTM 模型
        return LSTMModel(wave_emb_dim=cfg['wave_emb_dim'], hidden_dim=cfg['hidden_dim'],  # 构建 LSTM 模型
                         num_layers=cfg['num_layers'], dropout=cfg['dropout'])  # 继续传入 LSTM 参数
    elif model_name == 'transformer':  # 判断是否为 Transformer 模型
        return TransformerModel(wave_emb_dim=cfg['wave_emb_dim'], d_model=cfg['d_model'],  # 构建 Transformer 模型
                                nhead=cfg['nhead'], num_layers=cfg['num_layers'],  # 继续传入 Transformer 参数
                                dim_feedforward=cfg['dim_feedforward'], dropout=cfg['dropout'])  # 继续传入 Transformer 参数
    elif model_name == 'deeponet':  # 判断是否为 DeepONet 模型
        return DeepONetModel(wave_emb_dim=cfg['wave_emb_dim'], trunk_hidden_dims=cfg['trunk_hidden_dims'])  # 构建 DeepONet 模型
    else:  # 处理未知模型名称
        raise ValueError(f"Unknown model name: {model_name}")  # 抛出错误提示

def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):  # 定义单个训练轮次函数
    model.train()  # 切换到训练模式
    total_loss = 0.0  # 初始化累计损失
    for wave, cond, length, target in dataloader:  # 遍历批次数据
        wave = wave.to(device)  # 将波形张量移到设备
        cond = cond.to(device)  # 将条件张量移到设备
        length = length.to(device)  # 将长度张量移到设备
        target = target.to(device)  # 将目标张量移到设备
        
        optimizer.zero_grad()  # 清空梯度
        
        # 前向传播
        # LSTM 和 Transformer 模型需要传入 length 参数
        if isinstance(model, (LSTMModel, TransformerModel)):  # 判断是否为需要长度信息的模型
            pred = model(wave, cond, length)  # 传入长度进行前向传播
        else:  # 处理其他模型
            pred = model(wave, cond)  # 仅传入波形和条件
            
        loss = criterion(pred, target)  # 计算损失
        loss.backward()  # 反向传播
        
        # 梯度裁剪
        if grad_clip > 0:  # 判断是否启用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 裁剪梯度范数
            
        optimizer.step()  # 更新参数
        total_loss += loss.item() * wave.size(0)  # 累计当前批次损失
        
    return total_loss / len(dataloader.dataset)  # 返回平均训练损失

def validate(model, dataloader, criterion, device):  # 定义验证函数
    model.eval()  # 切换到评估模式
    total_loss = 0.0  # 初始化累计损失
    all_preds = []  # 初始化预测结果列表
    all_targets = []  # 初始化真实值列表
    
    with torch.no_grad():  # 关闭梯度计算
        for wave, cond, length, target in dataloader:  # 遍历验证批次
            wave = wave.to(device)  # 将波形张量移到设备
            cond = cond.to(device)  # 将条件张量移到设备
            length = length.to(device)  # 将长度张量移到设备
            target = target.to(device)  # 将目标张量移到设备
            
            if isinstance(model, (LSTMModel, TransformerModel)):  # 判断是否为需要长度信息的模型
                pred = model(wave, cond, length)  # 传入长度进行推理
            else:  # 处理其他模型
                pred = model(wave, cond)  # 仅传入波形和条件
                
            loss = criterion(pred, target)  # 计算验证损失
            total_loss += loss.item() * wave.size(0)  # 累计验证损失
            
            all_preds.append(pred.cpu().numpy())  # 保存预测结果到 CPU 数组
            all_targets.append(target.cpu().numpy())  # 保存真实值到 CPU 数组
            
    val_loss = total_loss / len(dataloader.dataset)  # 计算平均验证损失
    all_preds = np.concatenate(all_preds, axis=0)  # 拼接所有预测结果
    all_targets = np.concatenate(all_targets, axis=0)  # 拼接所有真实值
    
    return val_loss, all_preds, all_targets  # 返回验证损失、预测值和真实值

def run_cross_validation(model_name, waves, conds, targets, meta, groups, device):  # 定义交叉验证主函数
    logger = logging.getLogger()  # 获取当前日志记录器
    logger.info(f"========== Starting 5-Fold GroupKFold CV for {model_name.upper()} ==========")  # 打印交叉验证开始信息
    
    gkf = GroupKFold(n_splits=TRAIN_CONFIG['num_folds'])  # 创建分组交叉验证器
    
    # 保存所有样本的预测值和真实值，来源为交叉验证中的 OOF 结果
    oof_preds = np.zeros_like(targets)  # 初始化 OOF 预测数组
    fold_metrics = []  # 初始化每折指标列表
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(waves, targets, groups)):  # 遍历每一折划分
        logger.info(f"--- Model: {model_name}, Fold {fold+1}/{TRAIN_CONFIG['num_folds']} ---")  # 打印当前折信息
        
        # 划分数据
        train_waves, val_waves = waves[train_idx], waves[val_idx]  # 划分训练和验证波形
        train_conds, val_conds = conds[train_idx], conds[val_idx]  # 划分训练和验证条件
        train_targets, val_targets = targets[train_idx], targets[val_idx]  # 划分训练和验证目标
        train_meta = [meta[i] for i in train_idx]  # 提取训练样本元数据
        val_meta = [meta[i] for i in val_idx]  # 提取验证样本元数据
        
        # 仅在训练折上实例化并拟合归一化器
        scaler = DataScaler()  # 创建数据归一化器
        train_dataset = SeisMLDataset(train_waves, train_conds, train_targets, train_meta, scaler, fit_scaler=True)  # 构建训练数据集
        val_dataset = SeisMLDataset(val_waves, val_conds, val_targets, val_meta, scaler, fit_scaler=False)  # 构建验证数据集
        
        # 保存归一化器，供推理和评估使用
        scaler_path = os.path.join(OUTPUT_DIR, f"{model_name}_scaler_fold{fold}.json")  # 拼接归一化器路径
        scaler.save(scaler_path)  # 保存归一化器
        
        # 构建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)  # 构建训练加载器
        val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)  # 构建验证加载器
        
        # 初始化模型
        model = get_model(model_name).to(device)  # 构建并移动模型到设备
        criterion = nn.MSELoss()  # 定义均方误差损失
        optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],  # 创建优化器
                                weight_decay=TRAIN_CONFIG['weight_decay'])  # 设置权重衰减
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)  # 创建学习率调度器
        
        # 训练循环
        best_val_loss = float('inf')  # 初始化最佳验证损失
        patience_counter = 0  # 初始化早停计数器
        checkpoint_path = os.path.join(OUTPUT_DIR, f"{model_name}_fold{fold}.pt")  # 拼接模型检查点路径
        
        train_history = []  # 初始化训练损失历史
        val_history = []  # 初始化验证损失历史
        
        for epoch in range(1, TRAIN_CONFIG['max_epochs'] + 1):  # 遍历每个训练轮次
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, TRAIN_CONFIG['grad_clip'])  # 执行一轮训练
            val_loss, _, _ = validate(model, val_loader, criterion, device)  # 执行一轮验证
            
            scheduler.step(val_loss)  # 根据验证损失调整学习率
            
            train_history.append(train_loss)  # 记录训练损失
            val_history.append(val_loss)  # 记录验证损失
            
            if val_loss < best_val_loss:  # 判断当前验证损失是否更优
                best_val_loss = val_loss  # 更新最佳验证损失
                model.save_checkpoint(checkpoint_path)  # 保存当前最佳模型
                patience_counter = 0  # 重置早停计数器
            else:  # 当前验证损失未提升
                patience_counter += 1  # 增加早停计数器
                
            if epoch % 50 == 0 or epoch == 1:  # 每 50 轮或第 1 轮输出一次日志
                logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best Val: {best_val_loss:.6f}")  # 打印训练状态
                
            if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:  # 判断是否触发早停
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")  # 打印早停信息
                break  # 结束当前折训练
                
        # 保存第 0 折的训练历史，供后续绘制学习曲线
        if fold == 0:  # 如果是第 0 折
            with open(os.path.join(OUTPUT_DIR, f"{model_name}_history.json"), 'w') as f:  # 打开历史文件准备写入
                json.dump({  # 写入 JSON 内容
                    'train_loss': train_history,  # 保存训练损失历史
                    'val_loss': val_history  # 保存验证损失历史
                }, f, indent=4)  # 设置缩进格式
                
        # 在验证折上评估最佳模型
        model.load_checkpoint(checkpoint_path, device=device)  # 载入最佳模型权重
        _, fold_norm_preds, _ = validate(model, val_loader, criterion, device)  # 在验证集上重新推理
        
        # 将预测值和真实值反标准化为物理量纲
        fold_preds = scaler.inverse_transform_targets(fold_norm_preds)  # 将预测值反归一化
        fold_targets_unnorm = val_targets  # 原始目标值本身已经是未归一化状态
        
        # 保存到 OOF 数组中
        oof_preds[val_idx] = fold_preds  # 将当前折预测写回 OOF 数组
        
        # 计算当前折的评估指标
        metrics = {}  # 初始化当前折指标字典
        for c_idx, ch in enumerate(TARGET_CHANNELS):  # 遍历每个通道
            y_true = fold_targets_unnorm[:, c_idx].flatten()  # 展平真实值
            y_pred = fold_preds[:, c_idx].flatten()  # 展平预测值
            
            r2 = r2_score(y_true, y_pred)  # 计算R²
            mae = np.mean(np.abs(y_true - y_pred))  # 计算MAE
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))  # 计算RMSE
            
            metrics[f"{ch}_R2"] = float(r2)  # 保存R²结果
            metrics[f"{ch}_MAE"] = float(mae)  # 保存MAE结果
            metrics[f"{ch}_RMSE"] = float(rmse)  # 保存RMSE结果
            
            logger.info(f"Fold {fold+1} | {ch} -> R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")  # 打印当前折指标
            
        fold_metrics.append(metrics)  # 保存当前折指标到列表
        
    # 计算整体 OOF 指标
    oof_metrics = {}  # 初始化总体指标字典
    logger.info(f"--- Overall OOF Performance for {model_name.upper()} ---")  # 打印总体评估标题
    for c_idx, ch in enumerate(TARGET_CHANNELS):  # 遍历每个通道
        y_true = targets[:, c_idx].flatten()  # 展平真实值
        y_pred = oof_preds[:, c_idx].flatten()  # 展平预测值
        
        r2 = r2_score(y_true, y_pred)  # 计算总体R²
        mae = np.mean(np.abs(y_true - y_pred))  # 计算总体MAE
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))  # 计算总体RMSE
        
        oof_metrics[f"{ch}_R2"] = float(r2)  # 保存总体R²
        oof_metrics[f"{ch}_MAE"] = float(mae)  # 保存总体MAE
        oof_metrics[f"{ch}_RMSE"] = float(rmse)  # 保存总体RMSE
        
        logger.info(f"OOF Overall | {ch} -> R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")  # 打印总体指标
        
    # 保存 OOF 预测结果和指标
    np.savez(os.path.join(OUTPUT_DIR, f"{model_name}_oof_predictions.npz"),  # 保存 OOF 预测文件
             preds=oof_preds, targets=targets, meta=meta)  # 写入预测值、真实值和元数据
             
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_metrics.json"), 'w') as f:  # 打开指标文件准备写入
        json.dump({  # 写入 JSON 内容
            'fold_metrics': fold_metrics,  # 保存每折指标
            'oof_metrics': oof_metrics  # 保存整体指标
        }, f, indent=4)  # 设置缩进格式
        
    return oof_metrics  # 返回整体 OOF 指标

def main():  # 定义主函数
    set_seed(TRAIN_CONFIG['random_seed'])  # 固定随机种子
    log_file = os.path.join(OUTPUT_DIR, "training.log")  # 拼接日志文件路径
    logger = setup_logger(log_file)  # 初始化日志器
    
    device = TRAIN_CONFIG['device']  # 读取训练设备配置
    logger.info(f"Using device: {device}")  # 打印设备信息
    
    # 1. 读取原始数据集
    logger.info("Loading raw dataset...")  # 打印数据读取信息
    waves, conds, targets, meta = load_raw_dataset(DATA_DIR)  # 读取原始数据
    logger.info(f"Successfully loaded {len(meta)} samples.")  # 打印样本数量
    
    # 2. 定义分组（按文件夹条件分组）
    unique_conds = [tuple(c) for c in conds]  # 将条件数组转为元组列表
    cond_to_group = {cond: idx for idx, cond in enumerate(sorted(list(set(unique_conds))))}  # 构建条件到组编号的映射
    groups = np.array([cond_to_group[c] for c in unique_conds])  # 生成分组标签数组
    logger.info(f"Number of unique geometric cases (groups): {len(cond_to_group)}")  # 打印组数
    
    # 3. 训练所有模型结构
    summary_metrics = {}  # 初始化汇总指标字典
    for model_name in MODEL_CONFIGS.keys():  # 遍历所有模型名称
        oof_metrics = run_cross_validation(model_name, waves, conds, targets, meta, groups, device)  # 执行交叉验证
        summary_metrics[model_name] = oof_metrics  # 保存该模型的总体指标
        
    # 保存所有模型的汇总结果
    with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), 'w') as f:  # 打开汇总指标文件
        json.dump(summary_metrics, f, indent=4)  # 保存所有模型指标
        
    logger.info("All model architectures trained and evaluated successfully!")  # 打印完成信息

if __name__ == '__main__':  # 判断是否直接运行脚本
    main()  # 执行主函数
