#!/usr/bin/env python3
"""
统一模型评估脚本 - 适配新的文件组织结构
"""

import argparse
import os
import sys
import torch
import numpy as np
import yaml
import pickle

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.training.utils import get_device, create_loss_function
from src.utils.metrics import calculate_metrics, print_regression_report, print_classification_report
from src.utils.logger import setup_logger
from src.utils.data_loader import load_data_from_path
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估序列预测模型')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='实验目录路径 (results/实验名)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='测试数据路径（文件或文件夹）')
    parser.add_argument('--use_best_model', action='store_true',
                        help='使用最佳模型而不是最终模型')
    parser.add_argument('--recursive', action='store_true',
                        help='递归搜索子文件夹中的CSV文件')
    parser.add_argument('--no_source_column', action='store_true',
                        help='不添加源文件列')
    
    return parser.parse_args()


def load_model_and_config(experiment_dir, use_best_model, device):
    """从实验目录加载模型和配置"""
    # 加载配置
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 选择模型文件
    if use_best_model:
        model_path = os.path.join(experiment_dir, 'best_model.pth')
    else:
        model_path = os.path.join(experiment_dir, 'final_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载预处理器
    preprocessor_path = os.path.join(experiment_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    return config, model_path, preprocessor


def create_model_from_config(config, input_size, device):
    """根据配置创建模型"""
    model_type = config['model']['type'].lower()
    model_config = config['model']
    
    if model_type == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            num_classes=1,  # 回归任务
            dropout=model_config.get('dropout', 0.2),
            bidirectional=model_config.get('bidirectional', True)
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_size=input_size,
            d_model=model_config.get('hidden_size', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            num_classes=1,
            dropout=model_config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    model.to(device)
    return model


def create_data_loader(X, y, batch_size=32):
    """创建数据加载器"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, data_loader, criterion, device):
    """评估模型"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            output = model(data)
            # 确保输出和目标的形状匹配
            if len(output.shape) > 1 and output.shape[1] == 1:
                output = output.squeeze()
            
            loss = criterion(output, target)
            
            # 按样本数累计损失
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    return avg_loss, predictions, targets


def main():
    """主函数"""
    args = parse_args()
    
    print("=== 模型评估 ===")
    print(f"实验目录: {args.experiment_dir}")
    print(f"测试数据: {args.data_path}")
    print(f"使用最佳模型: {args.use_best_model}")
    
    # 获取设备
    device = get_device('auto')
    
    # 设置日志
    logger = setup_logger("evaluate")
    
    # 加载模型和配置
    config, model_path, preprocessor = load_model_and_config(
        args.experiment_dir, args.use_best_model, device
    )
    logger.info(f"加载模型: {model_path}")
    
    # 加载测试数据
    if os.path.isfile(args.data_path):
        # 单个CSV文件
        print(f"检测到单个文件: {args.data_path}")
        import pandas as pd
        data = pd.read_csv(args.data_path)
        print(f"加载数据完成: {data.shape}")
    elif os.path.isdir(args.data_path):
        # 文件夹，使用批量加载
        print(f"检测到文件夹: {args.data_path}")
        data = load_data_from_path(
            args.data_path,
            recursive=args.recursive,
            add_source_column=not args.no_source_column
        )
        
        print(f"从文件夹 {args.data_path} 加载数据完成: {data.shape}")
        if 'source_file' in data.columns:
            file_count = data['source_file'].nunique()
            print(f"合并了 {file_count} 个CSV文件")
    else:
        raise FileNotFoundError(f"路径不存在: {args.data_path}")
    
    logger.info(f"测试数据形状: {data.shape}")
    
    # 使用预处理器处理测试数据
    X_test, y_test = preprocessor.transform(data)
    
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    
    # 创建模型
    input_size = X_test.shape[2]
    model = create_model_from_config(config, input_size, device)
    
    # 加载模型权重
    if args.use_best_model:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    logger.info(f"模型参数数量: {model.get_num_parameters()}")
    
    # 创建数据加载器
    test_loader = create_data_loader(X_test, y_test, batch_size=32)
    
    # 创建损失函数
    criterion = create_loss_function('regression', 1)
    
    # 评估模型
    print("\n开始评估...")
    test_loss, predictions, targets = evaluate_model(model, test_loader, criterion, device)
    
    # 反归一化预测结果和目标值
    if hasattr(preprocessor, 'inverse_transform_target'):
        print("反归一化预测结果和目标值...")
        predictions_original = preprocessor.inverse_transform_target(predictions)
        targets_original = preprocessor.inverse_transform_target(targets)
    else:
        predictions_original = predictions
        targets_original = targets
    
    # 计算指标（使用原始尺度的数据）
    metrics = calculate_metrics(targets_original, predictions_original, task_type='regression')
    
    # 打印结果
    print(f"\n=== 评估结果 ===")
    print(f"测试损失 (归一化): {test_loss:.6f}")
    print("回归指标 (原始尺度):")
    print_regression_report(targets_original, predictions_original)
    
    # 保存评估结果
    results = {
        'test_loss': float(test_loss),
        'model_path': model_path,
        'data_path': args.data_path,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'test_samples': len(targets_original)
    }
    
    results_file = os.path.join(args.experiment_dir, 'evaluation_results.yaml')
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    # 保存预测结果（原始尺度）
    predictions_file = os.path.join(args.experiment_dir, 'test_predictions.csv')
    import pandas as pd
    
    # 构建结果DataFrame
    result_dict = {
        'CIR': data['CIR'],
        'target': targets_original,
        'prediction': predictions_original,
        'error': targets_original - predictions_original
    }
    
    # 添加配置中指定需要保存的列
    if hasattr(preprocessor, 'keep_original_columns'):
        for col in preprocessor.keep_original_columns:
            if col in data.columns and col not in result_dict:
                result_dict[col] = data[col].values
                print(f"已添加列 '{col}' 到预测结果中")
    
    df_results = pd.DataFrame(result_dict)
    df_results.to_csv(predictions_file, index=False)
    
    print(f"\n评估完成！")
    print(f"评估结果: {results_file}")
    print(f"预测结果: {predictions_file}")
    
    logger.info("评估完成")


if __name__ == '__main__':
    main()