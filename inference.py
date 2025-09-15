#!/usr/bin/env python3
"""
统一模型推理脚本 - 适配新的文件组织结构
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import yaml
import pickle

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.training.utils import get_device
from src.utils.logger import setup_logger
from src.utils.data_loader import load_data_from_path
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型推理')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='实验目录路径 (results/实验名)')
    parser.add_argument('--input_path', type=str, required=True,
                        help='输入数据路径（文件或文件夹）')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出文件路径')
    parser.add_argument('--use_best_model', action='store_true',
                        help='使用最佳模型而不是最终模型')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
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


def create_data_loader(X, batch_size=32):
    """创建推理数据加载器"""
    X_tensor = torch.FloatTensor(X)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def inference_model(model, data_loader, device):
    """模型推理"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for (data,) in data_loader:
            data = data.to(device)
            
            output = model(data)
            # 确保输出格式正确
            if len(output.shape) > 1 and output.shape[1] == 1:
                output = output.squeeze()
            
            all_predictions.extend(output.cpu().numpy())
    
    return np.array(all_predictions)


def main():
    """主函数"""
    args = parse_args()
    
    print("=== 模型推理 ===")
    print(f"实验目录: {args.experiment_dir}")
    print(f"输入数据: {args.input_path}")
    print(f"输出路径: {args.output_path}")
    print(f"使用最佳模型: {args.use_best_model}")
    
    # 获取设备
    device = get_device('auto')
    
    # 设置日志
    logger = setup_logger("inference")
    
    # 加载模型和配置
    config, model_path, preprocessor = load_model_and_config(
        args.experiment_dir, args.use_best_model, device
    )
    logger.info(f"加载模型: {model_path}")
    
    # 加载输入数据
    if os.path.isfile(args.input_path):
        # 单个CSV文件
        print(f"检测到单个文件: {args.input_path}")
        data = pd.read_csv(args.input_path)
        print(f"加载数据完成: {data.shape}")
    elif os.path.isdir(args.input_path):
        # 文件夹，使用批量加载
        print(f"检测到文件夹: {args.input_path}")
        data = load_data_from_path(
            args.input_path,
            recursive=args.recursive,
            add_source_column=not args.no_source_column
        )
        
        print(f"从文件夹 {args.input_path} 加载数据完成: {data.shape}")
        if 'source_file' in data.columns:
            file_count = data['source_file'].nunique()
            print(f"合并了 {file_count} 个CSV文件")
    else:
        raise FileNotFoundError(f"路径不存在: {args.input_path}")
    
    logger.info(f"输入数据形状: {data.shape}")
    
    # 使用预处理器处理数据（只需要特征）
    # 创建临时目标列用于预处理（推理时不使用）
    if preprocessor.target_column not in data.columns:
        data[preprocessor.target_column] = 0  # 临时填充
    
    X_input, _ = preprocessor.transform(data)
    
    print(f"输入数据形状: X={X_input.shape}")
    
    # 创建模型
    input_size = X_input.shape[2]
    model = create_model_from_config(config, input_size, device)
    
    # 加载模型权重
    if args.use_best_model:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    logger.info(f"模型参数数量: {model.get_num_parameters()}")
    
    # 创建数据加载器
    input_loader = create_data_loader(X_input, batch_size=args.batch_size)
    
    # 进行推理
    print("\n开始推理...")
    predictions = inference_model(model, input_loader, device)
    
    # 反归一化预测结果到原始尺度
    if hasattr(preprocessor, 'inverse_transform_target') and preprocessor.normalize_targets:
        print("反归一化预测结果到原始尺度...")
        predictions_original = preprocessor.inverse_transform_target(predictions)
        print(f"归一化预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"原始尺度预测范围: [{predictions_original.min():.4f}, {predictions_original.max():.4f}]")
    else:
        print("预测结果已为原始尺度")
        predictions_original = predictions
    
    print(f"推理完成，生成 {len(predictions_original)} 个预测结果")
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'sample_id': range(len(predictions_original)),
        'prediction': predictions_original
    })
    
    # 如果原始数据有标识列，添加到结果中
    if 'id' in data.columns:
        results_df['original_id'] = data['id'].values
    
    # 保存到CSV
    results_df.to_csv(args.output_path, index=False)
    
    print(f"\n推理完成！")
    print(f"预测结果已保存到: {args.output_path}")
    print(f"预测统计:")
    print(f"  - 最小值: {predictions_original.min():.4f}")
    print(f"  - 最大值: {predictions_original.max():.4f}")
    print(f"  - 平均值: {predictions_original.mean():.4f}")
    print(f"  - 标准差: {predictions_original.std():.4f}")
    
    logger.info("推理完成")


if __name__ == '__main__':
    main()
