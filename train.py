#!/usr/bin/env python3
"""
统一训练脚本 - 支持多种数据格式
使用专业的training模块进行训练
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.training.trainer import (ModelTrainer, create_optimizer,
                                  create_scheduler)
from src.training.utils import (get_device, set_seed, create_loss_function,
                                print_device_info)
from src.utils.logger import create_experiment_logger
from src.utils.data_loader import load_data_from_path


def create_data_loader(X, y, batch_size=32, shuffle=True):
    """创建数据加载器"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)





def create_model(config, input_size, device):
    """创建模型"""
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一序列预测模型训练')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据路径（文件或文件夹）')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help='实验名称')
    parser.add_argument('--recursive', action='store_true',
                        help='递归搜索子文件夹中的CSV文件')
    parser.add_argument('--no_source_column', action='store_true',
                        help='不添加源文件列')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    args = parse_args()
    
    print("=== 统一序列预测模型训练 ===")
    print(f"实验名称: {args.experiment_name}")
    print(f"配置文件: {args.config}")
    print(f"数据文件: {args.data_path}")
    
    # 创建实验专用目录
    experiment_dir = os.path.join("results", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"实验目录: {experiment_dir}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    if 'seed' in config:
        set_seed(config['seed'])
        print(f"设置随机种子: {config['seed']}")
    
    # 打印设备信息
    print_device_info()
    
    # 获取设备
    device = get_device(config.get('device', 'auto'))
    
    # 创建实验日志记录器（保存到实验目录）
    logger = create_experiment_logger(
        args.experiment_name,
        experiment_dir  # 将日志保存到实验目录
    )
    logger.info(f"开始实验: {args.experiment_name}")
    
    # 创建预处理器并加载数据
    print("\n1. 创建数据预处理器并加载数据...")
    
    # 判断输入路径类型并相应加载数据
    if os.path.isfile(args.data_path):
        # 单个CSV文件，使用原有方式
        print(f"检测到单个文件: {args.data_path}")
        preprocessor, data = DataPreprocessor.from_config_and_data(
            config, args.data_path)
        print(f"加载数据完成: {data.shape}")
    elif os.path.isdir(args.data_path):
        # 文件夹，使用新的批量加载方式
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
            print(f"文件列表: {list(data['source_file'].unique())}")
        
        # 创建预处理器（直接传入数据）
        preprocessor, _ = DataPreprocessor.from_config_and_data(
            config, "", data=data)
    else:
        raise FileNotFoundError(f"路径不存在: {args.data_path}")
    
    # 原始数据已在preprocessor中保存，不需要额外副本
    
    logger.info(f"数据形状: {data.shape}")
    logger.info(f"输入列: {preprocessor.input_columns}")
    logger.info(f"目标列: {preprocessor.target_column}")
    
    # 划分数据
    print("\n2. 划分数据集...")
    train_data, val_data, test_data = preprocessor.split_data(
        data,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"验证集: X={X_val.shape}, y={y_val.shape}")
    print(f"测试集: X={X_test.shape}, y={y_test.shape}")
    
    logger.info(
        f"数据形状 - 训练: {X_train.shape}, 验证: {X_val.shape}, "
        f"测试: {X_test.shape}"
    )
    
    # 创建数据加载器
    print("\n3. 创建数据加载器...")
    batch_size = config['training']['batch_size']
    train_loader = create_data_loader(
        X_train, y_train, batch_size, shuffle=True
    )
    val_loader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = create_data_loader(X_test, y_test, batch_size, shuffle=False)
    
    # 创建模型
    print("\n4. 创建模型...")
    input_size = X_train.shape[2]  # 特征维度
    model = create_model(config, input_size, device)
    
    print(f"输入特征维度: {input_size}")
    print(f"模型参数数量: {model.get_num_parameters()}")
    logger.info(
        f"模型: {config['model']['type']}, "
        f"参数数量: {model.get_num_parameters()}"
    )
    
    # 创建损失函数
    loss_type = config['training'].get('loss_function', {}).get('type', 'auto')
    criterion = create_loss_function('regression', 1, loss_type)
    logger.info(f"损失函数: {type(criterion).__name__}")
    
    # 创建优化器和调度器
    # 将嵌套的配置扁平化以适配trainer模块的期望格式
    training_config = config['training'].copy()
    training_config['optimizer'] = training_config['optimizer']['type']
    
    # 处理调度器配置
    scheduler_config = training_config['scheduler']
    training_config['scheduler'] = scheduler_config['type']
    training_config['step_size'] = scheduler_config.get('step_size', 30)
    training_config['gamma'] = scheduler_config.get('gamma', 0.5)
    training_config['scheduler_patience'] = scheduler_config.get(
        'patience', 10
    )
    
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    
    # 创建训练器
    print("\n5. 创建训练器...")
    
    # 修改训练配置，将检查点保存到实验目录
    training_config_with_dir = config['training'].copy()
    training_config_with_dir['checkpoint_dir'] = os.path.join(
        experiment_dir, 'checkpoints'
    )
    training_config_with_dir['tensorboard'] = config.get(
        'logging', {}
    ).get('tensorboard', False)
    training_config_with_dir['log_dir'] = experiment_dir
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=training_config_with_dir,
        scheduler=scheduler,
        logger=logger
    )
    
    # 开始训练
    logger.info("\n6. 开始训练...")
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)
    
    # 测试模型
    print("\n7. 测试模型...")
    logger.info("开始测试模型...")
    
    # 加载最佳模型
    checkpoint_best_path = os.path.join(
        trainer.checkpoint_dir, 'best_model.pth'
    )
    if os.path.exists(checkpoint_best_path):
        checkpoint = torch.load(checkpoint_best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("加载最佳模型进行测试")
    
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += criterion(output, target).item()
            
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"测试损失: {avg_test_loss:.4f}")
    
    # 保存结果
    print("\n8. 保存模型和配置...")
    logger.info("开始保存实验结果...")
    
    # 保存最终模型
    final_model_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # 复制最佳模型到实验目录根目录（方便访问）
    checkpoint_best_path = os.path.join(
        trainer.checkpoint_dir, 'best_model.pth'
    )
    if os.path.exists(checkpoint_best_path):
        import shutil
        best_model_dest = os.path.join(experiment_dir, "best_model.pth")
        shutil.copy2(checkpoint_best_path, best_model_dest)
        logger.info(f"复制最佳模型到: {best_model_dest}")

    # 反归一化预测结果和目标值
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if hasattr(preprocessor, 'inverse_transform_target'):
        logger.info("反归一化预测结果和目标值...")
        predictions_original = preprocessor.inverse_transform_target(
            predictions
        )
        targets_original = preprocessor.inverse_transform_target(targets)
    else:
        predictions_original = predictions
        targets_original = targets

    # 计算详细指标
    from src.utils.metrics import calculate_metrics
    metrics = calculate_metrics(
        targets_original, predictions_original, task_type='regression'
    )

    # 保存评估结果
    eval_results = {
        'test': {
            'test_loss': float(avg_test_loss),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'num_samples': len(targets_original)
        }
    }
    
    eval_results_file = os.path.join(experiment_dir, 'test_results.yaml')
    with open(eval_results_file, 'w', encoding='utf-8') as f:
        yaml.dump(eval_results, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"详细评估结果已保存到: {eval_results_file}")

    # 保存预测结果（原始尺度）
    predictions_file = os.path.join(experiment_dir, 'test_predictions.csv')
    
    # 使用预处理器的方法获取包含文件数据的预测结果
    base_paths = [os.path.join(args.data_path, 'signals'), '../signals', 'data/signals']
    df_results = preprocessor.get_test_predictions_with_file_data(
        predictions_original, targets_original, base_paths
    )
    
    df_results.to_csv(predictions_file, index=False)
    logger.info(f"测试集预测结果已保存到: {predictions_file}")
    
    # 保存预处理器
    import pickle
    preprocessor_path = os.path.join(experiment_dir, "preprocessor.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # 保存配置
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 保存训练结果和指标
    results = {
        'experiment_name': args.experiment_name,
        'final_test_loss': float(avg_test_loss),
        'best_validation_loss': float(trainer.best_val_loss),
        'model_type': config['model']['type'],
        'model_parameters': model.get_num_parameters(),
        'data_type': 'unified',
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'test_samples': X_test.shape[0],
        'input_size': input_size,
        'epochs_trained': trainer.current_epoch
    }
    
    results_path = os.path.join(experiment_dir, "results.yaml")
    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    print("✓ 训练完成！")
    print(f"实验目录: {experiment_dir}")
    print(f"最佳模型: {os.path.join(experiment_dir, 'best_model.pth')}")
    print(f"最终模型: {final_model_path}")
    print(f"预处理器: {preprocessor_path}")
    print(f"配置文件: {config_path}")
    print(f"训练结果: {results_path}")
    print(f"检查点目录: {trainer.checkpoint_dir}")
    if training_config_with_dir.get('tensorboard'):
        print(f"TensorBoard日志: {os.path.join(experiment_dir, 'run_*')}")
    
    logger.info("训练流程完成")


if __name__ == '__main__':
    main()
