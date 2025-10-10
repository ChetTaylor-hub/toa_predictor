#!/usr/bin/env python3
"""
对比学习训练脚本 - 支持无监督预训练和有监督微调
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.training.contrastive_trainer import (
    ContrastiveModelWrapper, ContrastiveTrainer, FineTuneTrainer
)
from src.training.trainer import ModelTrainer, create_optimizer, create_scheduler
from src.training.utils import (get_device, set_seed, create_loss_function,
                                print_device_info)
from src.utils.logger import create_experiment_logger
from src.utils.data_loader import load_data_from_path


def create_data_loader(X, y=None, batch_size=32, shuffle=True):
    """创建数据加载器"""
    if y is not None:
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        # 对比学习预训练时，y可以为None
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.zeros(X.shape[0])  # 创建虚拟标签
        dataset = TensorDataset(X_tensor, y_tensor)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_backbone_model(config, input_size, device):
    """创建主干网络模型"""
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
    
    return model


def contrastive_pretrain(model_wrapper, train_loader, val_loader, config, 
                        experiment_dir, logger, device):
    """对比学习预训练"""
    logger.info("开始对比学习预训练...")
    
    # 创建对比学习训练器
    contrastive_config = config.get('contrastive', {})
    contrastive_config.update({
        'checkpoint_dir': os.path.join(experiment_dir, 'pretrain_checkpoints'),
        'log_dir': experiment_dir,
        'tensorboard': config.get('logging', {}).get('tensorboard', False)
    })
    
    trainer = ContrastiveTrainer(
        model=model_wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=contrastive_config,
        logger=logger
    )
    
    # 开始预训练
    pretrain_epochs = contrastive_config.get('pretrain_epochs', 50)
    trainer.train(pretrain_epochs)
    
    logger.info("对比学习预训练完成")
    return trainer


def supervised_finetune(model_wrapper, train_loader, val_loader, config,
                       experiment_dir, logger, device):
    """有监督微调"""
    logger.info("开始有监督微调...")
    
    # 创建损失函数
    loss_type = config.get('finetune', {}).get('loss_function', {}).get('type', 'auto')
    criterion = create_loss_function('regression', 1, loss_type)
    
    # 微调配置
    finetune_config = config.get('finetune', {})
    finetune_config.update({
        'finetune_checkpoint_dir': os.path.join(experiment_dir, 'finetune_checkpoints'),
        'log_dir': experiment_dir,
        'tensorboard': config.get('logging', {}).get('tensorboard', False)
    })
    
    # 创建微调训练器
    trainer = FineTuneTrainer(
        contrastive_model=model_wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        config=finetune_config,
        logger=logger
    )
    
    # 开始微调
    finetune_epochs = finetune_config.get('epochs', 30)
    trainer.train(finetune_epochs)
    
    logger.info("有监督微调完成")
    return trainer


def supervised_training(model, train_loader, val_loader, config, 
                       experiment_dir, logger, device):
    """传统有监督训练"""
    logger.info("开始传统有监督训练...")
    
    # 创建损失函数
    loss_type = config['training'].get('loss_function', {}).get('type', 'auto')
    criterion = create_loss_function('regression', 1, loss_type)
    
    # 将嵌套的配置扁平化
    training_config = config['training'].copy()
    training_config['optimizer'] = training_config['optimizer']['type']
    
    # 处理调度器配置
    scheduler_config = training_config['scheduler']
    training_config['scheduler'] = scheduler_config['type']
    training_config['step_size'] = scheduler_config.get('step_size', 30)
    training_config['gamma'] = scheduler_config.get('gamma', 0.5)
    training_config['scheduler_patience'] = scheduler_config.get('patience', 10)
    
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    
    # 修改训练配置
    training_config_with_dir = training_config.copy()
    training_config_with_dir['checkpoint_dir'] = os.path.join(
        experiment_dir, 'checkpoints'
    )
    training_config_with_dir['tensorboard'] = config.get('logging', {}).get('tensorboard', False)
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
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)
    
    logger.info("传统有监督训练完成")
    return trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对比学习序列预测模型训练')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据路径（文件或文件夹）')
    parser.add_argument('--experiment_name', type=str, default='contrastive_experiment',
                        help='实验名称')
    parser.add_argument('--recursive', action='store_true',
                        help='递归搜索子文件夹中的CSV文件')
    parser.add_argument('--no_source_column', action='store_true',
                        help='不添加源文件列')
    parser.add_argument('--mode', type=str, choices=['supervised', 'contrastive', 'both'],
                        default=None, help='训练模式：supervised（传统有监督）、contrastive（对比学习）、both（预训练+微调）')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    args = parse_args()
    
    print("=== 对比学习序列预测模型训练 ===")
    print(f"实验名称: {args.experiment_name}")
    print(f"配置文件: {args.config}")
    print(f"数据文件: {args.data_path}")
    
    # 创建实验专用目录
    experiment_dir = os.path.join("results", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"实验目录: {experiment_dir}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定训练模式
    training_mode = args.mode or config.get('training_mode', 'both')
    print(f"训练模式: {training_mode}")
    
    # 设置随机种子
    if 'seed' in config:
        set_seed(config['seed'])
        print(f"设置随机种子: {config['seed']}")
    
    # 打印设备信息
    print_device_info()
    
    # 获取设备
    device = get_device(config.get('device', 'auto'))
    
    # 创建实验日志记录器
    logger = create_experiment_logger(args.experiment_name, experiment_dir)
    logger.info(f"开始实验: {args.experiment_name}")
    logger.info(f"训练模式: {training_mode}")
    
    # 加载和预处理数据
    print("\n1. 加载和预处理数据...")
    
    if os.path.isfile(args.data_path):
        print(f"检测到单个文件: {args.data_path}")
        preprocessor, data = DataPreprocessor.from_config_and_data(
            config, args.data_path)
    elif os.path.isdir(args.data_path):
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
        
        preprocessor, _ = DataPreprocessor.from_config_and_data(
            config, "", data=data)
    else:
        raise FileNotFoundError(f"路径不存在: {args.data_path}")
    
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
    
    # 创建数据加载器
    print("\n3. 创建数据加载器...")
    if training_mode == 'contrastive':
        # 对比学习预训练不需要标签
        batch_size = config.get('contrastive', {}).get('batch_size', 32)
        train_loader = create_data_loader(X_train, batch_size=batch_size, shuffle=True)
        val_loader = create_data_loader(X_val, batch_size=batch_size, shuffle=False)
    else:
        # 有监督训练需要标签
        if training_mode == 'supervised':
            batch_size = config['training']['batch_size']
        else:  # both
            batch_size = config.get('contrastive', {}).get('batch_size', 32)
            
        train_loader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
    
    test_loader = create_data_loader(X_test, y_test, batch_size, shuffle=False)
    
    # 创建模型
    print("\n4. 创建模型...")
    input_size = X_train.shape[2]
    backbone = create_backbone_model(config, input_size, device)
    
    print(f"输入特征维度: {input_size}")
    print(f"Backbone参数数量: {backbone.get_num_parameters()}")
    
    # 根据训练模式执行不同的训练策略
    if training_mode == 'supervised':
        # 传统有监督训练
        print("\n5. 开始传统有监督训练...")
        backbone.to(device)
        trainer = supervised_training(
            backbone, train_loader, val_loader, config,
            experiment_dir, logger, device
        )
        final_model = backbone
        
    elif training_mode == 'contrastive':
        # 仅对比学习预训练
        print("\n5. 开始对比学习预训练...")
        contrastive_config = config.get('contrastive', {})
        model_wrapper = ContrastiveModelWrapper(
            backbone=backbone,
            projection_dim=contrastive_config.get('projection_dim', 128),
            projection_hidden_dim=contrastive_config.get('projection_hidden_dim', 256)
        )
        model_wrapper.to(device)
        
        trainer = contrastive_pretrain(
            model_wrapper, train_loader, val_loader, config,
            experiment_dir, logger, device
        )
        final_model = model_wrapper
        
    elif training_mode == 'both':
        # 对比学习预训练 + 有监督微调
        print("\n5. 开始对比学习预训练...")
        contrastive_config = config.get('contrastive', {})
        model_wrapper = ContrastiveModelWrapper(
            backbone=backbone,
            projection_dim=contrastive_config.get('projection_dim', 128),
            projection_hidden_dim=contrastive_config.get('projection_hidden_dim', 256)
        )
        model_wrapper.to(device)
        
        # 预训练阶段 - 不需要标签
        pretrain_train_loader = create_data_loader(
            X_train, 
            batch_size=contrastive_config.get('batch_size', 32), 
            shuffle=True
        )
        pretrain_val_loader = create_data_loader(
            X_val, 
            batch_size=contrastive_config.get('batch_size', 32), 
            shuffle=False
        )
        
        contrastive_trainer = contrastive_pretrain(
            model_wrapper, pretrain_train_loader, pretrain_val_loader, config,
            experiment_dir, logger, device
        )
        
        # 微调阶段 - 需要标签
        print("\n6. 开始有监督微调...")
        finetune_batch_size = config.get('finetune', {}).get('batch_size', 32)
        finetune_train_loader = create_data_loader(
            X_train, y_train, finetune_batch_size, shuffle=True
        )
        finetune_val_loader = create_data_loader(
            X_val, y_val, finetune_batch_size, shuffle=False
        )
        
        finetune_trainer = supervised_finetune(
            model_wrapper, finetune_train_loader, finetune_val_loader, config,
            experiment_dir, logger, device
        )
        final_model = model_wrapper
        trainer = finetune_trainer
    
    # 测试模型
    print(f"\n{7 if training_mode == 'both' else 6}. 测试模型...")
    logger.info("开始测试模型...")
    
    # 选择要加载的模型
    if training_mode == 'supervised':
        checkpoint_path = os.path.join(
            experiment_dir, 'checkpoints', 'best_model.pth'
        )
    elif training_mode == 'contrastive':
        checkpoint_path = os.path.join(
            experiment_dir, 'pretrain_checkpoints', 'contrastive_latest.pth'
        )
    else:  # both
        checkpoint_path = os.path.join(
            experiment_dir, 'finetune_checkpoints', 'best_finetune_model.pth'
        )
    
    # 加载最佳模型进行测试
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        final_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"加载模型进行测试: {checkpoint_path}")
    
    final_model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    # 选择适当的损失函数
    if training_mode == 'contrastive':
        # 对比学习模式下，使用MSE进行测试评估
        criterion = torch.nn.MSELoss()
    else:
        loss_type = config.get('training', {}).get('loss_function', {}).get('type', 'auto')
        if training_mode == 'both':
            loss_type = config.get('finetune', {}).get('loss_function', {}).get('type', 'auto')
        criterion = create_loss_function('regression', 1, loss_type)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if training_mode == 'contrastive':
                # 对比学习模式，需要使用原始输出
                output = final_model(data, return_projection=False)
            elif isinstance(final_model, ContrastiveModelWrapper):
                # 微调后的模型
                output = final_model(data, return_projection=False)
            else:
                # 传统有监督训练的模型
                output = final_model(data)
            
            if len(output.shape) > 1 and output.shape[1] == 1:
                output = output.squeeze()
            
            test_loss += criterion(output, target).item()
            
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"测试损失: {avg_test_loss:.4f}")
    
    # 保存结果
    print(f"\n{8 if training_mode == 'both' else 7}. 保存模型和结果...")
    
    # 反归一化预测结果
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if hasattr(preprocessor, 'inverse_transform_target'):
        predictions_original = preprocessor.inverse_transform_target(predictions)
        targets_original = preprocessor.inverse_transform_target(targets)
    else:
        predictions_original = predictions
        targets_original = targets
    
    # 计算指标
    from src.utils.metrics import calculate_metrics
    metrics = calculate_metrics(
        targets_original, predictions_original, task_type='regression'
    )
    
    # 保存评估结果
    eval_results = {
        'training_mode': training_mode,
        'test': {
            'test_loss': float(avg_test_loss),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'num_samples': len(targets_original)
        }
    }
    
    eval_results_file = os.path.join(experiment_dir, 'test_results.yaml')
    with open(eval_results_file, 'w', encoding='utf-8') as f:
        yaml.dump(eval_results, f, default_flow_style=False, allow_unicode=True)
    
    # 保存预测结果
    predictions_file = os.path.join(experiment_dir, 'test_predictions.csv')
    base_paths = [os.path.join(args.data_path, 'signals'), '../signals', 'data/signals']
    df_results = preprocessor.get_test_predictions_with_file_data(
        predictions_original, targets_original, base_paths
    )
    df_results.to_csv(predictions_file, index=False)
    
    # 保存配置和预处理器
    import pickle
    preprocessor_path = os.path.join(experiment_dir, "preprocessor.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 保存最终模型
    final_model_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    
    # 保存训练结果
    results = {
        'experiment_name': args.experiment_name,
        'training_mode': training_mode,
        'final_test_loss': float(avg_test_loss),
        'model_type': config['model']['type'],
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'test_samples': X_test.shape[0],
        'input_size': input_size,
    }
    
    if hasattr(trainer, 'current_epoch'):
        results['epochs_trained'] = trainer.current_epoch
    if hasattr(trainer, 'best_val_loss'):
        results['best_validation_loss'] = float(trainer.best_val_loss)
    
    results_path = os.path.join(experiment_dir, "results.yaml")
    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    print("✓ 训练完成！")
    print(f"训练模式: {training_mode}")
    print(f"实验目录: {experiment_dir}")
    print(f"最终模型: {final_model_path}")
    print(f"预处理器: {preprocessor_path}")
    print(f"配置文件: {config_path}")
    print(f"训练结果: {results_path}")
    print(f"测试结果: {eval_results_file}")
    print(f"预测结果: {predictions_file}")
    
    logger.info("对比学习训练流程完成")


if __name__ == '__main__':
    main()
