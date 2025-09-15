import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from typing import Dict, Any, Optional, Callable
from ..utils.metrics import calculate_metrics
from ..utils.logger import setup_logger


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0,
                 mode: str = 'min'):
        """
        初始化早停机制
        
        Args:
            patience: 耐心值
            min_delta: 最小改进量
            mode: 模式 ('min' 或 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数
            
        Returns:
            是否早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 config: Dict[str, Any],
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 logger=None):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            config: 配置字典
            scheduler: 学习率调度器
            logger: 日志记录器
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger or setup_logger("trainer")
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # 早停
        if config.get('early_stopping', False):
            self.early_stopping = EarlyStopping(
                patience=config.get('patience', 15),
                mode='min'
            )
        else:
            self.early_stopping = None
            
        # TensorBoard
        if config.get('tensorboard', False):
            log_dir = os.path.join(config.get('log_dir', 'logs'), 
                                 f"run_{int(time.time())}")
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            
        # 检查点目录
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 确保输出和目标的形状匹配
            if len(output.shape) > 1 and output.shape[1] == 1:
                output = output.squeeze()  # 移除最后一个维度
            
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 按样本数累计损失
            total_loss += loss.item()
            
            # 记录训练进度
            if batch_idx % 100 == 0 and batch_idx > 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.6f}'
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                # 确保输出和目标的形状匹配
                if len(output.shape) > 1 and output.shape[1] == 1:
                    output = output.squeeze()  # 移除最后一个维度
                
                loss = self.criterion(output, target)
                
                # 按样本数累计损失
                total_loss += loss.item()
                
                # 收集预测和真实值
                if len(output.shape) > 1 and output.shape[1] > 1:
                    # 分类任务
                    predictions = torch.argmax(output, dim=1)
                else:
                    # 回归任务
                    predictions = output
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        # 计算评估指标
        # 对于连续值，总是当作回归任务处理
        task_type = 'regression'
        try:
            metrics = calculate_metrics(
                np.array(all_targets),
                np.array(all_predictions),
                task_type=task_type
            )
        except Exception as e:
            # 如果计算指标失败，只返回损失
            self.logger.warning(f"计算指标失败: {e}")
            metrics = {}
        
        return {'loss': avg_loss, **metrics}
    
    def train(self, num_epochs: int):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
        """
        self.logger.info(f"开始训练，共 {num_epochs} 个epoch")
        self.logger.info(f"模型参数数量: {self.model.get_num_parameters()}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_results = self.validate_epoch()
            val_loss = val_results['loss']
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录指标
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f'Epoch {self.current_epoch}/{num_epochs} - '
                f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                f'LR: {current_lr:.8f}'
            )
            
            # TensorBoard记录
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, self.current_epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, self.current_epoch)
                
                for metric_name, metric_value in val_results.items():
                    if metric_name != 'loss':
                        self.writer.add_scalar(f'Metrics/{metric_name}', 
                                             metric_value, self.current_epoch)
            
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                self.logger.info(f"保存最佳模型，验证损失: {val_loss:.6f}")
            
            # 早停检查
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    self.logger.info(f"早停触发，在第 {self.current_epoch} epoch")
                    break
            
            # 定期保存检查点
            if self.current_epoch % 10 == 0:
                self.save_checkpoint(is_best=False)
        
        self.logger.info("训练完成")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存检查点
        
        Args:
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存当前检查点
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"加载检查点: {checkpoint_path}, Epoch: {self.current_epoch}")


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        config: 配置
        
    Returns:
        优化器
    """
    optimizer_type = config.get('optimizer', 'adam').lower()
    learning_rate = float(config.get('learning_rate', 0.001))
    weight_decay = float(config.get('weight_decay', 1e-5))
    
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = float(config.get('momentum', 0.9))
        return optim.SGD(model.parameters(), lr=learning_rate, 
                        momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer, 
                    config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置
        
    Returns:
        学习率调度器
    """
    scheduler_type = config.get('scheduler', None)
    
    if scheduler_type is None:
        return None
    elif scheduler_type == 'step':
        step_size = int(config.get('step_size', 30))
        gamma = float(config.get('gamma', 0.5))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = int(config.get('epochs', 100))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'plateau':
        patience = int(config.get('scheduler_patience', 10))
        factor = float(config.get('scheduler_factor', 0.5))
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
