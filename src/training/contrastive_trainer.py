import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..utils.metrics import calculate_metrics
from ..utils.logger import setup_logger


class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.5):
        """
        初始化对比损失
        
        Args:
            temperature: 温度参数，控制softmax的尖锐度
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        计算对比损失
        z1, z2: [N, D]，同一个 batch 两个增强视图的特征
        """
        N = z1.size(0)
        z1 = F.normalize(z1, dim=1)            # 归一化
        z2 = F.normalize(z2, dim=1)            # 归一化
        z = torch.cat([z1, z2], dim=0)       # [2N, D]

        sim = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]
        mask = torch.eye(2*N, device=z.device).bool()

        # exp 相似度
        exp_sim = torch.exp(sim) * (~mask)

        # 正样本对：i <-> i+N
        pos_sim = torch.exp((z1 * z2).sum(dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2N]

        # 分母：每个样本与所有其他样本的 exp(sim)
        denom = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sim / denom).mean()
        return loss
    
    # def forward(self, features: torch.Tensor, 
    #             labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     """
    #     计算对比损失
        
    #     Args:
    #         features: 特征向量 (batch_size, feature_dim)
    #         labels: 标签 (batch_size,) 如果为None则使用自监督方式
            
    #     Returns:
    #         对比损失
    #     """
    #     batch_size = features.size(0)
        
    #     # 标准化特征
    #     features = F.normalize(features, dim=1)
        
    #     # 计算相似度矩阵
    #     similarity_matrix = torch.matmul(features, features.T) / \
    #         self.temperature
        
    #     # 创建掩码，排除对角线元素（自己与自己的相似度）
    #     mask = torch.eye(batch_size, device=features.device).bool()
        
    #     if labels is not None:
    #         # 有监督对比学习
    #         # 创建正样本掩码（相同标签的样本对）
    #         labels = labels.contiguous().view(-1, 1)
    #         positive_mask = torch.eq(labels, labels.T).float()
    #         positive_mask = positive_mask * (~mask).float()  # 排除对角线
            
    #         # 计算正样本的平均相似度
    #         positive_similarity = similarity_matrix * positive_mask
    #         num_positives = positive_mask.sum(dim=1, keepdim=True)
            
    #         # 避免除零
    #         num_positives = torch.where(
    #             num_positives > 0,
    #             num_positives,
    #             torch.ones_like(num_positives)
    #         )
            
    #         # 对于每个样本，计算与正样本的平均相似度
    #         positive_logits = positive_similarity.sum(dim=1, keepdim=True) / \
    #             num_positives
            
    #         # 使用InfoNCE损失
    #         numerator = torch.exp(positive_logits)
    #         denominator = torch.exp(
    #             similarity_matrix * (~mask).float()
    #         ).sum(dim=1, keepdim=True)
    #         loss = -torch.log(numerator / denominator).mean()
    #     else:
    #         # 自监督对比学习（SimCLR风格）
    #         # 假设batch中相邻的样本是正样本对
    #         # 这里采用简化版本，实际应用中需要根据具体数据增强策略调整
            
    #         # 创建正样本索引对
    #         positive_indices = torch.arange(batch_size, device=features.device)
    #         # 简单策略：将batch分为两半，前半部分与后半部分配对
    #         if batch_size % 2 == 0:
    #             positive_indices = torch.cat([
    #                 positive_indices[batch_size//2:],  # 后半部分
    #                 positive_indices[:batch_size//2]   # 前半部分
    #             ])
    #         else:
    #             # 奇数情况下，最后一个样本与第一个样本配对
    #             positive_indices = torch.cat([
    #                 positive_indices[batch_size//2:-1],  
    #                 positive_indices[:batch_size//2],
    #                 positive_indices[0:1]
    #             ])
            
    #         # 计算正样本的相似度
    #         positive_similarity = torch.sum(
    #             features * features[positive_indices], dim=1
    #         )
            
    #         # 计算与所有其他样本的相似度
    #         all_similarity = similarity_matrix * (~mask).float()
            
    #         # InfoNCE损失
    #         numerator = torch.exp(positive_similarity / self.temperature)
    #         denominator = torch.exp(all_similarity).sum(dim=1)
    #         loss = -torch.log(numerator / denominator).mean()
        
    #     return loss


class InfoNCELoss(nn.Module):
    """
    标准的 InfoNCE 损失函数。
    
    Args:
        temperature (float): 温度系数，用于缩放相似度。默认 0.07。
    """
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        输入：
            z1: Tensor, shape = [batch_size, feature_dim]
            z2: Tensor, shape = [batch_size, feature_dim]
        输出：
            loss: 标量 Tensor
        """

        # 特征归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 正样本相似度 (batch 内同索引)
        pos_sim = torch.sum(z1 * z2, dim=-1) / self.temperature  # [batch_size]

        # 构造所有样本的相似度矩阵（包括正负样本）
        logits = torch.matmul(z1, z2.T) / self.temperature  # [batch_size, batch_size]

        # 构造标签：正样本是对角线
        labels = torch.arange(logits.size(0), device=z1.device)

        # InfoNCE：交叉熵形式
        loss = F.cross_entropy(logits, labels)

        return loss
    

class ContrastiveProjectionHead(nn.Module):
    """对比学习投影头"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 output_dim: int = 128):
        """
        初始化投影头
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(ContrastiveProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.projection(x)


class ContrastiveDataAugmentation:
    """对比学习数据增强"""
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 dropout_rate: float = 0.1,
                 time_shift_ratio: float = 0.1):
        """
        初始化数据增强
        
        Args:
            noise_std: 噪声标准差
            dropout_rate: 随机dropout比率
            time_shift_ratio: 时间偏移比率
        """
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.time_shift_ratio = time_shift_ratio
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def random_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """随机dropout时间步"""
        mask = torch.rand_like(x) > self.dropout_rate
        return x * mask
    
    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """时间偏移"""
        seq_len = x.size(1)
        max_shift = int(seq_len * self.time_shift_ratio)
        
        if max_shift == 0:
            return x
        
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        
        if shift > 0:
            # 右移
            return torch.cat([torch.zeros_like(x[:, :shift]), x[:, :-shift]], dim=1)
        elif shift < 0:
            # 左移
            return torch.cat([x[:, -shift:], torch.zeros_like(x[:, :(-shift)])], dim=1)
        else:
            return x
    
    def augment(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成两个增强视图
        
        Args:
            x: 输入序列 (batch_size, seq_len, feature_dim)
            
        Returns:
            两个增强后的视图
        """
        # 视图1：噪声 + dropout
        view1 = self.add_noise(x)
        view1 = self.random_dropout(view1)
        
        # 视图2：时间偏移 + 噪声
        view2 = self.time_shift(x)
        view2 = self.add_noise(view2)
        
        return view1, view2


class ContrastiveModelWrapper(nn.Module):
    """对比学习模型包装器"""
    
    def __init__(self, 
                 backbone: nn.Module,
                 projection_dim: int = 128,
                 projection_hidden_dim: int = 256):
        """
        初始化对比学习模型
        
        Args:
            backbone: 主干网络（LSTM或Transformer）
            projection_dim: 投影维度
            projection_hidden_dim: 投影隐藏层维度
        """
        super(ContrastiveModelWrapper, self).__init__()
        
        self.backbone = backbone
        
        # 获取主干网络的输出维度
        if hasattr(backbone, 'lstm') and hasattr(backbone, 'fc'):
            # 对于LSTM模型，计算LSTM输出后的维度
            lstm_output_size = backbone.hidden_size * 2 if backbone.bidirectional else backbone.hidden_size
            
            if isinstance(backbone.fc, nn.Sequential):
                # 找到第一个线性层的输出维度作为特征维度
                for layer in backbone.fc:
                    if isinstance(layer, nn.Linear):
                        backbone_output_dim = layer.out_features
                        break
                else:
                    # 如果没找到线性层，使用LSTM输出维度
                    backbone_output_dim = lstm_output_size
            elif isinstance(backbone.fc, nn.Linear):
                backbone_output_dim = backbone.fc.in_features
            else:
                backbone_output_dim = lstm_output_size
        else:
            # 对于其他模型结构
            backbone_output_dim = backbone.num_classes if hasattr(backbone, 'num_classes') else 128
        
        # 投影头
        self.projection_head = ContrastiveProjectionHead(
            input_dim=backbone_output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim
        )
        
        # 保存原始分类/回归头
        self.original_head = backbone.fc if hasattr(backbone, 'fc') else None
        
    def forward(self, x: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列
            return_projection: 是否返回投影特征
            
        Returns:
            如果return_projection=True，返回投影特征
            否则返回原始输出
        """
        # 获取backbone的特征表示
        if hasattr(self.backbone, 'lstm'):
            # LSTM模型
            batch_size = x.size(0)
            h0, c0 = self.backbone._init_hidden(batch_size, x.device)
            lstm_out, (hn, cn) = self.backbone.lstm(x, (h0, c0))
            
            if self.backbone.bidirectional:
                final_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
            else:
                final_hidden = hn[-1]
            
            # 获取投影前的特征（通过第一个线性层）
            if isinstance(self.backbone.fc, nn.Sequential):
                # 只通过第一个dropout和第一个linear层
                features = final_hidden
                for i, layer in enumerate(self.backbone.fc):
                    features = layer(features)
                    # 在第一个线性层之后停止
                    if isinstance(layer, nn.Linear):
                        break
            else:
                features = final_hidden
                
        else:
            # 其他模型结构
            # 这里需要根据具体模型调整
            features = self.backbone(x)
        
        if return_projection:
            # 返回投影特征用于对比学习
            return self.projection_head(features)
        else:
            # 返回原始输出用于下游任务
            if hasattr(self.backbone, 'lstm'):
                # 对于LSTM模型，使用原始的final_hidden通过完整的fc层
                return self.backbone.fc(final_hidden)
            else:
                # 对于其他模型，直接返回backbone的输出
                return self.backbone(x)
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ContrastiveTrainer:
    """对比学习训练器"""
    
    def __init__(self,
                 model: ContrastiveModelWrapper,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: torch.device = torch.device('cpu'),
                 config: Dict[str, Any] = None,
                 logger=None):
        """
        初始化对比学习训练器
        
        Args:
            model: 对比学习模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            device: 设备
            config: 配置
            logger: 日志记录器
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        self.logger = logger or setup_logger("contrastive_trainer")
        
        # 训练配置
        self.learning_rate = float(self.config.get('learning_rate', 0.001))
        self.weight_decay = float(self.config.get('weight_decay', 1e-5))
        self.temperature = float(self.config.get('temperature', 0.5))
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=int(self.config.get('epochs', 100))
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(self.config.get('step_size', 30)),
                gamma=float(self.config.get('gamma', 0.5))
            )
        else:
            self.scheduler = None
        
        # 损失函数
        self.contrastive_loss = InfoNCELoss(temperature=self.temperature)
        
        # 数据增强
        self.data_augmentation = ContrastiveDataAugmentation(
            noise_std=self.config.get('noise_std', 0.01),
            dropout_rate=self.config.get('dropout_rate', 0.1),
            time_shift_ratio=self.config.get('time_shift_ratio', 0.1)
        )
        
        # 训练状态
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        
        # TensorBoard
        if self.config.get('tensorboard', False):
            log_dir = os.path.join(
                self.config.get('log_dir', 'logs'),
                f"contrastive_run_{int(time.time())}"
            )
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # 检查点目录
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'contrastive_checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # 数据增强生成两个视图
            view1, view2 = self.data_augmentation.augment(data)
            
            # 将两个视图合并为一个batch
            # combined_views = torch.cat([view1, view2], dim=0)
            
            # 前向传播获取投影特征
            # projections = self.model(combined_views, return_projection=True)
            projections1 = self.model(view1, return_projection=True)
            projections2 = self.model(view2, return_projection=True)
            
            # 计算对比损失
            loss = self.contrastive_loss(projections1, projections2)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Contrastive Loss: {loss.item():.6f}'
                )
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """验证一个epoch"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                
                # 数据增强
                view1, view2 = self.data_augmentation.augment(data)
                # combined_views = torch.cat([view1, view2], dim=0)
                
                # 获取投影特征
                projections1 = self.model(view1, return_projection=True)
                projections2 = self.model(view2, return_projection=True)

                # 计算损失
                loss = self.contrastive_loss(projections1, projections2)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
        """
        self.logger.info(f"开始对比学习预训练，共 {num_epochs} 个epoch")
        self.logger.info(f"模型参数数量: {self.model.get_num_parameters()}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f'Epoch {self.current_epoch}/{num_epochs} - '
                f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                f'LR: {current_lr:.8f}'
            )
            
            # TensorBoard记录
            if self.writer:
                self.writer.add_scalar('Contrastive/Train_Loss', train_loss, self.current_epoch)
                self.writer.add_scalar('Contrastive/Val_Loss', val_loss, self.current_epoch)
                self.writer.add_scalar('Contrastive/Learning_Rate', current_lr, self.current_epoch)
            
            # 定期保存检查点
            if self.current_epoch % 10 == 0 or self.current_epoch == num_epochs:
                self.save_checkpoint()
        
        self.logger.info("对比学习预训练完成")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存当前检查点
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'contrastive_checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最新模型
        latest_path = os.path.join(self.checkpoint_dir, 'contrastive_latest.pth')
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"保存检查点: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"加载检查点: {checkpoint_path}, Epoch: {self.current_epoch}")


class FineTuneTrainer:
    """微调训练器"""
    
    def __init__(self,
                 contrastive_model: ContrastiveModelWrapper,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 device: torch.device,
                 config: Dict[str, Any],
                 logger=None):
        """
        初始化微调训练器
        
        Args:
            contrastive_model: 预训练的对比学习模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            device: 设备
            config: 配置
            logger: 日志记录器
        """
        self.model = contrastive_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger or setup_logger("finetune_trainer")
        
        # 冻结backbone的选项
        if config.get('freeze_backbone', False):
            self.freeze_backbone()
        
        # 优化器
        learning_rate = float(config.get('finetune_learning_rate', 0.0001))
        weight_decay = float(config.get('weight_decay', 1e-5))
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=int(config.get('scheduler_patience', 5)),
            factor=float(config.get('scheduler_factor', 0.5))
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 检查点目录
        self.checkpoint_dir = config.get('finetune_checkpoint_dir', 'finetune_checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard
        if config.get('tensorboard', False):
            log_dir = os.path.join(
                config.get('log_dir', 'logs'),
                f"finetune_run_{int(time.time())}"
            )
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        # 只训练分类/回归头
        if self.model.original_head is not None:
            for param in self.model.original_head.parameters():
                param.requires_grad = True
        
        self.logger.info("已冻结backbone参数，只训练分类头")
    
    def unfreeze_backbone(self):
        """解冻backbone参数"""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        self.logger.info("已解冻backbone参数")
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播（返回下游任务的输出）
            output = self.model(data, return_projection=False)
            
            # 确保输出形状正确
            if len(output.shape) > 1 and output.shape[1] == 1:
                output = output.squeeze()
            
            loss = self.criterion(output, target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Finetune Loss: {loss.item():.6f}'
                )
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data, return_projection=False)
                
                if len(output.shape) > 1 and output.shape[1] == 1:
                    output = output.squeeze()
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算评估指标
        try:
            metrics = calculate_metrics(
                np.array(all_targets),
                np.array(all_predictions),
                task_type='regression'
            )
        except Exception as e:
            self.logger.warning(f"计算指标失败: {e}")
            metrics = {}
        
        return {'loss': avg_loss, **metrics}
    
    def train(self, num_epochs: int):
        """
        微调训练
        
        Args:
            num_epochs: 训练轮数
        """
        self.logger.info(f"开始微调训练，共 {num_epochs} 个epoch")
        
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
            self.scheduler.step(val_loss)
            
            # 记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f'Epoch {self.current_epoch}/{num_epochs} - '
                f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                f'LR: {current_lr:.8f}'
            )
            
            # TensorBoard记录
            if self.writer:
                self.writer.add_scalar('Finetune/Train_Loss', train_loss, self.current_epoch)
                self.writer.add_scalar('Finetune/Val_Loss', val_loss, self.current_epoch)
                self.writer.add_scalar('Finetune/Learning_Rate', current_lr, self.current_epoch)
                
                for metric_name, metric_value in val_results.items():
                    if metric_name != 'loss':
                        self.writer.add_scalar(f'Finetune/{metric_name}',
                                             metric_value, self.current_epoch)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                self.logger.info(f"保存最佳微调模型，验证损失: {val_loss:.6f}")
            
            # 定期保存检查点
            if self.current_epoch % 10 == 0:
                self.save_checkpoint(is_best=False)
        
        self.logger.info("微调训练完成")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
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
            f'finetune_checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_finetune_model.pth')
            torch.save(checkpoint, best_path)
