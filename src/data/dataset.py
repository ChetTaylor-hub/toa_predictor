import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional


class SequenceDataset(Dataset):
    """序列数据集类"""
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 device: Optional[torch.device] = None):
        """
        初始化序列数据集
        
        Args:
            X: 特征序列 (num_samples, sequence_length, num_features)
            y: 目标值 (num_samples,)
            device: 设备
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)  # 假设是分类任务，使用LongTensor
        
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        return self.X[idx], self.y[idx]
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.X.shape[-1]
    
    def get_sequence_length(self) -> int:
        """获取序列长度"""
        return self.X.shape[1]


class RegressionSequenceDataset(SequenceDataset):
    """回归序列数据集类"""
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 device: Optional[torch.device] = None):
        """
        初始化回归序列数据集
        
        Args:
            X: 特征序列 (num_samples, sequence_length, num_features)
            y: 目标值 (num_samples,)
            device: 设备
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)  # 回归任务使用FloatTensor
        
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)


def create_data_loaders(train_data: Tuple[np.ndarray, np.ndarray],
                       val_data: Tuple[np.ndarray, np.ndarray],
                       test_data: Tuple[np.ndarray, np.ndarray],
                       batch_size: int = 32,
                       task_type: str = "classification",
                       device: Optional[torch.device] = None,
                       shuffle_train: bool = True):
    """
    创建数据加载器
    
    Args:
        train_data: 训练数据 (X, y)
        val_data: 验证数据 (X, y)
        test_data: 测试数据 (X, y)
        batch_size: 批次大小
        task_type: 任务类型 ("classification" or "regression")
        device: 设备
        shuffle_train: 是否打乱训练数据
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # 选择数据集类
    dataset_class = (SequenceDataset if task_type == "classification" 
                    else RegressionSequenceDataset)
    
    # 创建数据集
    train_dataset = dataset_class(*train_data, device=device)
    val_dataset = dataset_class(*val_data, device=device)
    test_dataset = dataset_class(*test_data, device=device)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
