import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """基础模型抽象类"""
    
    def __init__(self, input_size: int, num_classes: int):
        """
        初始化基础模型
        
        Args:
            input_size: 输入特征维度
            num_classes: 输出类别数（分类）或1（回归）
        """
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': self.__class__.__name__,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_parameters()
        }
    
    def save_model(self, filepath: str, epoch: int = None, 
                   optimizer_state: Dict = None, 
                   best_metric: float = None):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            epoch: 训练轮次
            optimizer_state: 优化器状态
            best_metric: 最佳指标
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_model_info(),
            'epoch': epoch,
            'best_metric': best_metric
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: torch.device = None):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            device: 设备
            
        Returns:
            加载的模型和检查点信息
        """
        if device is None:
            device = torch.device('cpu')
            
        checkpoint = torch.load(filepath, map_location=device)
        model_config = checkpoint['model_config']
        
        # 根据模型类型创建模型实例
        # 这里需要根据实际的模型类进行调整
        model = cls(
            input_size=model_config['input_size'],
            num_classes=model_config['num_classes']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint
