import torch
import torch.nn as nn
import math
from .base_model import BaseModel


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerModel(BaseModel):
    """Transformer模型"""
    
    def __init__(self, 
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 1,
                 dropout: float = 0.1,
                 dim_feedforward: int = 512):
        """
        初始化Transformer模型
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            num_classes: 输出类别数
            dropout: Dropout率
            dim_feedforward: 前馈网络维度
        """
        super(TransformerModel, self).__init__(input_size, num_classes)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        dim_feedforward = input_size * 4
        
        # 输入映射层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, input_size)
            
        Returns:
            输出张量 (batch_size, num_classes)
        """
        # 输入映射
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        encoded = self.transformer_encoder(x)
        
        # 全局平均池化
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(pooled)
        
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers
        })
        return info
