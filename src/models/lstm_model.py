import torch
import torch.nn as nn
from .base_model import BaseModel
from typing import Optional


class LSTMModel(BaseModel):
    """LSTM模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            num_classes: 输出类别数
            dropout: Dropout率
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__(input_size, num_classes)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 计算LSTM输出维度
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes),
            # 使用exp确保输出为正值（适用于回归任务如时间序列预测）
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, input_size)
            
        Returns:
            输出张量 (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0, c0 = self._init_hidden(batch_size, x.device)
        
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 对于双向LSTM，连接前向和后向的最后隐藏状态
            final_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            final_hidden = hn[-1]
        
        # 全连接层
        output = self.fc(final_hidden)

        return output
    
    def _init_hidden(self, batch_size: int, device: torch.device):
        """
        初始化隐藏状态
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            (h0, c0): 初始隐藏状态和细胞状态
        """
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(
            self.num_layers * num_directions, 
            batch_size, 
            self.hidden_size,
            device=device
        )
        c0 = torch.zeros(
            self.num_layers * num_directions, 
            batch_size, 
            self.hidden_size,
            device=device
        )
        
        return h0, c0
    
    def get_model_info(self):
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout
        })
        return info


class SimpleLSTMModel(BaseModel):
    """简化版LSTM模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_classes: int = 1):
        """
        初始化简化版LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_classes: 输出类别数
        """
        super(SimpleLSTMModel, self).__init__(input_size, num_classes)
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        output = self.fc(hn[-1])
        
        return output
