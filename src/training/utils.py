import torch
import torch.nn as nn
from typing import Dict, Any


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config: str = "auto") -> torch.device:
    """
    获取设备
    
    Args:
        device_config: 设备配置 ("auto", "cpu", "cuda", "mps")
        
    Returns:
        torch.device
    """
    if device_config == "auto":
        # 优先级: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用CUDA GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("使用Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("使用CPU")
    elif device_config == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            print("CUDA不可用，回退到CPU")
            device = torch.device("cpu")
    elif device_config == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("使用Apple Silicon GPU (MPS)")
        else:
            print("MPS不可用，回退到CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_loss_function(task_type: str = "classification", 
                        num_classes: int = 1,
                        loss_type: str = "auto") -> nn.Module:
    """
    创建损失函数
    
    Args:
        task_type: 任务类型 ("classification" or "regression")
        num_classes: 类别数量
        loss_type: 损失函数类型 ("auto", "mse", "mae", "huber", "crossentropy", "bce")
        
    Returns:
        损失函数
    """
    if task_type == "classification":
        if loss_type == "auto":
            if num_classes > 2:
                return nn.CrossEntropyLoss()
            else:
                return nn.BCEWithLogitsLoss()
        elif loss_type == "crossentropy":
            return nn.CrossEntropyLoss()
        elif loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            # 分类任务默认使用交叉熵
            if num_classes > 2:
                return nn.CrossEntropyLoss()
            else:
                return nn.BCEWithLogitsLoss()
    else:  # regression
        if loss_type == "auto" or loss_type == "mae":
            return nn.L1Loss()  # MAE损失
        elif loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "huber":
            return nn.HuberLoss()
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            # 回归任务默认使用MAE
            return nn.L1Loss()


def save_training_config(config: Dict[str, Any], filepath: str):
    """保存训练配置"""
    import yaml
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_training_config(filepath: str) -> Dict[str, Any]:
    """加载训练配置"""
    import yaml
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def detect_available_devices():
    """检测可用的设备并返回设备信息"""
    devices = []
    
    # 检测CPU
    devices.append({
        'type': 'cpu',
        'name': 'CPU',
        'available': True
    })
    
    # 检测CUDA
    if torch.cuda.is_available():
        devices.append({
            'type': 'cuda',
            'name': f'CUDA GPU: {torch.cuda.get_device_name()}',
            'available': True,
            'device_count': torch.cuda.device_count(),
            'memory': torch.cuda.get_device_properties(0).total_memory / 1024**3
        })
    else:
        devices.append({
            'type': 'cuda',
            'name': 'CUDA GPU',
            'available': False
        })
    
    # 检测MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append({
            'type': 'mps',
            'name': 'Apple Silicon GPU (MPS)',
            'available': True
        })
    else:
        devices.append({
            'type': 'mps',
            'name': 'Apple Silicon GPU (MPS)',
            'available': False
        })
    
    return devices


def print_device_info():
    """打印设备信息"""
    print("=== 可用设备信息 ===")
    devices = detect_available_devices()
    
    for device in devices:
        status = "✓" if device['available'] else "✗"
        print(f"{status} {device['name']}")
        
        if device['type'] == 'cuda' and device['available']:
            print(f"  - 设备数量: {device['device_count']}")
            print(f"  - 显存: {device['memory']:.1f} GB")
    
    # 推荐设备
    recommended = get_device("auto")
    print(f"\n推荐使用设备: {recommended}")
    print("=" * 30)


def get_optimal_device_config():
    """获取最优设备配置建议"""
    devices = detect_available_devices()
    
    if any(d['type'] == 'cuda' and d['available'] for d in devices):
        return "cuda"
    elif any(d['type'] == 'mps' and d['available'] for d in devices):
        return "mps"
    else:
        return "cpu"
