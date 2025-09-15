import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from typing import Dict, Any, Union


def calculate_classification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def calculate_regression_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        指标字典
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     task_type: str = "classification") -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        task_type: 任务类型 ("classification" or "regression")
        
    Returns:
        指标字典
    """
    if task_type == "classification":
        return calculate_classification_metrics(y_true, y_pred)
    else:
        return calculate_regression_metrics(y_true, y_pred)


def print_classification_report(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              target_names: list = None):
    """
    打印分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称列表
    """
    print("\n=== 分类报告 ===")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\n=== 混淆矩阵 ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


def print_regression_report(y_true: np.ndarray, y_pred: np.ndarray):
    """
    打印回归报告
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    """
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    print("\n=== 回归报告 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.6f}")
    
    # 计算残差统计
    residuals = y_true - y_pred
    print(f"\n残差统计:")
    print(f"残差均值: {np.mean(residuals):.6f}")
    print(f"残差标准差: {np.std(residuals):.6f}")
    print(f"残差最大值: {np.max(residuals):.6f}")
    print(f"残差最小值: {np.min(residuals):.6f}")


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self, task_type: str = "classification"):
        """
        初始化指标跟踪器
        
        Args:
            task_type: 任务类型
        """
        self.task_type = task_type
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        更新指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        """
        self.y_true.extend(y_true.flatten())
        self.y_pred.extend(y_pred.flatten())
    
    def compute(self) -> Dict[str, float]:
        """计算当前指标"""
        if not self.y_true:
            return {}
        
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        return calculate_metrics(y_true, y_pred, self.task_type)
    
    def get_best_metric(self, metrics: Dict[str, float]) -> float:
        """
        获取主要指标值
        
        Args:
            metrics: 指标字典
            
        Returns:
            主要指标值
        """
        if self.task_type == "classification":
            return metrics.get('f1', 0.0)
        else:
            return -metrics.get('rmse', float('inf'))  # 负RMSE，越大越好
