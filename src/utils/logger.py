import logging
import os
import sys
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO):
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        日志记录器
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_logger(experiment_name: str, log_dir: str = "logs"):
    """
    创建实验日志记录器
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        
    Returns:
        日志记录器
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    return setup_logger(experiment_name, log_file)
