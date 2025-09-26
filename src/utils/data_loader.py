#!/usr/bin/env python3
"""
数据加载工具 - 支持单文件和文件夹批量加载
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


def find_csv_files(path: str, recursive: bool = True) -> List[str]:
    """
    在指定路径查找所有CSV文件
    
    Args:
        path: 文件或文件夹路径
        recursive: 是否递归搜索子文件夹
        
    Returns:
        CSV文件路径列表
    """
    csv_files = []
    
    if os.path.isfile(path):
        # 如果是单个文件
        if path.lower().endswith('.csv'):
            csv_files.append(path)
        else:
            raise ValueError(f"文件必须是CSV格式: {path}")
    elif os.path.isdir(path):
        # 如果是文件夹
        if recursive:
            # 递归搜索
            pattern = os.path.join(path, '**', '*.csv')
            csv_files = glob.glob(pattern, recursive=True)
        else:
            # 只搜索当前文件夹
            pattern = os.path.join(path, '*.csv')
            csv_files = glob.glob(pattern)
    else:
        raise FileNotFoundError(f"路径不存在: {path}")
    
    # 排序以确保一致的处理顺序
    csv_files.sort()
    
    if not csv_files:
        raise FileNotFoundError(f"在路径 {path} 中未找到CSV文件")
    
    return csv_files


def load_csv_files(csv_files: List[str], 
                   combine_method: str = 'concat',
                   add_source_column: bool = True) -> pd.DataFrame:
    """
    加载多个CSV文件并合并
    
    Args:
        csv_files: CSV文件路径列表
        combine_method: 合并方法 ('concat', 'append')
        add_source_column: 是否添加源文件列
        
    Returns:
        合并后的DataFrame
    """
    if not csv_files:
        raise ValueError("CSV文件列表不能为空")
    
    dataframes = []
    
    for file_path in csv_files:
        try:
            logger.info(f"加载文件: {file_path}")
            df = pd.read_csv(file_path)
            
            if add_source_column:
                # 添加源文件信息
                df['source_file'] = os.path.basename(file_path)
                df['source_path'] = file_path
            
            dataframes.append(df)
            logger.info(f"文件 {file_path} 加载成功，形状: {df.shape}")
            
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            raise
    
    # 合并所有数据
    if combine_method == 'concat':
        combined_df = pd.concat(dataframes, ignore_index=True)
    elif combine_method == 'append':
        combined_df = dataframes[0]
        for df in dataframes[1:]:
            combined_df = combined_df.append(df, ignore_index=True)
    else:
        raise ValueError(f"不支持的合并方法: {combine_method}")
    
    logger.info(f"数据合并完成，最终形状: {combined_df.shape}")
    logger.info(f"合并了 {len(csv_files)} 个文件")
    
    return combined_df


def load_data_from_path(data_path: str, 
                       recursive: bool = True,
                       combine_method: str = 'concat',
                       add_source_column: bool = True) -> pd.DataFrame:
    """
    从路径加载数据（支持单文件和文件夹）
    
    Args:
        data_path: 数据路径（文件或文件夹）
        recursive: 是否递归搜索子文件夹
        combine_method: 合并方法
        add_source_column: 是否添加源文件列
        
    Returns:
        加载的数据DataFrame
    """
    logger.info(f"开始加载数据: {data_path}")
    
    if "complete_dataset.json" in set(os.listdir(data_path)):
        data = load_enhanced_dataset(data_path)
    else:
        # 查找CSV文件
        csv_files = find_csv_files(data_path, recursive=recursive)
    
        logger.info(f"找到 {len(csv_files)} 个CSV文件:")
        for i, file_path in enumerate(csv_files, 1):
            logger.info(f"  {i}. {file_path}")
    
        # 加载并合并数据
        data = load_csv_files(
            csv_files, 
            combine_method=combine_method,
            add_source_column=add_source_column
        )
    
    logger.info(f"数据加载完成: {data.shape}")
    
    return data


def validate_data_consistency(data: pd.DataFrame, 
                            required_columns: Optional[List[str]] = None) -> bool:
    """
    验证数据一致性
    
    Args:
        data: 要验证的数据
        required_columns: 必需的列名列表
        
    Returns:
        是否通过验证
    """
    if data.empty:
        logger.error("数据为空")
        return False
    
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"缺少必需的列: {missing_columns}")
            return False
    
    # 检查是否有重复行
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"发现 {duplicates} 行重复数据")
    
    # 检查缺失值
    missing_info = data.isnull().sum()
    if missing_info.sum() > 0:
        logger.warning("发现缺失值:")
        for col, count in missing_info[missing_info > 0].items():
            logger.warning(f"  {col}: {count} 个缺失值")
    
    logger.info("数据一致性验证完成")
    return True


def get_data_summary(data: pd.DataFrame) -> dict:
    """
    获取数据摘要信息
    
    Args:
        data: 数据DataFrame
        
    Returns:
        数据摘要字典
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
    }
    
    # 如果有源文件信息，统计文件数量
    if 'source_file' in data.columns:
        summary['source_files'] = data['source_file'].nunique()
        summary['files_list'] = data['source_file'].unique().tolist()
    
    return summary


def load_file_data(file_paths: List[str], column_name: str,
                   base_paths: Optional[List[str]] = None) -> List[str]:
    """
    加载文件数据并转换为字符串格式
    
    Args:
        file_paths: 文件路径列表
        column_name: 列名（用于日志记录）
        base_paths: 尝试的基础路径列表
        
    Returns:
        加载的文件数据列表（转换为字符串格式以便保存到CSV）
    """
    if base_paths is None:
        base_paths = ['', 'signals', '../signals', 'data/signals']
    
    loaded_data = []
    successful_loads = 0
    
    for file_path in file_paths:
        data_loaded = False
        
        # 处理空值或无效路径
        if not file_path or pd.isna(file_path):
            loaded_data.append('')
            continue
            
        # 尝试不同的基础路径
        for base_path in base_paths:
            try:
                if base_path:
                    full_path = os.path.join(base_path, file_path)
                else:
                    full_path = file_path
                
                if os.path.exists(full_path):
                    # 根据文件扩展名加载不同类型的文件
                    if full_path.endswith('.npy'):
                        data = np.load(full_path)
                        # 将numpy数组转换为字符串列表格式
                        loaded_data.append(str(data.tolist()))
                        successful_loads += 1
                        data_loaded = True
                        break
                    elif full_path.endswith('.csv'):
                        data = pd.read_csv(full_path)
                        # 将DataFrame转换为JSON字符串格式
                        loaded_data.append(data.to_json(orient='records'))
                        successful_loads += 1
                        data_loaded = True
                        break
                    elif full_path.endswith(('.txt', '.dat')):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            data = f.read()
                        loaded_data.append(data)
                        successful_loads += 1
                        data_loaded = True
                        break
                        
            except Exception as e:
                logger.warning(f"加载文件 {full_path} 失败: {e}")
                continue
        
        if not data_loaded:
            logger.warning(f"无法加载文件: {file_path}")
            loaded_data.append('')
    
    logger.info(
        f"成功加载 {successful_loads}/{len(file_paths)} 个 {column_name} 文件"
    )
    return loaded_data


def process_file_columns(data: pd.DataFrame, 
                        file_columns: List[str],
                        base_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """
    处理DataFrame中的文件列，将文件路径替换为实际文件内容
    
    Args:
        data: 输入DataFrame
        file_columns: 需要处理的文件列名列表
        base_paths: 尝试的基础路径列表
        
    Returns:
        处理后的DataFrame（添加了_data列）
    """
    result_data = data.copy()
    
    for col in file_columns:
        if col in data.columns:
            logger.info(f"正在处理文件列: {col}")
            
            # 加载文件数据
            file_data = load_file_data(
                data[col].values.tolist(), col, base_paths
            )
            
            # 创建新的数据列名
            data_col_name = col.replace('_file', '_data')
            result_data[data_col_name] = file_data
            
            logger.info(f"已将文件列 '{col}' 转换为数据列 '{data_col_name}'")
    
    return result_data


def load_enhanced_dataset(data_dir: str, meta_file: Optional[str] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
    """
    加载 enhanced_toa_generator_refactored.py 生成的数据
    (metadata.csv 或 complete_dataset.json)，自动读取信号文件
    Args:
        data_dir: 数据集目录（如 toa_dataset_enhanced_refactored）
        meta_file: 指定元数据文件（默认自动检测）
        limit: 只加载前 limit 个样本（可选）
    Returns:
        DataFrame: 包含 CIR 波形和 TOA/primary_toa
    """
    import json
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    # 自动检测元数据文件
    if meta_file is None:
        metadata_csv = data_path / "metadata.csv"
        dataset_json = data_path / "complete_dataset.json"
        
        if metadata_csv.exists():
            df = pd.read_csv(str(metadata_csv))
        elif dataset_json.exists():
            # JSON Lines 格式
            with open(str(dataset_json), 'r') as f:
                rows = [json.loads(line) for line in f]
            df = pd.DataFrame(rows)
        else:
            raise FileNotFoundError("未找到元数据文件")
    else:
        meta_path = Path(meta_file)
        if meta_path.suffix == '.csv':
            df = pd.read_csv(str(meta_path))
        else:
            with open(str(meta_path), 'r') as f:
                rows = [json.loads(line) for line in f]
            df = pd.DataFrame(rows)
    
    # 只加载部分样本
    if limit is not None:
        df = df.iloc[:limit]
    
    # 读取 CIR 波形
    cir_list = []
    signals_path = data_path / "signals"
    
    for i, row in df.iterrows():
        cir_path = signals_path / row['rx_signal_file']
        cir = np.load(str(cir_path))
        cir_list.append(cir)
    
    df['CIR'] = cir_list
    df['TOA'] = (df['primary_toa'] * df['sampling_frequency']).astype(float)
    return df