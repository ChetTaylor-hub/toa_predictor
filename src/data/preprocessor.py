import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Optional, List, Tuple, Union


class DataPreprocessor:
    """统一的数据预处理器，自动检测和处理序列数据"""
    
    def __init__(self, 
                 input_columns: List[str],
                 target_column: str,
                 sequence_length: Optional[int] = None,
                 max_sequence_length: int = 1000,
                 normalize_features: bool = True,
                 normalize_targets: bool = True,
                 scaler_type: str = "minmax"):
        """
        初始化数据预处理器
        
        Args:
            input_columns: 输入列名列表
            target_column: 目标列名
            sequence_length: 固定序列长度，None表示自动确定
            max_sequence_length: 最大序列长度限制
            normalize_features: 是否标准化特征
            normalize_targets: 是否标准化目标
            scaler_type: 标准化类型 ("standard" or "minmax")
        """
        self.input_columns = input_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.scaler_type = scaler_type
        
        # 内部状态
        self.feature_scaler = None
        self.target_scaler = None
        self.column_types = {}  # 记录每列的数据类型
        self.fitted = False
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
    def _get_scaler(self):
        """获取标准化器"""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化类型: {self.scaler_type}")
    
    def _detect_column_type(self, data: pd.DataFrame, column: str) -> str:
        """
        检测列的数据类型
        
        Args:
            data: 数据DataFrame
            column: 列名
            
        Returns:
            'sequence': 序列数据（字符串表示的数组）
            'numeric': 数值数据
        """
        if column not in data.columns:
            raise ValueError(f"列 '{column}' 不存在于数据中")
        
        # 检查第一个非空值
        sample_values = data[column].dropna().head(5)
        
        for value in sample_values:
            if isinstance(value, (list, np.ndarray)):
                return 'sequence'
            value_str = str(value)
            # 检查是否为字符串表示的数组
            if value_str.startswith('[') and value_str.endswith(']'):
                try:
                    # 尝试解析为数组
                    parsed = ast.literal_eval(value_str)
                    if isinstance(parsed, list) and len(parsed) > 10:  # 长度>10认为是序列
                        return 'sequence'
                except (ValueError, SyntaxError):
                    continue
        
        return 'numeric'
    
    def _parse_sequence_string(self, seq_str) -> np.ndarray:
        """解析序列字符串或直接返回array"""
        if isinstance(seq_str, np.ndarray):
            return seq_str.astype(np.float32)
        if isinstance(seq_str, list):
            return np.array(seq_str, dtype=np.float32)
        try:
            seq_list = ast.literal_eval(seq_str)
            return np.array(seq_list, dtype=np.float32)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法解析序列数据: {str(e)}")
    
    def _process_sequence_column(self, data: pd.DataFrame, column: str) -> np.ndarray:
        """
        处理序列列，兼容array/list/字符串
        
        Args:
            data: 数据DataFrame
            column: 列名
            
        Returns:
            处理后的序列数组，形状为 [N, L]
        """
        sequences = []
        sequence_lengths = []
        
        print(f"处理序列列 '{column}'...")
        
        for i, seq in enumerate(data[column]):
            try:
                seq_array = self._parse_sequence_string(seq)
                sequences.append(seq_array)
                sequence_lengths.append(len(seq_array))
            except Exception as e:
                print(f"跳过第{i}行，解析失败: {str(e)}")
                continue
        
        # 确定序列长度
        if self.sequence_length is None:
            self.sequence_length = min(max(sequence_lengths), self.max_sequence_length)
            print(f"自动确定序列长度: {self.sequence_length}")
        
        # 统一序列长度
        processed_sequences = []
        for seq in sequences:
            if len(seq) >= self.sequence_length:
                # 截断到指定长度
                processed_seq = seq[:self.sequence_length]
            else:
                # 填充到指定长度
                processed_seq = np.zeros(self.sequence_length, dtype=np.float32)
                processed_seq[:len(seq)] = seq

            processed_sequences.append(processed_seq)
        
        result = np.array(processed_sequences)
        print(f"序列处理完成，形状: {result.shape}")
        return result
    
    def _process_numeric_column(self, data: pd.DataFrame, column: str) -> np.ndarray:
        """
        处理数值列
        
        Args:
            data: 数据DataFrame
            column: 列名
            
        Returns:
            数值数组
        """
        return data[column].values.astype(np.float32)
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        拟合预处理器
        
        Args:
            data: 输入数据
            
        Returns:
            self
        """
        print("开始拟合数据预处理器...")
        
        # 检查所需列是否存在
        missing_cols = []
        for col in self.input_columns + [self.target_column]:
            if col not in data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"数据中缺少列: {missing_cols}")
        
        # 检测每列的数据类型
        print("检测列数据类型...")
        for col in self.input_columns:
            col_type = self._detect_column_type(data, col)
            self.column_types[col] = col_type
            print(f"  {col}: {col_type}")
        
        # 处理输入列
        input_features = []
        for col in self.input_columns:
            if self.column_types[col] == 'sequence':
                col_data = self._process_sequence_column(data, col)
                input_features.append(col_data)
            else:
                col_data = self._process_numeric_column(data, col)
                input_features.append(col_data.reshape(-1, 1))
        
        # 合并所有输入特征
        if len(input_features) == 1:
            features = input_features[0]
        else:
            # 多列情况下需要合理合并
            features = np.concatenate(input_features, axis=1)
        
        # 处理目标列
        targets = self._process_numeric_column(data, self.target_column)
        
        # 拟合特征标准化器
        if self.normalize_features:
            print("拟合特征标准化器...")
            self.feature_scaler = self._get_scaler()
            
            if features.ndim == 2:
                # 2D数据：直接拟合
                self.feature_scaler.fit(features)
            elif features.ndim == 3:
                # 3D序列数据：展平后拟合
                n_samples, n_timesteps, n_features = features.shape
                features_flat = features.reshape(-1, n_features)
                self.feature_scaler.fit(features_flat)
        
        # 拟合目标标准化器
        if self.normalize_targets:
            print("拟合目标标准化器...")
            self.target_scaler = self._get_scaler()
            self.target_scaler.fit(targets.reshape(-1, 1))
        
        self.fitted = True
        print("数据预处理器拟合完成")
        return self
    
    def transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换数据
        
        Args:
            data: 输入数据
            
        Returns:
            (X, y): 特征和目标数组
        """
        if not self.fitted:
            raise ValueError("预处理器尚未拟合，请先调用fit方法")
        
        # 处理输入列
        input_features = []
        for col in self.input_columns:
            if self.column_types[col] == 'sequence':
                col_data = self._process_sequence_column(data, col)
                input_features.append(col_data)
            else:
                col_data = self._process_numeric_column(data, col)
                input_features.append(col_data.reshape(-1, 1))
        
        # 合并所有输入特征
        if len(input_features) == 1:
            features = input_features[0]
        else:
            features = np.concatenate(input_features, axis=1)
        
        # 处理目标列
        targets = self._process_numeric_column(data, self.target_column)
        
        # 标准化特征
        if self.normalize_features and self.feature_scaler is not None:
            if features.ndim == 2:
                features = self.feature_scaler.transform(features)
            elif features.ndim == 3:
                n_samples, n_timesteps, n_features = features.shape
                features_flat = features.reshape(-1, n_features)
                features_normalized = self.feature_scaler.transform(features_flat)
                features = features_normalized.reshape(n_samples, n_timesteps, n_features)
        
        # 标准化目标
        if self.normalize_targets and self.target_scaler is not None:
            targets_normalized = self.target_scaler.transform(targets.reshape(-1, 1))
            targets = targets_normalized.flatten()
        
        # 确保输出格式正确
        if features.ndim == 2 and len(self.input_columns) == 1 and self.column_types[self.input_columns[0]] == 'sequence':
            # 单个序列列：转换为3D格式 [N, L, 1]
            features = features.reshape(features.shape[0], features.shape[1], 1)
        
        print(f"转换完成: X={features.shape}, y={targets.shape}")
        return features, targets
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并转换数据"""
        return self.fit(data).transform(data)
    
    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        反转目标值的归一化
        
        Args:
            y_normalized: 归一化的目标值
            
        Returns:
            原始尺度的目标值
        """
        if self.normalize_targets and self.target_scaler is not None:
            if y_normalized.ndim == 1:
                y_reshaped = y_normalized.reshape(-1, 1)
            else:
                y_reshaped = y_normalized
            
            y_original = self.target_scaler.inverse_transform(y_reshaped)
            if y_normalized.ndim == 1:
                return y_original.flatten()
            else:
                return y_original
        else:
            return y_normalized
    
    def split_data(self, 
                   data: pd.DataFrame,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   random_state: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        划分数据集
        
        Args:
            data: 输入数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
            
        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        # 检查比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练、验证和测试集比例之和必须为1")
        
        # 转换数据
        X, y = self.fit_transform(data)
        
        # 生成索引
        indices = np.arange(len(data))
        
        # 第一次划分：分离出测试集
        temp_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=random_state, stratify=None
        )
        
        # 第二次划分：从剩余数据中分离训练集和验证集
        val_size = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            temp_indices, test_size=val_size, random_state=random_state, stratify=None
        )
        
        # 保存索引
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        # 根据索引获取数据
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    @classmethod
    def from_config_and_data(cls, config: dict, data_path: str = "", 
                           data: Optional[pd.DataFrame] = None) -> Tuple['DataPreprocessor', pd.DataFrame]:
        """
        从配置和数据路径或数据创建预处理器
        
        Args:
            config: 配置字典
            data_path: 数据文件路径（当data为None时使用）
            data: 直接传入的数据DataFrame（优先使用）
            
        Returns:
            (预处理器实例, 数据DataFrame)
        """
        # 加载数据
        if data is not None:
            print(f"使用传入的数据: {data.shape}")
            print(f"列名: {list(data.columns)}")
        else:
            if not data_path:
                raise ValueError("必须提供data_path或data参数")
            import pandas as pd
            data = pd.read_csv(data_path)
            print(f"成功加载数据: {data.shape}")
            print(f"列名: {list(data.columns)}")
        
        # 从配置提取参数
        data_config = config.get('data', {})
        
        # 创建预处理器
        preprocessor = cls(
            input_columns=data_config.get('input_columns', ['CIR']),
            target_column=data_config.get('target_column', 'TOA'),
            sequence_length=data_config.get('sequence_length'),
            max_sequence_length=data_config.get('max_sequence_length', 1000),
            normalize_features=data_config.get('normalize_features', True),
            normalize_targets=data_config.get('normalize_targets', True),
            scaler_type=data_config.get('scaler_type', 'minmax')
        )
        
        return preprocessor, data


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    加载CSV数据
    
    Args:
        filepath: CSV文件路径
        
    Returns:
        DataFrame: 加载的数据
    """
    try:
        data = pd.read_csv(filepath)
        print(f"成功加载数据: {data.shape}")
        print(f"列名: {list(data.columns)}")
        return data
    except Exception as e:
        raise FileNotFoundError(f"无法加载数据文件 {filepath}: {str(e)}")


def save_splits(train_data: Tuple[np.ndarray, np.ndarray],
                val_data: Tuple[np.ndarray, np.ndarray],
                test_data: Tuple[np.ndarray, np.ndarray],
                save_dir: str):
    """
    保存数据集划分
    
    Args:
        train_data: 训练集数据
        val_data: 验证集数据
        test_data: 测试集数据
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    print(f"数据集已保存到: {save_dir}")


def load_splits(load_dir: str) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    加载数据集划分
    
    Args:
        load_dir: 加载目录
        
    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    X_train = np.load(os.path.join(load_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(load_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(load_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(load_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(load_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(load_dir, 'y_test.npy'))
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



