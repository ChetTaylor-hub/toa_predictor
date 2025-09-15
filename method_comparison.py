#!/usr/bin/env python3
"""
深度学习模型 vs 传统方法性能对比分析
"""

import pandas as pd
import numpy as np
import yaml
import torch
import pickle
import os
import sys
import ast
import argparse
from scipy import signal
from scipy.ndimage import maximum_filter1d

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.training.utils import get_device
from src.utils.metrics import calculate_metrics
from src.utils.data_loader import load_data_from_path


class TraditionalToAEstimators:
    """传统ToA估计方法实现"""
    
    def __init__(self, sampling_rate=1.0):
        self.sampling_rate = sampling_rate
    
    def parse_cir_string(self, cir_str: str) -> np.ndarray:
        """解析CIR字符串为numpy数组"""
        try:
            return np.array(ast.literal_eval(cir_str))
        except:
            return np.array([])
    
    def peak_method(self, cir: np.ndarray, noise_threshold: float = 0.1) -> float:
        """Peak方法：寻找超过噪声阈值的第一个峰值"""
        peaks, _ = signal.find_peaks(np.abs(cir), height=noise_threshold)
        
        if len(peaks) == 0:
            above_threshold = np.where(np.abs(cir) > noise_threshold)[0]
            if len(above_threshold) > 0:
                return above_threshold[0] / self.sampling_rate
            else:
                return 0.0
        
        return peaks[0] / self.sampling_rate
    
    def ifp_method(self, cir: np.ndarray, noise_threshold: float = 0.1) -> float:
        """IFP方法：寻找CIR凹凸性改变且超过噪声阈值的第一个点"""
        cir_abs = np.abs(cir)
        if len(cir_abs) < 3:
            return 0.0
            
        # 计算二阶导数来检测凹凸性变化
        second_derivative = np.diff(cir_abs, n=2)
        
        # 寻找凹凸性改变点
        sign_changes = np.diff(np.sign(second_derivative))
        inflection_points = np.where(sign_changes != 0)[0] + 1
        
        # 在拐点中寻找第一个超过噪声阈值的点
        for point in inflection_points:
            if point < len(cir_abs) and cir_abs[point] > noise_threshold:
                return point / self.sampling_rate
        
        # 如果没有找到合适的拐点，回退到Peak方法
        return self.peak_method(cir, noise_threshold)
    
    def lde_method(self, cir: np.ndarray, noise_threshold: float = 0.1,
                   small_window: int = 3, large_window: int = 7, 
                   detection_factor: float = 5.0) -> float:
        """LDE方法：前沿检测"""
        cir_abs = np.abs(cir)
        
        # 应用移动平均滤波
        moving_avg = signal.convolve(cir_abs, np.ones(3)/3, mode='same')
        
        # 应用两个不同窗口大小的移动最大值滤波器
        max_small = maximum_filter1d(moving_avg, size=small_window)
        max_large = maximum_filter1d(moving_avg, size=large_window)
        
        # 寻找检测条件
        for i in range(len(max_small)):
            if (max_small[i] > noise_threshold and 
                max_large[i] > 0 and 
                max_small[i] > max_large[i] * detection_factor):
                return i / self.sampling_rate
        
        # 如果没有找到，回退到Peak方法
        return self.peak_method(cir, noise_threshold)


def evaluate_traditional_methods_on_data(test_data_path: str) -> dict:
    """在真实数据上评估传统方法"""
    
    try:
        # 加载测试数据
        if os.path.isfile(test_data_path):
            data = pd.read_csv(test_data_path)
        elif os.path.isdir(test_data_path):
            data = load_data_from_path(
                test_data_path, recursive=True, add_source_column=False
            )
        else:
            raise FileNotFoundError(f"数据路径不存在: {test_data_path}")
        
        # 检查必要的列
        if 'CIR' not in data.columns or 'TOA' not in data.columns:
            raise ValueError("数据必须包含CIR和TOA列")
        
        estimator = TraditionalToAEstimators()
        
        # 存储所有方法的结果
        results = {
            'Peak': {'predictions': [], 'targets': []},
            'IFP': {'predictions': [], 'targets': []},
            'LDE': {'predictions': [], 'targets': []}
        }
        
        successful_samples = 0
        
        for idx, row in data.iterrows():
            try:
                # 解析CIR数据
                cir = estimator.parse_cir_string(row['CIR'])
                true_toa = row['TOA']
                
                if len(cir) == 0:
                    continue
                
                threshold = 0.5 * max(np.abs(cir))
                # 计算各方法的估计结果
                peak_toa = estimator.peak_method(cir, threshold)
                ifp_toa = estimator.ifp_method(cir, threshold)
                lde_toa = estimator.lde_method(cir, threshold)
                
                # 存储结果
                results['Peak']['predictions'].append(peak_toa)
                results['Peak']['targets'].append(true_toa)
                
                results['IFP']['predictions'].append(ifp_toa)
                results['IFP']['targets'].append(true_toa)
                
                results['LDE']['predictions'].append(lde_toa)
                results['LDE']['targets'].append(true_toa)
                
                successful_samples += 1
                
            except Exception:
                continue
        
        # 计算每种方法的指标
        traditional_results = {}
        
        for method_name, method_data in results.items():
            if len(method_data['predictions']) > 0:
                predictions = np.array(method_data['predictions'])
                targets = np.array(method_data['targets'])
                
                metrics = calculate_metrics(
                    targets, predictions, task_type='regression'
                )
                
                traditional_results[method_name] = {
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'p90_error': np.percentile(
                        np.abs(predictions - targets), 90
                    ),
                    'description': get_method_description(method_name),
                    'samples': len(predictions),
                    'errors': np.abs(predictions - targets)
                }
        
        return traditional_results
        
    except Exception as e:
        print(f"传统方法评估失败: {e}")
        return None


def calculate_cdf_analysis(results_dict: dict) -> dict:
    """计算累积分布函数分析"""
    cdf_analysis = {}
    
    for method_name, method_data in results_dict.items():
        if 'errors' in method_data:
            errors = method_data['errors']
            
            # 计算CDF
            sorted_errors = np.sort(errors)
            n = len(sorted_errors)
            cdf_values = np.arange(1, n + 1) / n
            
            # 计算关键百分位数
            percentiles = [50, 75, 90, 95, 99]
            percentile_values = {}
            for p in percentiles:
                percentile_values[f'p{p}'] = np.percentile(errors, p)
            
            cdf_analysis[method_name] = {
                'sorted_errors': sorted_errors,
                'cdf_values': cdf_values,
                'percentiles': percentile_values,
                'mean_error': np.mean(errors),
                'std_error': np.std(errors)
            }
    
    return cdf_analysis


def print_cdf_summary(cdf_analysis: dict):
    """打印CDF分析摘要，包含完整性能指标"""
    
    print("\nToA估计方法性能对比 - CDF分析:")
    print("=" * 60)
    
    for method_name, analysis in cdf_analysis.items():
        print(f"\n{method_name}方法:")
        print(f"  MAE (平均绝对误差): {analysis['mean_error']:.3f}")
        print(f"  标准差: {analysis['std_error']:.3f}")
        
        percentiles = analysis['percentiles']
        print("  累积分布函数 (CDF) - 关键百分位数:")
        for p_name, p_value in percentiles.items():
            percentage = p_name[1:]  # 去掉'p'前缀
            print(f"    {percentage}%的误差小于等于: {p_value:.3f}")
        print()


def get_method_description(method_name: str) -> str:
    """获取方法描述"""
    descriptions = {
        'Peak': '寻找超过噪声阈值的第一个峰值',
        'IFP': '寻找CIR凹凸性改变且超过阈值的第一个点',
        'LDE': '前沿检测，使用双移动最大值滤波器'
    }
    return descriptions.get(method_name, '未知方法')


def load_and_evaluate_model(
    experiment_dir: str, test_data_path: str = None
) -> dict:
    """优先加载实验结果，必要时进行模型评估"""
    
    try:
        # 1. 优先尝试加载已有的评估结果
        results_file = os.path.join(experiment_dir, 'evaluation_results.yaml')
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                results = yaml.safe_load(f)
            
            # 检查是否有测试集结果
            if 'test' in results and 'metrics' in results['test']:
                test_metrics = results['test']['metrics']
                return {
                    'metrics': {
                        'mae': test_metrics.get('mae', 0),
                        'rmse': test_metrics.get('rmse', 0),
                    },
                    'test_samples': results['test'].get('num_samples', 0),
                    'evaluation_method': 'from_results_file'
                }
        
        # 2. 尝试从测试预测文件计算结果
        pred_file = os.path.join(experiment_dir, 'test_predictions.csv')
        if os.path.exists(pred_file):
            pred_data = pd.read_csv(pred_file)
            if ('target' in pred_data.columns and
                    'prediction' in pred_data.columns):
                targets = pred_data['target'].values
                predictions = pred_data['prediction'].values
                
                metrics = calculate_metrics(
                    targets, predictions, task_type='regression'
                )
                
                return {
                    'metrics': {
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                    },
                    'test_samples': len(targets),
                    'evaluation_method': 'from_predictions_file',
                    'errors': np.abs(predictions - targets)
                }
        
        # 3. 如果提供了测试数据，进行实时模型加载和评估
        if test_data_path and os.path.exists(test_data_path):
            return _load_model_and_predict(experiment_dir, test_data_path)
        
        print("警告: 未找到评估结果文件或测试数据")
        return None
        
    except Exception as e:
        print(f"模型结果加载失败: {e}")
        return None


def _load_model_and_predict(experiment_dir: str, test_data_path: str) -> dict:
    """实际加载模型并进行预测（仅在必要时调用）"""
    
    try:
        # 加载配置
        config_path = os.path.join(experiment_dir, 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载预处理器
        preprocessor_path = os.path.join(experiment_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # 选择模型文件（优先使用最佳模型）
        best_model_path = os.path.join(experiment_dir, 'best_model.pth')
        final_model_path = os.path.join(experiment_dir, 'final_model.pth')
        
        if os.path.exists(best_model_path):
            model_path = best_model_path
        elif os.path.exists(final_model_path):
            model_path = final_model_path
        else:
            raise FileNotFoundError("未找到模型文件")
        
        # 加载测试数据
        test_data = pd.read_csv(test_data_path)
        # 加载测试数据
        test_data = pd.read_csv(test_data_path)
        
        # 预处理测试数据
        X_test, y_test = preprocessor.transform(test_data)
        
        # 创建模型
        device = get_device('auto')
        input_size = X_test.shape[2]
        
        model_type = config['model']['type'].lower()
        model_config = config['model']
        
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=model_config.get('hidden_size', 128),
                num_layers=model_config.get('num_layers', 2),
                num_classes=1,
                dropout=model_config.get('dropout', 0.2),
                bidirectional=model_config.get('bidirectional', True)
            )
        elif model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=model_config.get('hidden_size', 128),
                nhead=model_config.get('nhead', 8),
                num_layers=model_config.get('num_layers', 6),
                num_classes=1,
                dropout=model_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model.to(device)
        
        # 加载模型权重
        if model_path.endswith('best_model.pth'):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 模型推理
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            # 分批处理以避免内存问题
            batch_size = 32
            for i in range(0, len(X_test), batch_size):
                batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
                batch_y = y_test[i:i+batch_size]
                
                output = model(batch_X).squeeze()
                if len(output.shape) == 0:  # 单个样本的情况
                    output = output.unsqueeze(0)
                
                predictions.extend(output.cpu().numpy())
                targets.extend(batch_y)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 反归一化（如果需要）
        if hasattr(preprocessor, 'inverse_transform_target'):
            predictions_orig = preprocessor.inverse_transform_target(
                predictions
            )
            targets_orig = preprocessor.inverse_transform_target(targets)
        else:
            predictions_orig = predictions
            targets_orig = targets
        
        # 计算指标
        metrics = calculate_metrics(
            targets_orig, predictions_orig, task_type='regression'
        )
        
        return {
            'metrics': {
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
            },
            'test_samples': len(targets_orig),
            'evaluation_method': 'live_model_prediction',
            'errors': np.abs(predictions_orig - targets_orig)
        }
        
    except Exception as e:
        print(f"模型加载和预测失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='比较ToA估计方法性能'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='测试数据路径（csv文件或包含csv文件的文件夹）'
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        help='模型目录路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 评估传统方法
    traditional_results = evaluate_traditional_methods_on_data(args.test_data)
    model_results = None
    
    # 评估模型（如果提供）
    if args.experiment_dir:
        model_results = load_and_evaluate_model(
            args.experiment_dir, args.test_data
        )
    
    # 计算和打印CDF分析（包含所有性能指标）
    if traditional_results:
        cdf_analysis = calculate_cdf_analysis(traditional_results)
        if model_results and 'errors' in model_results:
            errors = model_results['errors']
            cdf_analysis['Model'] = {
                'sorted_errors': np.sort(errors),
                'cdf_values': np.arange(1, len(errors) + 1) / len(errors),
                'percentiles': {
                    f'p{p}': np.percentile(errors, p)
                    for p in [50, 75, 90, 95, 99]
                },
                'mean_error': np.mean(errors),
                'std_error': np.std(errors)
            }
        
        print_cdf_summary(cdf_analysis)


if __name__ == "__main__":
    main()
