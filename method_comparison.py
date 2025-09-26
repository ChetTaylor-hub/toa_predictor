#!/usr/bin/env python3
"""
深度学习模型 vs 传统方法性能对比分析

该模块提供了多种ToA（到达时间）估计方法的性能对比功能，包括：
- 传统方法：Peak、IFP、LDE、匹配滤波
- 深度学习方法：LSTM、Transformer等

主要功能：
1. 实现多种传统ToA估计算法
2. 加载和评估深度学习模型
3. 生成详细的性能对比报告和CDF分析
"""

import ast
import argparse
import os
import pickle
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import signal
from scipy.ndimage import maximum_filter1d

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.models.lstm_model import LSTMModel
    from src.models.transformer_model import TransformerModel
    from src.training.utils import get_device
    from src.utils.metrics import calculate_metrics
    from src.utils.data_loader import load_data_from_path
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt

    def plot_array(arr, title="array trend"):
        plt.figure()
        plt.plot(arr)
        plt.title(title)
        plt.xlabel("index")
        plt.ylabel("value")
        plt.grid(True)
        plt.show()
except ImportError:
    def plot_array(arr, title="array trend"):
        print(f"Plotting {title}: matplotlib not available")


# 常量定义
class Config:
    """配置常量"""
    DEFAULT_SAMPLING_RATE = 1.0
    DEFAULT_NOISE_THRESHOLD = 0.1
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_K_SIGMA = 4.0
    DEFAULT_CONSECUTIVE = 3
    
    # 噪声阈值系数
    PEAK_NOISE_FACTOR = 0.6
    IFP_NOISE_FACTOR = 2.0
    IPS_NOISE_FACTOR = 0.4
    MATCHED_FILTER_NOISE_FACTOR = 0.8
    LDE_RATIO_FACTOR = 1.1
    
    # 文件路径
    POSSIBLE_SIGNAL_PATHS = [
        'signals',
        '../signals', 
        'data/signals'
    ]
    
    # CDF分析的百分位数
    CDF_PERCENTILES = [50, 75, 90, 95, 99]


class NoiseEstimator:
    """噪声估计工具类"""
    
    @staticmethod
    def estimate_noise(cir: np.ndarray,
                       noise_window: Optional[Tuple[int, int]] = None,
                       method: str = 'std') -> Tuple[float, float]:
        """
        估计噪声基线与噪声标准差
        
        Args:
            cir: 1D numpy array
            noise_window: tuple (start_idx, end_idx) for noise-only region
            method: 'std' or 'mad' (median absolute deviation)
            
        Returns:
            (mean_noise, sigma_noise)
        """
        n = len(cir)
        if noise_window is None:
            start = int(n * 0.5)
            noise_seg = cir[start:]
        else:
            s, e = noise_window
            noise_seg = cir[s:e]
        
        mean_noise = float(np.mean(noise_seg))
        
        if method == 'mad':
            mad = np.median(np.abs(noise_seg - np.median(noise_seg)))
            sigma = 1.4826 * mad
        else:
            sigma = float(np.std(noise_seg, ddof=1))
            
        return mean_noise, sigma


class CIRParser:
    """CIR字符串解析器"""
    
    @staticmethod
    def parse_cir_string(cir_str: str) -> np.ndarray:
        """
        解析CIR字符串为numpy数组
        
        支持多种格式：
        1. 标准Python列表格式: "[1.0, 2.0, 3.0]"
        2. 空格分隔格式: "1.0 2.0 3.0"
        3. 科学计数法格式
        
        Args:
            cir_str: CIR字符串
            
        Returns:
            解析后的numpy数组，解析失败返回空数组
        """
        if not cir_str or cir_str.isspace():
            return np.array([])
            
        parsing_methods = [
            CIRParser._parse_with_literal_eval,
            CIRParser._parse_with_split,
            CIRParser._parse_with_fromstring
        ]
        
        for method in parsing_methods:
            try:
                result = method(cir_str)
                if len(result) > 0:
                    return result
            except Exception:
                continue
                
        print(f"警告: 无法解析CIR字符串: {cir_str[:100]}...")
        return np.array([])
    
    @staticmethod
    def _parse_with_literal_eval(cir_str: str) -> np.ndarray:
        """使用ast.literal_eval解析"""
        return np.array(ast.literal_eval(cir_str))
    
    @staticmethod
    def _parse_with_split(cir_str: str) -> np.ndarray:
        """使用字符串分割解析"""
        clean_str = cir_str.strip('[]')
        if clean_str:
            elements = clean_str.split()
            return np.array([float(x) for x in elements])
        return np.array([])
    
    @staticmethod
    def _parse_with_fromstring(cir_str: str) -> np.ndarray:
        """使用numpy.fromstring解析"""
        clean_str = cir_str.strip('[]')
        return np.fromstring(clean_str, sep=' ', dtype=float)


class TraditionalToAEstimators:
    """传统ToA估计方法实现"""
    
    def __init__(self, sampling_rate: float = Config.DEFAULT_SAMPLING_RATE):
        """
        初始化估计器
        
        Args:
            sampling_rate: 采样率 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.cir_parser = CIRParser()
        self.noise_estimator = NoiseEstimator()
    
    def parse_cir_string(self, cir_str: str) -> np.ndarray:
        """解析CIR字符串（委托给CIRParser）"""
        return self.cir_parser.parse_cir_string(cir_str)
    
    def peak_method(self, cir: np.ndarray, 
                   noise_threshold: float = Config.DEFAULT_NOISE_THRESHOLD) -> float:
        """
        Peak方法：寻找超过噪声阈值的第一个峰值
        
        Args:
            cir: 信道冲激响应
            noise_threshold: 噪声阈值
            
        Returns:
            估计的TOA值（秒）
        """
        if len(cir) == 0:
            return 0.0
            
        cir_abs = np.abs(cir)
        # peaks, _ = signal.find_peaks(cir_abs, height=noise_threshold)
        # 找到峰值的索引
        peaks = np.argmax(cir_abs)

        # if len(peaks) == 0:
        #     # 如果没有找到峰值，查找第一个超过阈值的点
        #     above_threshold = np.where(cir_abs > noise_threshold)[0]
        #     if len(above_threshold) > 0:
        #         return above_threshold[0] / self.sampling_rate
        #     else:
        #         return 0.0

        return peaks / self.sampling_rate

    def ifp_method(self, cir: np.ndarray,
                   tx_signal_data: str = "",
                   upsample_factor: int = 8,
                   search_window: int = 100,
                   noise_threshold: float = Config.DEFAULT_NOISE_THRESHOLD
                   ) -> Optional[float]:
        """
        IFP（拐点）方法：基于相关峰值和反转点的TOA估计（DEVICE AND METHOD FOR  DETERMINING ATIME OF ARRIVAL OFA RECEIVE SEQUENCE）
        
        使用发射信号与CIR进行互相关，在主峰前搜索梯度最大的拐点位置
        
        Args:
            cir: 信道冲激响应
            tx_signal_data: 发射信号数据字符串
            upsample_factor: 上采样因子，提升分辨率
            search_window: 在主峰前搜索拐点的窗口大小（样本数）
            noise_threshold: 噪声阈值（用于验证检测结果）
            
        Returns:
            估计的TOA值（秒），如果未找到则返回None
        """
        if len(cir) < 3:
            return 0.0
            
        # Step 1: 获取发射信号
        tx_signal = self.load_tx_signal_from_data(tx_signal_data)
        if len(tx_signal) == 0:
            # 如果没有发射信号，回退到简单峰值检测
            return self.peak_method(cir, noise_threshold)
        
        # Step 2: 计算实数互相关
        cir_real = np.real(cir).astype(np.float64)
        tx_real = np.real(tx_signal).astype(np.float64)
        
        # 执行实数互相关
        correlation = np.correlate(cir_real, tx_real, mode='full')
        correlation_mag = np.abs(correlation).astype(np.float64)
        
        # Step 3: 上采样相关函数（提升分辨率）
        if upsample_factor > 1:
            from scipy.signal import resample
            try:
                new_length = len(correlation_mag) * upsample_factor
                correlation_mag = resample(correlation_mag, new_length)
            except Exception:
                # 如果resample失败，使用线性插值
                from scipy.interpolate import interp1d
                x_old = np.arange(len(correlation_mag))
                new_length = len(correlation_mag) * upsample_factor
                x_new = np.linspace(0, len(correlation_mag) - 1, new_length)
                f = interp1d(x_old, correlation_mag, kind='linear',
                             bounds_error=False, fill_value=0)
                correlation_mag = f(x_new)
        
        # Step 4: 找到主峰位置
        peak_idx = np.argmax(correlation_mag)
        peak_value = correlation_mag[peak_idx]
        
        # 检查峰值是否显著
        if peak_value < noise_threshold:
            return self.peak_method(cir, noise_threshold)
        
        # Step 5: 在主峰前的区间内，寻找拐点（梯度最大的位置）
        start_idx = max(0, peak_idx - search_window * upsample_factor)
        window = correlation_mag[start_idx:peak_idx]
        
        if len(window) > 2:
            # 计算一阶导数（梯度）
            grad = np.gradient(window)
            
            # 找最大梯度位置（对应拐点）
            if len(grad) > 0:
                rev_rel_idx = np.argmax(grad)
                rev_idx = start_idx + rev_rel_idx
                
                # 转换为延迟索引（考虑互相关的索引偏移）
                delay_idx = rev_idx - (len(tx_signal) - 1) * upsample_factor
                
                # 转换回原始采样率的索引
                toa_idx = delay_idx / upsample_factor
            else:
                # 退化为主峰
                delay_idx = peak_idx - (len(tx_signal) - 1) * upsample_factor
                toa_idx = delay_idx / upsample_factor
        else:
            # 如果搜索窗口太小，退化为主峰
            delay_idx = peak_idx - (len(tx_signal) - 1) * upsample_factor
            toa_idx = delay_idx / upsample_factor
        
        # 验证检测结果的合理性
        if toa_idx < 0:
            toa_idx = 0
        elif toa_idx >= len(cir):
            return self.peak_method(cir, noise_threshold)
            
        return toa_idx / self.sampling_rate
    
    # def lde_method(self,
    #                cir: np.ndarray,
    #                win_small: int = 8,
    #                win_large: int = 500,
    #                ratio_threshold: Optional[float] = None,
    #                smooth_window: int = 5
    #                ) -> Optional[float]:
    #     """
    #     LDE（Leading Edge Detection）前沿检测方法 - Max-Ratio 实现（The Future of the Operating Room: Surgical Planning and Navigation using High Accuracy Ultra-Wideband Positioning and Advanced Bone Measurement）

    #     使用最大值滤波器和比值阈值来检测信号前沿
        
    #     Args:
    #         cir: 信道冲激响应
    #         win_small: 小窗口长度
    #         win_large: 大窗口长度
    #         ratio_threshold: 比值阈值，用于判定前沿
    #         smooth_window: 平滑窗口长度
            
    #     Returns:
    #         估计的TOA值（秒），如果未找到则返回None
    #     """
    #     if len(cir) == 0:
    #         return None
            
    #     # 取绝对值并转换为浮点数
    #     signal = np.abs(np.asarray(cir).astype(float))
    #     n = len(signal)
        
    #     # 确保窗口大小不超过信号长度
    #     win_small = min(win_small, n // 4)
    #     win_large = min(win_large, n)
    #     smooth_window = min(smooth_window, n // 10)
        
    #     if win_small < 1 or win_large < 1 or smooth_window < 1:
    #         # 信号太短，回退到简单峰值检测
    #         return self.peak_method(cir)
        
    #     # Step 1: 平滑处理（移动平均）
    #     if smooth_window > 1:
    #         smoothing_kernel = np.ones(smooth_window) / smooth_window
    #         y = np.convolve(signal, smoothing_kernel, mode="same")
    #     else:
    #         y = signal.copy()
        
    #     # Step 2: 最大值滤波
    #     max_small = maximum_filter1d(y, size=win_small)
    #     max_large = maximum_filter1d(y, size=win_large)
        
    #     # Step 3: 计算比值，避免除零
    #     ratio = np.divide(max_large, max_small + 1e-12)
        
    #     # Step 4: 找到比值首次小于阈值的位置
    #     idx = np.where(ratio < ratio_threshold)[0]
        
    #     if len(idx) == 0:
    #         return None  # 未检测到前沿
            
    #     leading_edge_index = idx[0]
        
    #     # 转换为时间（秒）
    #     return leading_edge_index / self.sampling_rate
    
    def lde_method(self,
                cir,
                threshold=None,         # 绝对阈值（如果指定）或 None
                k_sigma=1.0,            # 若 threshold 为 None，则阈值 = mean_noise + k_sigma * sigma_noise
                noise_window=None,
                consecutive=1,          # 需要连续多少个样点高于阈值才认为触发（>=1）
                min_index=0,            # 忽略 min_index 之前的样点
                max_index=None,         # 忽略 max_index 之后的样点
                refine_subsample=False  # 是否对跨阈点做线性插值提升精度
                ):
        """
        对单个 CIR（采样波形）执行 LDE（Leading Edge Detection）
        返回: crossing_index (样点索引，可能是小数 if refined)
        说明:
        - LDE: 找到第一个满足 `样点 > 阈值` 且满足 consecutive 连续条件的样点。
        - refine_subsample: 在 first crossing 的样点与其前一样点之间做线性插值得到更精确到达时间。
        """
        cir = np.asarray(cir).astype(float)
        n = len(cir)
        if max_index is None:
            max_index = n - 1
        # 估计噪声与阈值
        mean_noise, sigma_noise = _estimate_noise(cir, noise_window=noise_window)
        if threshold is None:
            thresh = mean_noise + k_sigma * sigma_noise
        else:
            thresh = float(threshold)

        # 搜索区域限制
        i = max(min_index, 0)
        end_i = min(max_index, n - 1)

        # 连续判定：寻找第一个满足 consecutive 个连续样点 > threshold
        consec_count = 0
        first_cross_idx = None
        for idx in range(i, end_i + 1):
            if cir[idx] > thresh:
                consec_count += 1
                if consec_count >= consecutive:
                    first_cross_idx = idx - (consec_count - 1)  # 第一个满足的样点索引
                    break
            else:
                consec_count = 0

        if first_cross_idx is None:
            return None

        # 亚采样线性插值（在first_cross_idx与其前一样点之间）
        if refine_subsample and first_cross_idx > 0:
            y0 = cir[first_cross_idx - 1]
            y1 = cir[first_cross_idx]
            # 若 y1 == y0（非常罕见），则直接返回整数索引
            if y1 == y0:
                frac = 0.0
            else:
                # 线性插值求 crossing fraction f in [0,1] such that y0 + f*(y1-y0) = thresh
                frac = (thresh - y0) / (y1 - y0)
                # 截断到 [0,1]
                frac = max(0.0, min(1.0, frac))
            crossing_index = (first_cross_idx - 1) + frac
        else:
            crossing_index = float(first_cross_idx)

        return crossing_index

    
    def ips_method(self, cir: np.ndarray, 
                   tx_signal_data: str = "",
                   max_iter: int = 10,
                   energy_thresh: float = 0.01,
                   noise_threshold: float = Config.DEFAULT_NOISE_THRESHOLD
                   ) -> Optional[float]:
        """
        IPS（Iterative Peak Subtraction）方法：迭代峰值减除TOA估计
        
        通过迭代减除最强峰值对应的模板信号，逐步消除多径分量，
        最终找到最早到达的直视路径(LOS)
        
        Args:
            cir: 信道冲激响应
            tx_signal_data: 发射信号数据字符串（作为脉冲模板）
            max_iter: 最大迭代次数
            energy_thresh: 残差信号能量阈值（相对初始信号能量）
            noise_threshold: 噪声阈值（用于验证检测结果）
            
        Returns:
            估计的TOA值（秒），如果未找到则返回None
        """
        if len(cir) < 3:
            return 0.0
            
        # Step 1: 获取脉冲模板
        template = self.load_tx_signal_from_data(tx_signal_data)
        if len(template) == 0:
            # 如果没有发射信号模板，回退到峰值检测
            return self.peak_method(cir, noise_threshold)
        
        # Step 2: 转换为实数信号
        signal = np.real(cir).astype(np.float64)
        template = np.real(template).astype(np.float64)
        
        # Step 3: 执行迭代峰值减除
        los_index, peaks_found, residual = self._iterative_peak_subtraction(
            signal, template, max_iter, energy_thresh
        )
        
        # Step 4: 验证检测结果
        if los_index is None or los_index < 0 or los_index >= len(cir):
            return None
            
        # 检查LOS峰值是否显著
        if abs(signal[los_index]) < noise_threshold:
            return None
            
        return los_index / self.sampling_rate
    
    def _iterative_peak_subtraction(self, signal: np.ndarray, 
                                   template: np.ndarray,
                                   max_iter: int = 10,
                                   energy_thresh: float = 0.01
                                   ) -> tuple:
        """
        Iterative Peak Subtraction (IPS) 核心实现
        
        参数:
            signal : np.ndarray
                输入信号 (1D array)
            template : np.ndarray
                脉冲模板（与信号同采样率）
            max_iter : int
                最大迭代次数
            energy_thresh : float
                残差信号能量阈值（相对初始信号能量）
        
        返回:
            los_index : int
                检测到的直达波 (LOS path) 索引
            peaks_found : list
                每次迭代找到的峰索引
            residual : np.ndarray
                最终残差信号
        """
        residual = signal.copy()
        peaks_found = []
        init_energy = np.sum(residual**2)
        
        if init_energy == 0:
            return None, [], residual
        
        for i in range(max_iter):
            # Step 1: 找最大峰值
            peak_idx = np.argmax(np.abs(residual))
            peak_val = residual[peak_idx]
            peaks_found.append(peak_idx)
            
            # Step 2: 构造对齐模板 (zero-padded)
            aligned_template = np.zeros_like(residual)
            start = max(0, peak_idx - len(template)//2)
            end = min(len(residual), start + len(template))
            
            temp_start = max(0, len(template)//2 - peak_idx)
            temp_end = temp_start + (end - start)
            
            if temp_end <= len(template):
                aligned_template[start:end] = template[temp_start:temp_end]
            
            # Step 3: 估计缩放系数并减去
            template_max = np.max(np.abs(aligned_template))
            if template_max > 0:
                scale = peak_val / template_max
                residual = residual - scale * aligned_template
            
            # Step 4: 判断能量终止条件
            current_energy = np.sum(residual**2)
            if current_energy < energy_thresh * init_energy:
                break
        
        # 输出最早的峰作为LOS
        if len(peaks_found) > 0:
            los_index = min(peaks_found)
            return los_index, peaks_found, residual
        else:
            return None, [], residual

    
    def matched_filter_method(self, cir: np.ndarray, tx_signal_data: str,
                             noise_threshold: float = Config.MATCHED_FILTER_NOISE_FACTOR) -> Optional[float]:
        """
        实数匹配滤波+峰值检测方法
        
        使用发射信号作为模板，通过实数互相关找到最佳匹配位置
        
        Args:
            cir: 接收信号（CIR）
            tx_signal_data: 发射信号数据字符串
            noise_threshold: 噪声阈值系数
            
        Returns:
            估计的TOA值（秒）
        """
        # 从字符串解析发射信号
        tx_signal = self.load_tx_signal_from_data(tx_signal_data)
        
        if len(cir) == 0 or len(tx_signal) == 0:
            return 0.0
        
        # 转换为实数信号
        cir_real = np.real(cir).astype(np.float64)
        tx_real = tx_signal.astype(np.float64)  # 已经是实数
        
        # 取绝对值并归一化发射信号
        tx_norm = np.abs(tx_real) / max(np.abs(tx_real))
        
        # 执行实数匹配滤波
        correlation = np.correlate(cir_real, tx_norm, mode='full')
        
        # 检测峰值
        peak_delay = self._detect_real_correlation_peak(
            correlation, tx_real, noise_threshold
        )
        
        if peak_delay is None:
            return None
        
        return peak_delay / self.sampling_rate
    
    def _normalize_real_signal(self, signal: np.ndarray) -> np.ndarray:
        """实数信号能量归一化"""
        signal = np.asarray(signal, dtype=np.float64)
        energy = np.sum(signal**2)
        if energy > 0:
            return signal / np.sqrt(energy)
        return signal
    
    def _detect_real_correlation_peak(self, correlation: np.ndarray,
                                      tx_signal: np.ndarray,
                                      noise_threshold: float) -> Optional[int]:
        """
        检测实数相关函数中的峰值
        
        Args:
            correlation: 实数相关函数
            tx_signal: 发射信号（用于计算索引偏移）
            noise_threshold: 噪声阈值系数
            
        Returns:
            延迟样点数，如果未检测到显著峰值则返回None
        """
        # 对于实数相关，直接使用绝对值
        correlation_abs = np.abs(correlation)
        max_idx = np.argmax(correlation_abs)
        
        # 索引转换：full模式下索引0对应延迟-(len(tx_signal)-1)
        delay_samples = max_idx - (len(tx_signal) - 1)
        
        # 检查峰值显著性
        max_correlation = correlation_abs[max_idx]
        mean_corr = np.mean(correlation_abs)
        std_corr = np.std(correlation_abs)
        noise_level = mean_corr + noise_threshold * std_corr
        
        if max_correlation < noise_level:
            return None
            
        # 确保延迟为正值
        return int(max(0, delay_samples))
    
    def load_tx_signal_from_data(self, tx_signal_data: str) -> np.ndarray:
        """
        从保存的信号数据字符串中解析发射信号（实数）
        
        Args:
            tx_signal_data: 发射信号数据字符串（JSON或列表格式）
            
        Returns:
            实数发射信号数组，解析失败返回空数组
        """
        if not tx_signal_data or tx_signal_data.isspace():
            return np.array([])
            
        try:
            # 尝试解析JSON格式或Python列表格式
            if tx_signal_data.startswith('[') and tx_signal_data.endswith(']'):
                # Python列表格式
                signal_list = ast.literal_eval(tx_signal_data)
                # 转换为实数信号
                if isinstance(signal_list[0], (list, tuple)):
                    # 复数格式 [real, imag] 或 (real, imag)
                    # 提取实部
                    real_parts = [item[0] for item in signal_list]
                    return np.array(real_parts, dtype=np.float64)
                else:
                    # 已经是实数格式
                    return np.array(signal_list, dtype=np.float64)
            else:
                # 尝试其他格式解析
                import json
                signal_data = json.loads(tx_signal_data)
                if isinstance(signal_data, list):
                    # 同样处理复数和实数格式
                    if isinstance(signal_data[0], (list, tuple)):
                        # 复数格式，提取实部
                        real_parts = [item[0] for item in signal_data]
                        return np.array(real_parts, dtype=np.float64)
                    else:
                        return np.array(signal_data, dtype=np.float64)
                else:
                    print("警告: 无法识别的信号数据格式")
                    return np.array([])
                    
        except Exception as e:
            print(f"警告: 解析发射信号数据失败: {e}")
            return np.array([])
    



class MethodEvaluator:
    """方法评估器"""
    
    def __init__(self):
        self.estimator = TraditionalToAEstimators()
        self.method_configs = {
            'Peak': {
                'noise_factor': Config.PEAK_NOISE_FACTOR,
                'description': '寻找超过噪声阈值的第一个峰值'
            },
            'IFP': {
                'noise_factor': Config.IFP_NOISE_FACTOR,
                'description': '寻找CIR凹凸性改变且超过阈值的第一个点'
            },
            'LDE': {
                'description': '前沿检测，连续样点阈值检测'
            },
            'IPS': {
                'noise_factor': Config.IFP_NOISE_FACTOR,
                'description': '迭代峰值减除，消除多径找到最早LOS路径'
            },
            'MatchedFilter': {
                'description': '匹配滤波+峰值检测，使用原始发射信号'
            }
        }
    
    def evaluate_on_data(self, test_data_path: str) -> Optional[dict]:
        """
        在测试数据上评估传统方法
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            评估结果字典，失败时返回None
        """
        try:
            # 加载数据
            data = self._load_test_data(test_data_path)
            
            # 验证数据格式
            self._validate_data(data)
            
            # 检查是否支持匹配滤波
            has_tx_signal, signal_base_path = self._check_tx_signal_support(
                data, test_data_path
            )
            
            # 评估各种方法
            results = self._evaluate_all_methods(data, has_tx_signal, signal_base_path)
            
            # 计算性能指标
            return self._compute_performance_metrics(results)
            
        except Exception as e:
            print(f"传统方法评估失败: {e}")
            return None
    
    def _load_test_data(self, test_data_path: str) -> pd.DataFrame:
        """加载测试数据"""
        if os.path.isfile(test_data_path):
            return pd.read_csv(test_data_path)
        elif os.path.isdir(test_data_path):
            return load_data_from_path(
                test_data_path, recursive=True, add_source_column=False
            )
        else:
            raise FileNotFoundError(f"数据路径不存在: {test_data_path}")
    
    def _validate_data(self, data: pd.DataFrame):
        """验证数据格式"""
        required_cols = ['CIR']
        target_cols = ['TOA', 'target']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需列: {missing_cols}")
            
        has_target = any(col in data.columns for col in target_cols)
        if not has_target:
            raise ValueError(f"数据必须包含目标列: {target_cols}")
    
    def _check_tx_signal_support(self, data: pd.DataFrame, test_data_path: str) -> tuple:
        """检查是否支持发射信号数据（匹配滤波）"""
        # 检查是否有 tx_signal_data 列
        has_tx_signal = 'tx_signal_data' in data.columns
        signal_base_path = None
        
        if has_tx_signal:
            # 检查是否有有效的信号数据
            valid_signals = data['tx_signal_data'].notna() & (data['tx_signal_data'] != '')
            has_tx_signal = valid_signals.any()
            
        return has_tx_signal, signal_base_path
    
    def _evaluate_all_methods(self, data: pd.DataFrame, has_tx_signal: bool, signal_base_path: str) -> dict:
        """评估所有方法"""
        results = {}
        
        # 获取目标值
        if 'TOA' in data.columns:
            targets = data['TOA'].values
        elif 'target' in data.columns:
            targets = data['target'].values
        else:
            raise ValueError("找不到目标列")
        
        # 评估各种方法
        methods = {
            'Peak': self.estimator.peak_method,
            'LDE': self.estimator.lde_method
        }
        
        # IFP和IPS方法需要发射信号数据
        if has_tx_signal:
            methods['IFP'] = lambda cir, idx=None: (
                self.estimator.ifp_method(
                    cir, data.iloc[idx]['tx_signal_data']
                    if idx is not None else ""
                )
            )
            methods['IPS'] = lambda cir, idx=None: (
                self.estimator.ips_method(
                    cir, data.iloc[idx]['tx_signal_data']
                    if idx is not None else ""
                )
            )
            methods['MatchedFilter'] = lambda cir, idx=None: (
                self.estimator.matched_filter_method(
                    cir, data.iloc[idx]['tx_signal_data'] 
                    if idx is not None else ""
                )
            )
        else:
            # 没有发射信号数据时，IFP和IPS退化为Peak方法
            methods['IFP'] = self.estimator.peak_method
            methods['IPS'] = self.estimator.peak_method
        
        # 执行评估
        for method_name, method_func in methods.items():
            predictions = []
            
            for idx, cir_str in enumerate(data['CIR']):
                try:
                    # 解析CIR数据
                    if isinstance(cir_str, str):
                        cir = np.array(ast.literal_eval(cir_str))
                    else:
                        cir = np.array(cir_str)
                    
                    # 执行方法
                    if method_name == 'MatchedFilter':
                        prediction = method_func(cir, idx)
                    else:
                        prediction = method_func(cir)
                    
                    predictions.append(prediction)
                    
                except Exception as e:
                    print(f"方法 {method_name} 在样本 {idx} 上失败: {e}")
                    predictions.append(0)  # 使用默认值
            
            results[method_name] = {
                'predictions': np.array(predictions),
                'targets': targets
            }
        
        return results
    
    def _compute_performance_metrics(self, results: dict) -> dict:
        """计算性能指标"""
        performance = {}
        
        for method_name, result in results.items():
            predictions = result['predictions']
            targets = result['targets']
            
            # 计算误差
            errors = np.abs(predictions - targets)
            
            # 计算性能指标
            performance[method_name] = {
                'MAE': np.mean(errors),
                'RMSE': np.sqrt(np.mean(errors ** 2)),
                'Max_Error': np.max(errors),
                'Std_Error': np.std(errors),
                'Median_Error': np.median(errors)
            }
        
        return performance


def evaluate_traditional_methods_on_data(test_data_path: str) -> Optional[dict]:
    """
    在真实数据上评估传统方法
    
    Args:
        test_data_path: 测试数据路径（CSV文件或目录）
        
    Returns:
        评估结果字典，包含各方法的性能指标
    """
    try:
        # 加载和验证数据
        data = _load_test_data(test_data_path)
        _validate_test_data(data)
        
        # 初始化估计器和结果存储
        estimator = TraditionalToAEstimators()
        results = _initialize_results_storage()
        
        # 检查匹配滤波支持
        has_tx_signal, signal_base_path = _check_matched_filter_support(
            data, test_data_path
        )
        
        # 评估所有样本
        _evaluate_samples(data, estimator, results, has_tx_signal, signal_base_path)
        
        # 计算性能指标
        return _compute_method_metrics(results)
        
    except Exception as e:
        print(f"传统方法评估失败: {e}")
        return None


def _load_test_data(test_data_path: str) -> pd.DataFrame:
    """加载测试数据"""
    if os.path.isfile(test_data_path):
        return pd.read_csv(test_data_path)
    elif os.path.isdir(test_data_path):
        return load_data_from_path(
            test_data_path, recursive=True, add_source_column=False
        )
    else:
        raise FileNotFoundError(f"数据路径不存在: {test_data_path}")


def _validate_test_data(data: pd.DataFrame):
    """验证测试数据格式"""
    if 'CIR' not in data.columns:
        raise ValueError("数据必须包含CIR列")
    
    if not ('TOA' in data.columns or 'target' in data.columns):
        raise ValueError("数据必须包含TOA或target列")


def _initialize_results_storage() -> dict:
    """初始化结果存储结构"""
    return {
        'Peak': {'predictions': [], 'targets': []},
        'IFP': {'predictions': [], 'targets': []},
        'LDE': {'predictions': [], 'targets': []},
        'IPS': {'predictions': [], 'targets': []},
        'MatchedFilter': {'predictions': [], 'targets': []}
    }


def _check_matched_filter_support(
    data: pd.DataFrame, test_data_path: str
) -> Tuple[bool, str]:
    """检查匹配滤波方法支持"""
    has_tx_signal = 'tx_signal_data' in data.columns
    
    if has_tx_signal:
        print("检测到tx_signal_data列，将评估匹配滤波方法")
        print("使用预测结果中保存的原始信号数据")
        signal_base_path = ''  # 不需要文件路径了
    else:
        print("未检测到tx_signal_data列，跳过匹配滤波方法")
        signal_base_path = ''
    
    return has_tx_signal, signal_base_path





def _evaluate_samples(data: pd.DataFrame, estimator: TraditionalToAEstimators,
                      results: dict, has_tx_signal: bool,
                      signal_base_path: str):
    """评估所有样本"""
    successful_samples = 0
    
    for idx, row in data.iterrows():
        try:
            # 解析CIR和目标值
            cir = estimator.parse_cir_string(row['CIR'])
            if len(cir) == 0:
                continue
                
            true_toa = row['TOA'] if 'TOA' in row else row['target']
            
            # 计算各方法的估计结果
            toa_estimates = _compute_toa_estimates(
                cir, estimator, row, has_tx_signal, signal_base_path
            )
            
            # 存储有效结果
            _store_valid_results(results, toa_estimates, true_toa)
            successful_samples += 1
            
        except Exception:
            continue
    
    print(f"成功处理 {successful_samples} 个样本")


def _compute_toa_estimates(cir: np.ndarray,
                           estimator: TraditionalToAEstimators,
                           row: pd.Series, has_tx_signal: bool,
                           signal_base_path: str) -> dict:
    """计算各种方法的TOA估计"""
    max_cir = max(np.abs(cir))
    
    estimates = {
        'Peak': estimator.peak_method(
            cir, noise_threshold=Config.PEAK_NOISE_FACTOR * max_cir
        ),
        'IFP': estimator.ifp_method(
            cir, tx_signal_data=row.get('tx_signal_data', ''),
            noise_threshold=Config.IFP_NOISE_FACTOR * max_cir
        ),
        'LDE': estimator.lde_method(cir),
        'IPS': estimator.ips_method(
            cir, tx_signal_data=row.get('tx_signal_data', ''),
            noise_threshold=Config.IPS_NOISE_FACTOR * max_cir
        ),
        'MatchedFilter': None
    }
    
    # 匹配滤波方法（使用保存的信号数据）
    if has_tx_signal and 'tx_signal_data' in row:
        if row['tx_signal_data'] and not pd.isna(row['tx_signal_data']):
            estimates['MatchedFilter'] = estimator.matched_filter_method(
                cir, row['tx_signal_data'],
                noise_threshold=Config.MATCHED_FILTER_NOISE_FACTOR
            )
    
    return estimates


def _store_valid_results(results: dict, estimates: dict, true_toa: float):
    """存储有效的估计结果"""
    for method_name, estimate in estimates.items():
        if estimate is not None:
            results[method_name]['predictions'].append(estimate)
            results[method_name]['targets'].append(true_toa)


def _compute_method_metrics(results: dict) -> dict:
    """计算各方法的性能指标"""
    method_metrics = {}
    
    for method_name, method_data in results.items():
        if len(method_data['predictions']) > 0:
            predictions = np.array(method_data['predictions'])
            targets = np.array(method_data['targets'])
            
            metrics = calculate_metrics(
                targets, predictions, task_type='regression'
            )
            
            errors = np.abs(predictions - targets)
            
            method_metrics[method_name] = {
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'p90_error': np.percentile(errors, 90),
                'description': get_method_description(method_name),
                'samples': len(predictions),
                'errors': errors
            }
    
    return method_metrics


def _estimate_noise(cir, noise_window=None, method='std'):
    """
    估计噪声基线与噪声标准差
    cir: 1D numpy array
    noise_window: tuple (start_idx, end_idx) for noise-only region (optional)
                  if None, use first 10% samples as noise region
    method: 'std' or 'mad' (median absolute deviation)
    返回: (mean_noise, sigma_noise)
    """
    n = len(cir)
    if noise_window is None:
        start = int(n * 1 / 2)
        # end = max(1, int(0.1 * n))
        noise_seg = cir[start:]
    else:
        s, e = noise_window
        noise_seg = cir[s:e]
    mean_noise = float(np.mean(noise_seg))
    if method == 'mad':
        mad = np.median(np.abs(noise_seg - np.median(noise_seg)))
        sigma = 1.4826 * mad
    else:
        sigma = float(np.std(noise_seg, ddof=1))
    return mean_noise, sigma


def calculate_cdf_analysis(results_dict: dict) -> dict:
    """
    计算累积分布函数（CDF）分析
    
    Args:
        results_dict: 包含各方法结果的字典
        
    Returns:
        CDF分析结果字典
    """
    cdf_analysis = {}
    
    for method_name, method_data in results_dict.items():
        if 'errors' not in method_data:
            continue
            
        errors = method_data['errors']
        if len(errors) == 0:
            continue
        
        # 计算CDF
        sorted_errors = np.sort(errors)
        n = len(sorted_errors)
        cdf_values = np.arange(1, n + 1) / n
        
        # 计算关键百分位数
        percentile_values = {}
        for p in Config.CDF_PERCENTILES:
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
    """
    打印CDF分析摘要，包含完整性能指标
    
    Args:
        cdf_analysis: CDF分析结果字典
    """
    print("\nToA估计方法性能对比 - 详细分析:")
    print("=" * 80)
    
    for method_name, analysis in cdf_analysis.items():
        _print_method_analysis(method_name, analysis)


def _print_method_analysis(method_name: str, analysis: dict):
    """打印单个方法的分析结果"""
    print(f"\n【{method_name}方法】")
    print("-" * 40)
    
    # 基础统计指标
    _print_basic_metrics(analysis)
    
    # CDF百分位数
    _print_percentiles(analysis['percentiles'])
    
    # 误差分布概览
    _print_error_distribution(analysis['sorted_errors'])
    print()


def _print_basic_metrics(analysis: dict):
    """打印基础指标"""
    print("基础指标:")
    print(f"  MAE (平均绝对误差): {analysis['mean_error']:.4f}")
    
    if 'rmse' in analysis:
        print(f"  RMSE (均方根误差): {analysis['rmse']:.4f}")
    
    print(f"  标准差: {analysis['std_error']:.4f}")
    print(f"  样本数量: {len(analysis['sorted_errors'])}")


def _print_percentiles(percentiles: dict):
    """打印百分位数"""
    print("\n累积分布函数 (CDF) - 关键百分位数:")
    for p_name, p_value in percentiles.items():
        percentage = p_name[1:]  # 去掉'p'前缀
        print(f"  P{percentage}: {p_value:.4f} "
              f"(即{percentage}%的误差≤{p_value:.4f})")


def _print_error_distribution(sorted_errors: np.ndarray):
    """打印误差分布概览"""
    print("\n误差分布概览:")
    print(f"  最小误差: {sorted_errors[0]:.4f}")
    print(f"  最大误差: {sorted_errors[-1]:.4f}")
    print(f"  误差范围: {sorted_errors[-1] - sorted_errors[0]:.4f}")


def get_method_description(method_name: str) -> str:
    """
    获取方法描述
    
    Args:
        method_name: 方法名称
        
    Returns:
        方法描述字符串
    """
    descriptions = {
        'Peak': '寻找超过噪声阈值的第一个峰值',
        'IFP': '寻找CIR凹凸性改变且超过阈值的第一个点',
        'LDE': '前沿检测，连续样点阈值检测',
        'IPS': '迭代峰值减除，消除多径找到最早LOS路径',
        'MatchedFilter': '匹配滤波+峰值检测，使用原始发射信号'
    }
    return descriptions.get(method_name, '未知方法')


def load_and_evaluate_model(
    experiment_dir: str, test_data_path: Optional[str] = None
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
                targets = np.array(pred_data['target'].values)
                predictions = np.array(pred_data['prediction'].values)
                
                metrics = calculate_metrics(
                    targets, predictions, task_type='regression'
                )
                
                errors = np.abs(predictions - targets)
                
                return {
                    'metrics': {
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'p90_error': np.percentile(errors, 90)
                    },
                    'test_samples': len(targets),
                    'evaluation_method': 'from_predictions_file',
                    'errors': errors
                }
        
        # 3. 如果提供了测试数据，进行实时模型加载和评估
        if test_data_path and os.path.exists(test_data_path):
            return _load_model_and_predict(experiment_dir, test_data_path)
        
        print("警告: 未找到评估结果文件或测试数据")
        return {}
        
    except Exception as e:
        print(f"模型结果加载失败: {e}")
        return {}


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
        
        errors = np.abs(predictions_orig - targets_orig)
        
        return {
            'metrics': {
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'p90_error': np.percentile(errors, 90)
            },
            'test_samples': len(targets_orig),
            'evaluation_method': 'live_model_prediction',
            'errors': errors
        }
        
    except Exception as e:
        print(f"模型加载和预测失败: {e}")
        return {}


def main():
    """主函数：执行ToA估计方法性能对比"""
    args = _parse_arguments()
    
    print("开始ToA估计方法性能对比分析...")
    
    # 评估传统方法
    print("\n1. 评估传统方法...")
    traditional_results = evaluate_traditional_methods_on_data(args.test_data)
    
    # 评估深度学习模型
    print("\n2. 评估深度学习模型...")
    model_results = None
    if args.experiment_dir:
        model_results = load_and_evaluate_model(
            args.experiment_dir, args.test_data
        )
    
    # 生成综合分析报告
    print("\n3. 生成分析报告...")
    if traditional_results:
        _generate_comprehensive_report(traditional_results, model_results)
    else:
        print("传统方法评估失败，无法生成报告")


def _parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ToA估计方法性能对比分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的传统方法：
  - Peak: 峰值检测方法
  - IFP: 拐点检测方法  
  - LDE: 前沿检测方法
  - MatchedFilter: 匹配滤波方法（需要发射信号文件）

示例用法：
  python method_comparison.py --test_data data.csv --experiment_dir results/model
        """
    )
    
    parser.add_argument(
        '--test_data',
        type=str,
        default="results/toa_Synthesis_with_noisy_transformer_2/test_predictions.csv",
        help='测试数据路径（CSV文件或目录）'
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default="results/toa_Synthesis_with_noisy_transformer_2",
        help='深度学习模型实验目录路径'
    )
    
    return parser.parse_args()


def _generate_comprehensive_report(traditional_results: dict, 
                                 model_results: Optional[dict]):
    """生成综合分析报告"""
    # 计算CDF分析
    cdf_analysis = calculate_cdf_analysis(traditional_results)
    
    # 添加传统方法的RMSE信息
    _add_rmse_to_cdf(cdf_analysis, traditional_results)
    
    # 添加模型结果（如果有）
    if model_results and 'errors' in model_results:
        _add_model_to_cdf(cdf_analysis, model_results)
    
    # 打印详细报告
    print_cdf_summary(cdf_analysis)


def _add_rmse_to_cdf(cdf_analysis: dict, traditional_results: dict):
    """将RMSE信息添加到CDF分析中"""
    for method_name in cdf_analysis:
        if method_name in traditional_results:
            cdf_analysis[method_name]['rmse'] = (
                traditional_results[method_name]['rmse']
            )


def _add_model_to_cdf(cdf_analysis: dict, model_results: dict):
    """将模型结果添加到CDF分析中"""
    errors = model_results['errors']
    cdf_analysis['Model'] = {
        'sorted_errors': np.sort(errors),
        'cdf_values': np.arange(1, len(errors) + 1) / len(errors),
        'percentiles': {
            f'p{p}': np.percentile(errors, p)
            for p in Config.CDF_PERCENTILES
        },
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'rmse': model_results['metrics']['rmse']
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行失败: {e}")
        import traceback
        traceback.print_exc()
