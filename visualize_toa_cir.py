#!/usr/bin/env python3
"""
TOA在CIR中的可视化脚本

该脚本用于可视化和分析TOA数据集中CIR信号和TOA位置的关系
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(dataset_dir):
    """加载数据集"""
    dataset_dir = Path(dataset_dir)
    dataset_file = dataset_dir / "complete_dataset.json"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    return dataset, dataset_dir


def load_signal(dataset_dir, sample):
    """加载CIR信号"""
    signals_dir = dataset_dir / "signals"
    signal_file = signals_dir / sample['signal_file']
    
    if not signal_file.exists():
        raise FileNotFoundError(f"信号文件不存在: {signal_file}")
    
    return np.load(signal_file)


def plot_single_sample(dataset, dataset_dir, sample_idx, time_window=None):
    """绘制单个样本的CIR和TOA"""
    if sample_idx >= len(dataset):
        raise IndexError(f"样本索引 {sample_idx} 超出范围")

    sample = dataset[sample_idx]
    cir = load_signal(dataset_dir, sample)

    # 创建时间轴
    fs = sample['sampling_frequency']
    time_axis = np.arange(len(cir)) / fs

    # 应用时间窗口
    if time_window:
        window_samples = int(time_window * fs)
        if window_samples < len(cir):
            cir = cir[:window_samples]
            time_axis = time_axis[:window_samples]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制CIR信号
    ax1.plot(time_axis * 1000, np.real(cir), 'b-', 
             linewidth=0.8, label='CIR Real')
    if np.iscomplexobj(cir):
        ax1.plot(time_axis * 1000, np.imag(cir), 'r-', 
                 linewidth=0.8, label='CIR Imag')

    # 标记TOA位置
    toa_values = np.array(sample['toa_values'])
    primary_toa = sample['primary_toa']

    # 显示所有TOA
    for i, toa in enumerate(toa_values):
        if time_window is None or toa <= time_window:
            if i == 0:  # 第一个是主要TOA
                ax1.axvline(toa * 1000, color='red', linestyle='--', 
                           linewidth=2, 
                           label=f'Primary TOA: {toa*1000:.2f}ms')
            else:
                ax1.axvline(toa * 1000, color='orange', linestyle=':', 
                           linewidth=1.5, alpha=0.7, 
                           label='Multipath TOA' if i == 1 else '')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Sample {sample_idx}: CIR Signal and TOA Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制CIR包络
    cir_envelope = np.abs(cir)
    ax2.plot(time_axis * 1000, cir_envelope, 'g-', 
             linewidth=1.2, label='CIR Envelope')

    # 在包络图上也标记TOA
    for i, toa in enumerate(toa_values):
        if time_window is None or toa <= time_window:
            if i == 0:
                ax2.axvline(toa * 1000, color='red', linestyle='--', 
                           linewidth=2, 
                           label=f'Primary TOA: {toa*1000:.2f}ms')
            else:
                ax2.axvline(toa * 1000, color='orange', linestyle=':', 
                           linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Envelope Amplitude')
    ax2.set_title('CIR Envelope and TOA Positions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 添加样本信息
    info_text = (
        f"Environment: {sample['environment_type']}\n"
        f"Movement: {sample['movement_scenario']}\n"
        f"Model: {sample['propagation_model']}\n"
        f"Distance: {sample['distance']:.1f}m\n"
        f"Multipaths: {sample['num_multipath']}\n"
        f"Delay spread: {sample['delay_spread']*1000:.2f}ms"
    )

    fig.text(0.02, 0.02, info_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", 
                      facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_multiple_samples(dataset, dataset_dir, sample_indices, 
                         time_window=None):
    """绘制多个样本的对比"""
    n_samples = len(sample_indices)
    
    if n_samples > 6:
        logger.warning("样本数量过多，只显示前6个")
        sample_indices = sample_indices[:6]
        n_samples = 6

    # 计算子图布局
    cols = min(3, n_samples)
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i, sample_idx in enumerate(sample_indices):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        sample = dataset[sample_idx]
        cir = load_signal(dataset_dir, sample)

        # 创建时间轴
        fs = sample['sampling_frequency']
        time_axis = np.arange(len(cir)) / fs

        # 应用时间窗口
        if time_window:
            window_samples = int(time_window * fs)
            if window_samples < len(cir):
                cir = cir[:window_samples]
                time_axis = time_axis[:window_samples]

        # 绘制CIR包络
        cir_envelope = np.abs(cir)
        ax.plot(time_axis * 1000, cir_envelope, 'b-', linewidth=1.0)

        # 标记主要TOA
        primary_toa = sample['primary_toa']
        if time_window is None or primary_toa <= time_window:
            ax.axvline(primary_toa * 1000, color='red', linestyle='--', 
                      linewidth=2, alpha=0.8)

        # 标记其他TOA
        toa_values = np.array(sample['toa_values'])
        for toa in toa_values[1:]:  # 跳过主要TOA
            if time_window is None or toa <= time_window:
                ax.axvline(toa * 1000, color='orange', linestyle=':', 
                          linewidth=1, alpha=0.6)

        ax.set_title(f'Sample {sample_idx}\nDist: {sample["distance"]:.1f}m')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Envelope')
        ax.grid(True, alpha=0.3)

    # 隐藏空的子图
    for i in range(n_samples, rows * cols):
        if rows > 1:
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        elif cols > 1:
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def analyze_toa_statistics(dataset):
    """分析TOA统计特性"""
    primary_toas = []
    delay_spreads = []
    num_multipaths = []
    distances = []

    for sample in dataset:
        primary_toas.append(sample['primary_toa'] * 1000)  # 转换为ms
        delay_spreads.append(sample['delay_spread'] * 1000)  # 转换为ms
        num_multipaths.append(sample['num_multipath'])
        distances.append(sample['distance'])

    # 创建统计图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 主要TOA分布
    ax1.hist(primary_toas, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Primary TOA (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Primary TOA Distribution')
    ax1.grid(True, alpha=0.3)

    # 延迟扩展分布
    ax2.hist(delay_spreads, bins=30, alpha=0.7, 
             color='orange', edgecolor='black')
    ax2.set_xlabel('Delay Spread (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Delay Spread Distribution')
    ax2.grid(True, alpha=0.3)

    # 多径数量分布
    unique_counts, count_freq = np.unique(num_multipaths, return_counts=True)
    ax3.bar(unique_counts, count_freq, alpha=0.7, 
            color='green', edgecolor='black')
    ax3.set_xlabel('Number of Multipaths')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Multipath Count Distribution')
    ax3.grid(True, alpha=0.3)

    # TOA vs 距离
    ax4.scatter(distances, primary_toas, alpha=0.6, s=20)
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Primary TOA (ms)')
    ax4.set_title('TOA vs Distance')
    ax4.grid(True, alpha=0.3)

    # 添加理论线（假设声速1500m/s）
    max_dist = max(distances)
    theoretical_x = np.linspace(0, max_dist, 100)
    theoretical_y = theoretical_x / 1500 * 1000  # 转换为ms
    ax4.plot(theoretical_x, theoretical_y, 'r--', 
            label='Theoretical TOA (c=1500m/s)', alpha=0.8)
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print(f"\n=== TOA Statistics Analysis ===")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Primary TOA: mean={np.mean(primary_toas):.2f}ms, "
          f"std={np.std(primary_toas):.2f}ms")
    print(f"Delay spread: mean={np.mean(delay_spreads):.2f}ms, "
          f"std={np.std(delay_spreads):.2f}ms")
    print(f"Multipath count: mean={np.mean(num_multipaths):.1f}, "
          f"range=[{min(num_multipaths)}, {max(num_multipaths)}]")
    print(f"Distance: mean={np.mean(distances):.1f}m, "
          f"range=[{min(distances):.1f}, {max(distances):.1f}]m")


def find_interesting_samples(dataset, criteria="high_multipath"):
    """找到有趣的样本用于展示"""
    if criteria == "high_multipath":
        # 找到多径数量最多的样本
        multipath_counts = [sample['num_multipath'] for sample in dataset]
        max_multipath = max(multipath_counts)
        indices = [i for i, count in enumerate(multipath_counts) 
                  if count >= max_multipath - 1]
        return indices[:5]  # 返回前5个
    
    elif criteria == "large_delay_spread":
        # 找到延迟扩展最大的样本
        delay_spreads = [(i, sample['delay_spread']) 
                       for i, sample in enumerate(dataset)]
        delay_spreads.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in delay_spreads[:5]]
    
    elif criteria == "short_distance":
        # 找到短距离样本
        distances = [(i, sample['distance']) 
                    for i, sample in enumerate(dataset)]
        distances.sort(key=lambda x: x[1])
        return [i for i, _ in distances[:5]]
    
    elif criteria == "long_distance":
        # 找到长距离样本
        distances = [(i, sample['distance']) 
                    for i, sample in enumerate(dataset)]
        distances.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in distances[:5]]
    
    else:
        # 随机选择
        return np.random.choice(len(dataset), 
                              min(5, len(dataset)), 
                              replace=False).tolist()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="可视化TOA在CIR中的位置"
    )
    parser.add_argument("--dataset_dir", type=str, 
                       default="toa_dataset_test_fixed",
                       help="数据集目录路径")
    parser.add_argument("--sample_idx", type=int, default=None,
                       help="要显示的样本索引")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="要显示的样本数量（用于多样本对比）")
    parser.add_argument("--time_window", type=float, default=None,
                       help="时间窗口长度（秒）")
    parser.add_argument("--show_stats", action="store_true",
                       help="显示TOA统计分析")
    parser.add_argument("--criteria", type=str, default="high_multipath",
                       choices=["high_multipath", "large_delay_spread", 
                               "short_distance", "long_distance", "random"],
                       help="选择有趣样本的标准")

    args = parser.parse_args()

    try:
        # 加载数据集
        dataset, dataset_dir = load_dataset(args.dataset_dir)
        logger.info(f"加载了包含 {len(dataset)} 个样本的数据集")

        if args.show_stats:
            # 显示统计分析
            analyze_toa_statistics(dataset)

        if args.sample_idx is not None:
            # 显示指定样本
            plot_single_sample(dataset, dataset_dir, args.sample_idx, 
                             args.time_window)
        else:
            # 找到有趣的样本并显示
            interesting_samples = find_interesting_samples(dataset, 
                                                          args.criteria)
            selected_samples = interesting_samples[:args.num_samples]

            print(f"根据标准 '{args.criteria}' 选择的样本: {selected_samples}")

            if len(selected_samples) == 1:
                plot_single_sample(dataset, dataset_dir, selected_samples[0], 
                                 args.time_window)
            else:
                plot_multiple_samples(dataset, dataset_dir, selected_samples, 
                                     args.time_window)

    except Exception as e:
        logger.error(f"运行出错: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
