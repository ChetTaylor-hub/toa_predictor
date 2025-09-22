import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path


def load_metadata(output_dir, sample_id):
    """优先读取CSV，否则读取JSON Lines"""
    csv_path = Path(output_dir) / "metadata.csv"
    json_path = Path(output_dir) / "complete_dataset.json"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        row = df[df['sample_id'] == sample_id]
        if row.empty:
            raise ValueError(f"Sample {sample_id} not found in CSV")
        meta = row.iloc[0].to_dict()
        return meta
    elif json_path.exists():
        with open(json_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if obj['sample_id'] == sample_id:
                    return obj
        raise ValueError(f"Sample {sample_id} not found in JSON Lines")
    else:
        raise FileNotFoundError("No metadata file found")


def main():
    parser = argparse.ArgumentParser(description="可视化TOA在CIR中的位置")
    parser.add_argument('--output_dir', type=str, default='toa_dataset_enhanced_refactored', help='数据集输出目录')
    parser.add_argument('--sample_id', type=int, default=0, help='样本编号')
    args = parser.parse_args()

    meta = load_metadata(args.output_dir, args.sample_id)
    cir_path = Path(args.output_dir) / "signals" / meta['rx_signal_file']
    cir = np.load(cir_path)
    fs = int(meta['sampling_frequency'])
    primary_toa = float(meta['primary_toa'])

    # 计算TOA对应的采样点
    toa_idx = int(primary_toa * fs)

    # 读取tx_signal
    tx_signal = None
    if 'tx_signal_file' in meta:
        tx_signal_path = Path(args.output_dir) / "signals" / meta['tx_signal_file']
        if tx_signal_path.exists():
            tx_signal = np.load(tx_signal_path)

    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    # CIR
    axes[0].plot(cir, label='CIR')
    axes[0].axvline(toa_idx, color='r', linestyle='--', label=f'TOA ({primary_toa:.6f}s)')
    axes[0].set_title(f'CIR with TOA (Sample {args.sample_id})')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    # tx_signal 时域
    if tx_signal is not None:
        axes[1].plot(tx_signal, label='TX Signal', color='g')
        axes[1].set_title('Transmitted Signal (TX) - Time Domain')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()
        # tx_signal 频域
        tx_freq = np.fft.rfft(tx_signal)
        freq_axis = np.fft.rfftfreq(len(tx_signal), d=1/fs)
        axes[2].plot(freq_axis, np.abs(tx_freq), label='TX Signal Spectrum', color='m')
        axes[2].set_title('Transmitted Signal (TX) - Frequency Domain')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Magnitude')
        axes[2].legend()
    else:
        axes[1].text(0.5, 0.5, 'TX Signal Not Found', ha='center', va='center', fontsize=14)
        axes[1].set_title('Transmitted Signal (TX) - Time Domain')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Amplitude')
        axes[2].text(0.5, 0.5, 'TX Signal Not Found', ha='center', va='center', fontsize=14)
        axes[2].set_title('Transmitted Signal (TX) - Frequency Domain')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
