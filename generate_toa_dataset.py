#!/usr/bin/env python3
"""
TOA Dataset Generator using UnderwaterAcoustics.jl Python API

This script generates a dataset containing Channel Impulse Response (CIR) and
Time of Arrival (TOA) data for underwater acoustic communication scenarios.
"""

import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Optional
import argparse

# Import Julia interface
try:
    from juliacall import Main as jl
    import juliapkg

    # Ensure required Julia packages are installed
    try:
        jl.seval("using UnderwaterAcoustics")
    except Exception:
        print("Installing UnderwaterAcoustics.jl...")
        juliapkg.add("UnderwaterAcoustics",
                     "0efb1f7a-1ce7-46d2-9f48-546a4c8fbb99")
        jl.seval("using UnderwaterAcoustics")

    try:
        jl.seval("using AcousticsToolbox")
    except Exception:
        print("Installing AcousticsToolbox.jl...")
        juliapkg.add("AcousticsToolbox",
                     "268a15bc-5756-47d6-9bea-fa5dc21c97f8")
        jl.seval("using AcousticsToolbox")

except ImportError:
    print("Error: juliacall not found. "
          "Please install with: pip install juliacall")
    exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_array(arr, title="array trend"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(arr, marker='o')
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.grid(True)
    plt.show()


class TOADatasetGenerator:
    """Generator for TOA dataset using UnderwaterAcoustics.jl"""

    def __init__(self, output_dir: str = "toa_dataset_python"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.signals_dir = self.output_dir / "signals"
        self.signals_dir.mkdir(exist_ok=True)

        # Dataset storage
        self.dataset = []

        # Simulation parameters
        self.fs = 192000  # Sampling frequency (Hz)
        self.signal_duration = 0.5  # Signal duration (s) - 增加到500ms以包含TOA
        self.frequency = 10000  # Signal frequency (Hz)

    def generate_environment(self, scenario_type: str = "shallow_water"):
        """Generate underwater environment - iso-velocity"""
        if scenario_type == "shallow_water":
            bathymetry = 20
            seabed = jl.SandyClay
        elif scenario_type == "deep_water":
            bathymetry = 200
            seabed = jl.SandyClay
        elif scenario_type == "variable_depth":
            # 使用固定深度和等速度声速剖面以支持PekerisRayTracer
            bathymetry = 100
            seabed = jl.SandyClay
        else:
            bathymetry = 20
            seabed = jl.SandyClay

        # 为PekerisRayTracer使用恒定声速
        soundspeed = 1500  # 恒定声速

        return jl.UnderwaterEnvironment(
            bathymetry=bathymetry,
            seabed=seabed,
            soundspeed=soundspeed
        )

    def generate_transceiver_positions(self, scenario: str):
        """Generate transmitter and receiver positions"""
        if scenario == "fixed_tx_moving_rx":
            tx_x, tx_z = 0.0, -5.0
            rx_x = np.random.uniform(50, 500)
            rx_z = np.random.uniform(-15, -3)
        elif scenario == "moving_tx_fixed_rx":
            # 发射器x坐标必须为0以支持Bellhop
            tx_x = 0.0
            tx_z = np.random.uniform(-15, -3)
            rx_x, rx_z = 200.0, -10.0
        elif scenario == "both_moving":
            # 发射器x坐标必须为0以支持Bellhop
            tx_x = 0.0
            tx_z = np.random.uniform(-18, -2)
            rx_x = np.random.uniform(100, 400)
            rx_z = np.random.uniform(-18, -2)
        else:
            tx_x, tx_z = 0.0, -5.0
            rx_x, rx_z = 100.0, -10.0

        frequency = np.random.uniform(8000, 12000)
        spl = np.random.uniform(160, 180)

        tx = jl.AcousticSource(tx_x, tx_z, frequency, spl=spl)
        rx = jl.AcousticReceiver(rx_x, rx_z)

        return tx, rx

    def compute_rays_and_arrivals(self, env, tx, rx, propagation_model):
        """Compute ray arrivals and channel response"""
        if propagation_model == "Bellhop":
            try:
                pm = jl.Bellhop(env)
            except Exception:
                logger.warning("Bellhop not available, using PekerisRayTracer")
                pm = jl.PekerisRayTracer(env)
        else:
            pm = jl.PekerisRayTracer(env)

        rays = jl.arrivals(pm, tx, rx)

        # noise_level = np.random.uniform(1e-6, 1e6)
        noise_level = 1e-3  # 降低噪声水平
        ch = jl.channel(pm, tx, rx, self.fs,
                        noise=jl.RedGaussianNoise(noise_level))

        return rays, ch

    def extract_toa_and_cir(self, rays, ch, distance):
        """Extract TOA and CIR data - 模拟真实的接收过程"""
        ray_times = []
        ray_amplitudes = []
        ray_phases = []

        rays_array = np.array(rays)

        for i in range(len(rays_array)):
            ray = rays_array[i]
            arrival_time = float(ray.time)
            amplitude_db = float(abs(ray.phasor))
            amplitude_linear = 10**(amplitude_db/20.0)
            # 从复数相位器中提取相位
            phase = float(np.angle(ray.phasor))

            ray_times.append(arrival_time)
            ray_amplitudes.append(amplitude_linear)
            ray_phases.append(phase)

        # 根据距离和最大TOA动态调整信号长度
        max_toa = max(ray_times) if ray_times else distance/1500.0
        min_toa = min(ray_times) if ray_times else distance/1500.0
        # 确保信号长度至少包含所有TOA，再加上一些缓冲
        min_duration = max_toa - min_toa + 0.01  # 额外10ms缓冲
        actual_duration = min_duration
        
        signal_length = int(actual_duration * self.fs)
        
        # 创建发射信号 - 使用脉冲信号，更贴近实际TOA检测场景
        pulse_duration = 0.001  # 1ms脉冲
        pulse_samples = int(pulse_duration * self.fs)
        
        # 生成调制脉冲（高斯包络的正弦波）
        t_pulse = np.arange(pulse_samples) / self.fs
        pulse_center = pulse_duration/2
        pulse_width = pulse_duration/8
        envelope = np.exp(-((t_pulse - pulse_center) / pulse_width)**2)
        carrier = np.sin(2 * np.pi * self.frequency * t_pulse)
        pulse_signal = envelope * carrier

        # 在整个观测时间窗口中，脉冲从t=0开始发射（正确的物理过程）
        full_signal = np.zeros(signal_length)
        full_signal[:pulse_samples] = pulse_signal

        # 计算primary_toa（需要在使用前定义）
        primary_toa = min(ray_times) if ray_times else 0.0
        
        total_length = int(self.fs * self.signal_duration)
        # 使用声学传播模型计算接收信号
        try:
            # Julia transmit计算的是发射信号经过传播后的接收信号
            received_signal = jl.transmit(ch, full_signal, fs=self.fs)
            # np.array(received_signal) [n,1] --> [n,]
            received_array = np.array(received_signal).squeeze()

            # 创建完整的CIR信号，初始为全零
            cir = np.zeros(total_length)
            
            # 将接收信号插入到正确的TOA位置
            primary_toa_samples = int(primary_toa * self.fs)
            
            # 将接收信号放置在主要TOA位置
            end_sample = primary_toa_samples + len(received_array)
            if end_sample <= total_length:
                cir[primary_toa_samples:end_sample] = received_array
            else:
                # 如果接收信号太长，截断它
                max_samples = total_length - primary_toa_samples
                if max_samples > 0:
                    cir[primary_toa_samples:] = received_array[:max_samples]
                    
        except Exception:
            # 如果Julia传播失败，手动构建CIR
            cir = np.zeros(total_length, dtype=complex)
            
            # 为每个多径分量添加延时的脉冲副本
            for t, amp, phase in zip(ray_times, ray_amplitudes, ray_phases):
                if t < actual_duration:
                    delay_samples = int(t * self.fs)
                    # 确保延时后的信号不超出范围
                    end_idx = min(delay_samples + pulse_samples, total_length)
                    actual_pulse_samples = end_idx - delay_samples
                    
                    if actual_pulse_samples > 0:
                        # 添加带有幅度和相位的延时脉冲
                        pulse_part = pulse_signal[:actual_pulse_samples]
                        phase_factor = np.exp(1j * phase)
                        delayed_pulse = pulse_part * amp * phase_factor
                        cir[delay_samples:end_idx] += delayed_pulse
            
            # 添加噪声
            noise_level = 0.01 * np.max(np.abs(cir))
            real_noise = np.random.randn(signal_length)
            imag_noise = np.random.randn(signal_length)
            noise = noise_level * (real_noise + 1j * imag_noise)
            cir += noise
            
            # 转换为实数（取实部，模拟实数接收器）
            cir = np.real(cir)

        # primary_toa已经在上面计算过了
        toa_array = np.array(ray_times)

        features = {
            'num_rays': len(ray_times),
            'primary_toa': primary_toa,
            'max_amplitude': max(ray_amplitudes) if ray_amplitudes else 0.0,
            'rms_amplitude': (np.sqrt(np.mean(np.array(ray_amplitudes)**2))
                              if ray_amplitudes else 0.0),
            'delay_spread': (max(ray_times) - min(ray_times)
                             if len(ray_times) > 1 else 0.0),
            'actual_duration': actual_duration,
            'pulse_duration': pulse_duration
        }

        return toa_array, cir, features

    def generate_single_sample(self, sample_id: int, env_type: str,
                               movement_scenario: str,
                               propagation_model: str) -> Dict:
        """Generate a single dataset sample"""
        env = self.generate_environment(env_type)
        tx, rx = self.generate_transceiver_positions(movement_scenario)

        # 获取位置信息 - 修复索引问题
        tx_location = jl.location(tx)
        rx_location = jl.location(rx)
        
        # 正确访问位置坐标
        tx_pos = (float(tx_location.x), float(tx_location.z))
        rx_pos = (float(rx_location.x), float(rx_location.z))
        distance = np.sqrt((tx_pos[0] - rx_pos[0])**2 +
                           (tx_pos[1] - rx_pos[1])**2)

        rays, ch = self.compute_rays_and_arrivals(env, tx, rx,
                                                  propagation_model)
        toa_array, cir, features = self.extract_toa_and_cir(rays, ch, distance)

        # Save CIR signal
        signal_filename = f"signal_{sample_id:06d}.npy"
        signal_path = self.signals_dir / signal_filename
        np.save(signal_path, cir)

        return {
            'sample_id': sample_id,
            'signal_file': signal_filename,
            'tx_position': tx_pos,
            'rx_position': rx_pos,
            'distance': distance,
            'environment_type': env_type,
            'movement_scenario': movement_scenario,
            'propagation_model': propagation_model,
            'toa_values': toa_array.tolist(),
            'primary_toa': features['primary_toa'],
            'num_multipath': features['num_rays'],
            'delay_spread': features['delay_spread'],
            'max_amplitude': features['max_amplitude'],
            'rms_amplitude': features['rms_amplitude'],
            'sampling_frequency': self.fs,
            'signal_duration': features['actual_duration'],  # 使用实际信号持续时间
            'base_signal_duration': self.signal_duration,   # 保存基础配置
            'pulse_duration': features['pulse_duration'],   # 保存脉冲持续时间
            'carrier_frequency': self.frequency,
            'timestamp': time.time()
        }

    def generate_dataset(self, num_samples: int = 1000,
                         env_types: Optional[List[str]] = None,
                         movement_scenarios: Optional[List[str]] = None,
                         propagation_models: Optional[List[str]] = None
                         ) -> None:
        """Generate complete dataset"""
        if env_types is None:
            env_types = ["shallow_water", "deep_water", "variable_depth"]

        if movement_scenarios is None:
            movement_scenarios = ["fixed_tx_moving_rx", "moving_tx_fixed_rx",
                                  "both_moving"]

        if propagation_models is None:
            propagation_models = ["PekerisRayTracer", "Bellhop"]

        logger.info(f"Generating {num_samples} samples...")

        for i in tqdm(range(num_samples), desc="Generating TOA dataset"):
            try:
                env_type = np.random.choice(env_types)
                movement_scenario = np.random.choice(movement_scenarios)
                propagation_model = np.random.choice(propagation_models)

                sample_data = self.generate_single_sample(
                    i, env_type, movement_scenario, propagation_model
                )

                self.dataset.append(sample_data)

            except Exception as e:
                logger.error(f"Error generating sample {i}: {str(e)}")
                continue

        logger.info(f"Successfully generated {len(self.dataset)} samples")

    def save_dataset(self) -> None:
        """Save dataset to files"""
        # Save complete dataset as JSON
        dataset_file = self.output_dir / "complete_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(self.dataset, f, indent=2)

        # Create CSV metadata file
        try:
            import pandas as pd

            csv_data = []
            for sample in self.dataset:
                csv_row = {
                    'sample_id': sample['sample_id'],
                    'signal_file': sample['signal_file'],
                    'tx_x': sample['tx_position'][0],
                    'tx_z': sample['tx_position'][1],
                    'rx_x': sample['rx_position'][0],
                    'rx_z': sample['rx_position'][1],
                    'distance': sample['distance'],
                    'environment_type': sample['environment_type'],
                    'movement_scenario': sample['movement_scenario'],
                    'propagation_model': sample['propagation_model'],
                    'primary_toa': sample['primary_toa'],
                    'num_multipath': sample['num_multipath'],
                    'delay_spread': sample['delay_spread'],
                    'max_amplitude': sample['max_amplitude'],
                    'rms_amplitude': sample['rms_amplitude'],
                    'sampling_frequency': sample['sampling_frequency'],
                    'carrier_frequency': sample['carrier_frequency']
                }
                csv_data.append(csv_row)

            df = pd.DataFrame(csv_data)
            csv_file = self.output_dir / "metadata.csv"
            df.to_csv(csv_file, index=False)

        except ImportError:
            logger.warning("pandas not available, skipping CSV export")

        # Create README
        readme_content = f"""# TOA Dataset

## Description
This dataset contains Channel Impulse Response (CIR) and Time of Arrival (TOA)
data for underwater acoustic communication scenarios, generated using
UnderwaterAcoustics.jl.

## Dataset Structure
- `complete_dataset.json`: Complete dataset with all metadata and TOA arrays
- `metadata.csv`: Simplified metadata in CSV format
- `signals/`: Directory containing CIR signals as NumPy arrays (.npy files)

## Dataset Statistics
- Total samples: {len(self.dataset)}
- Sampling frequency: {self.fs} Hz
- Signal duration: {self.signal_duration} s
- Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Usage
```python
import json
import numpy as np

# Load dataset
with open('complete_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load a signal
sample = dataset[0]
signal = np.load(f"signals/{{sample['signal_file']}}")
toa = sample['primary_toa']
```
"""

        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)

        logger.info(f"Dataset saved to {self.output_dir}")
        logger.info(f"- {len(self.dataset)} samples")
        logger.info(f"- JSON: {dataset_file}")
        logger.info(f"- Signals: {self.signals_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate TOA dataset using UnderwaterAcoustics.jl"
    )
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="toa_dataset_python",
                        help="Output directory for dataset")

    args = parser.parse_args()

    # Create generator
    generator = TOADatasetGenerator(args.output_dir)

    # Generate dataset
    generator.generate_dataset(num_samples=args.num_samples)

    # Save dataset
    generator.save_dataset()

    print("Dataset generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {len(generator.dataset)} samples")


if __name__ == "__main__":
    main()
