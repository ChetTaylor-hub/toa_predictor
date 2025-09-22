#!/usr/bin/env python3
"""
Enhanced TOA Dataset Generator using UnderwaterAcoustics.jl Python API

This script generates a comprehensive and diverse dataset containing Channel Impulse 
Response (CIR) and Time of Arrival (TOA) data for underwater acoustic communication 
scenarios with maximum variability across all parameters.

Refactored for better readability, maintainability, and extensibility.
"""

import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Union, Tuple
import argparse
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import Julia interface
try:
    from juliacall import Main as jl
    import juliapkg
    
    # Julia package management
    JULIA_PACKAGES = {
        "UnderwaterAcoustics": "0efb1f7a-1ce7-46d2-9f48-546a4c8fbb99",
        "AcousticsToolbox": "268a15bc-5756-47d6-9bea-fa5dc21c97f8",
        "AcousticRayTracers": "f2c4e8d0-3d1a-11ee-0a9c-5fbcddc9b1b0"
    }
    
    for package, uuid in JULIA_PACKAGES.items():
        try:
            jl.seval(f"using {package}")
        except Exception:
            print(f"Installing {package}.jl...")
            juliapkg.add(package, uuid)
            jl.seval(f"using {package}")

except ImportError:
    print("Error: juliacall not found. Please install with: pip install juliacall")
    exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    fs: int = 192000  # Sampling frequency (Hz)
    signal_duration: float = 1.0  # Base signal duration (s)
    max_retries: int = 3
    timeout_seconds: int = 5
    

@dataclass
class EnvironmentParams:
    """Parameters for underwater environment"""
    bathymetry: float
    seabed_type: str
    surface_type: Optional[str]
    temperature: float
    salinity: float
    pH: float
    soundspeed_profile: str


@dataclass
class GeometryParams:
    """Parameters for transceiver geometry"""
    geometry_type: str
    tx_position: Tuple[float, float]
    rx_position: Tuple[float, float]
    frequency: float
    spl: float


@dataclass
class SignalParams:
    """Parameters for signal generation"""
    type: str
    duration: float
    frequency: float
    bandwidth: float
    samples: int = 0


class BoundaryConditions:
    """Constants for underwater boundary conditions"""
    
    SEABED_TYPES = [
        'Rock', 'Pebbles', 'SandyGravel', 'VeryCoarseSand',
        'MuddySandyGravel', 'CoarseSand', 'GravellyMuddySand', 
        'MediumSand', 'MuddyGravel', 'FineSand', 'MuddySand',
        'VeryFineSand', 'ClayeySand', 'CoarseSilt', 'SandySilt',
        'MediumSilt', 'SandyMud', 'FineSilt', 'SandyClay',
        'VeryFineSilt', 'SiltyClay', 'Clay'
    ]
    
    ELASTIC_SEABED_TYPES = [
        'ElasticRock', 'ElasticPebbles', 'ElasticSandyGravel',
        'ElasticVeryCoarseSand', 'ElasticMuddySandyGravel', 
        'ElasticCoarseSand', 'ElasticGravellyMuddySand',
        'ElasticMediumSand', 'ElasticMuddyGravel', 'ElasticFineSand',
        'ElasticMuddySand', 'ElasticVeryFineSand', 'ElasticClayeySand',
        'ElasticCoarseSilt', 'ElasticSandySilt'
    ]
    
    SEA_STATES = [
        'SeaState0', 'SeaState1', 'SeaState2', 'SeaState3', 
        'SeaState4', 'SeaState5', 'SeaState6', 'SeaState7',
        'SeaState8', 'SeaState9'
    ]


class PropagationModel(ABC):
    """Abstract base class for propagation models"""
    
    @abstractmethod
    def is_compatible(self, env_params: EnvironmentParams, 
                      geometry_params: Optional[GeometryParams] = None) -> bool:
        """Check if model is compatible with environment and geometry"""
        pass
    
    @abstractmethod
    def create_model(self, env, config: Dict):
        """Create the propagation model instance"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name"""
        pass


class PekerisRayTracerModel(PropagationModel):
    """Pekeris Ray Tracer implementation"""
    
    def is_compatible(self, env_params: EnvironmentParams, 
                      geometry_params: Optional[GeometryParams] = None) -> bool:
        # Limitations: Iso-velocity, range-independent
        # But serve as fallback for maximum compatibility
        if (env_params.soundspeed_profile != 'iso_velocity'):
            return False
        return True  # Always compatible as fallback model
    
    def create_model(self, env, config: Dict):
        max_bounces = config.get('max_bounces', np.random.randint(1, 10))
        return jl.PekerisRayTracer(env, max_bounces=max_bounces)
    
    def get_name(self) -> str:
        return 'PekerisRayTracer'


class PekerisModeSolverModel(PropagationModel):
    """Pekeris Mode Solver implementation"""
    
    def is_compatible(self, env_params: EnvironmentParams, 
                      geometry_params: Optional[GeometryParams] = None) -> bool:
        # Limitations: Iso-velocity, pressure-release surface,
        # fluid half-space seabed, no seabed absorption (no leaky modes)
        if (env_params.soundspeed_profile != 'iso_velocity' or
            env_params.surface_type is not None):
            return False
        return True
    
    def create_model(self, env, config: Dict):
        return jl.PekerisModeSolver(env)
    
    def get_name(self) -> str:
        return 'PekerisModeSolver'


class KrakenModel(PropagationModel):
    """Kraken normal mode model implementation"""
    
    def is_compatible(self, env_params: EnvironmentParams, 
                      geometry_params: Optional[GeometryParams] = None) -> bool:
        # Kraken要求surface_type为None（FluidBoundary），否则不兼容
        if env_params.surface_type is not None:
            return False
        # The Fortran Kraken propagation model requires transmitter at (x=0,y=0)
        # and all receivers in the right half-plane (x>0, y=0).
        # Constraints: No elastic seabed, no surface effects,
        # adequate depth for mode propagation
        
        # Check geometry constraints if provided
        if geometry_params is not None:
            tx_x, tx_z = geometry_params.tx_position
            rx_x, rx_z = geometry_params.rx_position
            # Kraken requires tx at origin and rx in right half-plane
            if tx_x != 0.0 or rx_x <= 0.0:
                return False
        
        return True
    
    def create_model(self, env, config: Dict):
        return jl.Kraken(env)

    def get_name(self) -> str:
        return 'Kraken'


class OrcaModel(PropagationModel):
    """Orca normal mode model implementation"""
    
    def is_compatible(self, env_params: EnvironmentParams, 
                      geometry_params: Optional[GeometryParams] = None) -> bool:
        # Orca要求surface_type为None（FluidBoundary），否则不兼容
        if env_params.surface_type is not None:
            return False
        # The Fortran Orca propagation model requires transmitter at (x=0,y=0)
        # and all receivers in the right half-plane (x>0, y=0).
        # Same constraints as Kraken: No elastic seabed, no surface effects
        
        # Check geometry constraints if provided
        if geometry_params is not None:
            tx_x, tx_z = geometry_params.tx_position
            rx_x, rx_z = geometry_params.rx_position
            # Orca requires tx at origin and rx in right half-plane
            if tx_x != 0.0 or rx_x <= 0.0:
                return False
        
        return True
    
    def create_model(self, env, config: Dict):
        return jl.Orca(env)

    def get_name(self) -> str:
        return 'Orca'


class BellhopModel(PropagationModel):
    """Bellhop ray/beam tracer implementation"""
    
    def is_compatible(self, env_params: EnvironmentParams, 
                      geometry_params: Optional[GeometryParams] = None) -> bool:
        # The Fortran Bellhop propagation model requires transmitter at (x=0,y=0)
        # and all receivers in the right half-plane (x>0, y=0).
        # Generally compatible but has some limitations:
        # Does NOT support: 3D environments, range-dependent sound speed,
        # N2-linear/analytic/quadratic interpolation for sound speed,
        # arbitrary reflection coefficients, directional/line sources

        # Check surface type
        if env_params.surface_type is not None:
            return False
        # Check geometry constraints if provided
        if geometry_params is not None:
            tx_x, tx_z = geometry_params.tx_position
            rx_x, rx_z = geometry_params.rx_position
            # Bellhop requires tx at origin and rx in right half-plane
            if tx_x != 0.0 or rx_x <= 0.0:
                return False
        
        return True  # Generally compatible with most scenarios
    
    def create_model(self, env, config: Dict):
        return jl.Bellhop(env)
    
    def get_name(self) -> str:
        return 'Bellhop'


class RaySolverModel(PropagationModel):
    """RaySolver 2½D acoustic Gaussian beam tracer implementation"""
    
    def is_compatible(self, env_params: EnvironmentParams, 
                      geometry_params: Optional[GeometryParams] = None) -> bool:
        # RaySolver requires transmitter at (x=0,y=0) and all receivers
        # in the right half-plane (x>0, y=0).
        # Julia implementation, differentiable, supports complex environments
        
        # Check geometry constraints if provided
        if geometry_params is not None:
            tx_x, tx_z = geometry_params.tx_position
            rx_x, rx_z = geometry_params.rx_position
            # RaySolver requires tx at origin and rx in right half-plane
            if tx_x != 0.0 or rx_x <= 0.0:
                return False
        
        return True  # Generally compatible, minimal limitations
    
    def create_model(self, env, config: Dict):
        return jl.RaySolver(env)
    
    def get_name(self) -> str:
        return 'RaySolver'


class PropagationModelFactory:
    """Factory for creating propagation models"""
    
    def __init__(self):
        self.models = {
            'PekerisRayTracer': PekerisRayTracerModel(),
            'PekerisModeSolver': PekerisModeSolverModel(),
            'Kraken': KrakenModel(),
            'Orca': OrcaModel(),
            'Bellhop': BellhopModel(),
            'RaySolver': RaySolverModel(),
        }
        
        # Weighted selection probabilities
        self.weights = {
            'PekerisRayTracer': 0.375,  # 3/8
            'PekerisModeSolver': 0.125,  # 1/8
            'Bellhop': 0.125,  # 1/8
            'Kraken': 0.125,  # 1/8
            'Orca': 0.125,  # 1/8
            'RaySolver': 0.125  # 1/8
        }
    
    def select_model(self, env_params: EnvironmentParams,
                     config: Dict, 
                     geometry_params: Optional[GeometryParams] = None) -> Tuple[str, PropagationModel]:
        """Select a compatible propagation model"""
        # Filter compatible models
        compatible_models = {
            name: model for name, model in self.models.items()
            if model.is_compatible(env_params, geometry_params)
        }

        # compatible_models Intersection config.propagation_model
        compatible_models = {
            name: model for name, model in compatible_models.items()
            if name == config.get('propagation_model')
        }

        if not compatible_models:
            # Fallback to RayTracer
            return 'PekerisRayTracer', self.models['PekerisRayTracer']
        
        # Use weighted random selection among compatible models
        names = list(compatible_models.keys())
        weights = [self.weights.get(name, 0.1) for name in names]
        weights = np.array(weights) / sum(weights)  # Normalize
        
        selected_name = np.random.choice(names, p=weights)
        return selected_name, compatible_models[selected_name]


class SignalGenerator:
    """Signal generation utilities"""
    
    @staticmethod
    def generate_signal(signal_params: SignalParams, fs: int) -> Tuple[np.ndarray, Dict]:
        """Generate diverse signal types"""
        duration = signal_params.duration
        frequency = int(signal_params.frequency)
        bandwidth = int(signal_params.bandwidth)
        signal_type = signal_params.type
        
        samples = int(duration * fs)
        t = np.arange(samples) / fs
        
        generators = {
            'chirp': SignalGenerator._generate_chirp,
            'pulse': SignalGenerator._generate_pulse,
            'msequence': SignalGenerator._generate_msequence,
            'ofdm': SignalGenerator._generate_ofdm,
            'tone': SignalGenerator._generate_tone
        }
        
        generator = generators.get(signal_type, SignalGenerator._generate_pulse)
        signal = generator(t, frequency, bandwidth, duration, samples)
        
        return signal, {
            'type': signal_type,
            'duration': duration,
            'frequency': frequency,
            'bandwidth': bandwidth,
            'samples': samples
        }
    
    @staticmethod
    def _generate_chirp(t, frequency, bandwidth, duration, samples):
        """ 线性调频信号 """
        f_start = frequency - bandwidth/2
        f_end = frequency + bandwidth/2
        signal = np.sin(2 * np.pi * (f_start * t + 
                                   (f_end - f_start) * t**2 / (2 * duration)))
        return signal
    
    @staticmethod
    def _generate_pulse(t, frequency, bandwidth, duration, samples):
        """ 高斯包络脉冲信号 """
        center = duration / 2
        sigma = duration / 8
        envelope = np.exp(-((t - center) / sigma)**2)
        carrier = np.sin(2 * np.pi * frequency * t)
        return envelope * carrier
    
    @staticmethod
    def _generate_msequence(t, frequency, bandwidth, duration, samples):
        """ 伪随机二进制序列 BPSK """
        fs = len(t) / duration
        bits_per_second = bandwidth
        samples_per_bit = int(fs / bits_per_second)
        num_bits = max(1, samples // samples_per_bit)
        bits = np.random.choice([-1, 1], num_bits)
        
        digital_signal = np.repeat(bits, samples_per_bit)[:samples]
        carrier = np.sin(2 * np.pi * frequency * t)

        if len(digital_signal) < samples:
            digital_signal = np.resize(digital_signal, samples)
        else:
            digital_signal = digital_signal[:samples]
        return digital_signal * carrier
    
    @staticmethod
    def _generate_ofdm(t, frequency, bandwidth, duration, samples):
        """ 正交频分复用 OFDM 信号 """
        num_carriers = min(64, int(bandwidth / 100))
        subcarriers = np.zeros(samples, dtype=complex)
        
        for i in range(num_carriers):
            f_sub = frequency - bandwidth/2 + i * (bandwidth/num_carriers)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 1.0)
            subcarriers += amplitude * np.exp(1j * (2 * np.pi * f_sub * t + phase))
        
        return np.real(subcarriers)
    
    @staticmethod
    def _generate_tone(t, frequency, bandwidth, duration, samples):
        """ 纯音信号 """
        return np.sin(2 * np.pi * frequency * t)


class EnvironmentGenerator:
    """Environment generation utilities"""
    
    @staticmethod
    def generate_environment(config: Dict) -> Tuple[object, EnvironmentParams]:
        """Generate diverse underwater environments"""
        # Bathymetry based on type
        bathymetry_ranges = {
            'shallow': (10, 50),
            'medium': (50, 200),
            'deep': (200, 1000),
            'very_deep': (1000, 4000)
        }
        
        bathymetry_type = config.get('bathymetry_type', 'medium')
        min_depth, max_depth = bathymetry_ranges.get(bathymetry_type, (20, 200))
        bathymetry = np.random.uniform(min_depth, max_depth)
        
        # Seabed selection
        if config.get('use_elastic_seabed', False) and np.random.random() < 0.3:
            seabed_type = np.random.choice(BoundaryConditions.ELASTIC_SEABED_TYPES)
        else:
            seabed_type = np.random.choice(BoundaryConditions.SEABED_TYPES)
        
        seabed = getattr(jl, seabed_type)
        
        # Surface conditions
        surface = None
        surface_type = None
        if config.get('include_surface_effects', False):
            surface_type = np.random.choice(BoundaryConditions.SEA_STATES)
            surface = getattr(jl, surface_type)
        
        # Environmental parameters
        temperature = np.random.uniform(0, 30)
        salinity = np.random.uniform(30, 40)
        pH = np.random.uniform(7.5, 8.5)
        
        # Sound speed calculation
        soundspeed = EnvironmentGenerator._calculate_soundspeed(
            config.get('soundspeed_profile', 'iso_velocity'), 
            temperature, salinity, bathymetry
        )
        
        # Create Julia environment
        env_params_dict = {
            'bathymetry': bathymetry,
            'seabed': seabed,
            'soundspeed': soundspeed,
            'temperature': temperature,
            'salinity': salinity,
            'pH': pH
        }
        
        if surface is not None:
            env_params_dict['surface'] = surface
        
        env = jl.UnderwaterEnvironment(**env_params_dict)
        
        env_params = EnvironmentParams(
            bathymetry=bathymetry,
            seabed_type=seabed_type,
            surface_type=surface_type,
            temperature=temperature,
            salinity=salinity,
            pH=pH,
            soundspeed_profile=config.get('soundspeed_profile', 'iso_velocity')
        )
        
        return env, env_params
    
    @staticmethod
    def _calculate_soundspeed(profile_type: str, temperature: float, 
                            salinity: float, bathymetry: float) -> float:
        """Calculate sound speed based on profile type"""
        if profile_type == 'iso_velocity':
            return jl.soundspeed(temperature, salinity, 0)
        elif profile_type == 'linear':
            surface_speed = jl.soundspeed(temperature, salinity, 0)
            return surface_speed  # Simplified
        elif profile_type == 'munk':
            return jl.soundspeed(temperature, salinity, 0)  # Simplified
        else:
            return jl.soundspeed(temperature, salinity, 0)


class GeometryGenerator:
    """Geometry generation utilities"""
    
    @staticmethod
    def generate_transceiver_geometry(config: Dict, bathymetry: Optional[float] = None) -> Tuple[object, object, GeometryParams]:
        """Generate diverse transmitter and receiver geometries, with bottom safety check"""
        geometry_type = config.get('geometry_type', 'random')
        geometry_functions = {
            'fixed_source': GeometryGenerator._fixed_source,
            'fixed_receiver': GeometryGenerator._fixed_receiver,
            'both_moving': GeometryGenerator._both_moving,
            'vertical_array': GeometryGenerator._vertical_array,
            'surface_bottom': GeometryGenerator._surface_bottom,
            'long_range': GeometryGenerator._long_range,
            'random': GeometryGenerator._random_geometry
        }
        generator = geometry_functions.get(geometry_type, GeometryGenerator._random_geometry)
        tx_x, tx_z, rx_x, rx_z = generator()
        
        # Signal parameters
        frequency = 12000
        spl = np.random.uniform(140, 200)
        
        # --- Bottom safety check ---
        min_depth = -5.0  # 5m above bottom
        if bathymetry is not None:
            max_depth = -bathymetry + 5.0
            tx_z = min(tx_z, min_depth)
            tx_z = max(tx_z, max_depth)
            rx_z = min(rx_z, min_depth)
            rx_z = max(rx_z, max_depth)
        
        # Create Julia objects
        tx = jl.AcousticSource(tx_x, tx_z, frequency, spl=spl)
        rx = jl.AcousticReceiver(rx_x, rx_z)
        
        geometry_params = GeometryParams(
            geometry_type=geometry_type,
            tx_position=(tx_x, tx_z),
            rx_position=(rx_x, rx_z),
            frequency=frequency,
            spl=spl
        )
        
        return tx, rx, geometry_params
    
    @staticmethod
    def _fixed_source():
        tx_x, tx_z = 0.0, np.random.uniform(-20, -2)
        rx_x = np.random.uniform(50, 2000)
        rx_z = np.random.uniform(-50, -1)
        return tx_x, tx_z, rx_x, rx_z
    
    @staticmethod
    def _fixed_receiver():
        tx_x = 0.0
        tx_z = np.random.uniform(-50, -1)
        rx_x, rx_z = np.random.uniform(500, 1500), np.random.uniform(-30, -5)
        return tx_x, tx_z, rx_x, rx_z
    
    @staticmethod
    def _both_moving():
        tx_x = 0.0
        tx_z = np.random.uniform(-100, -1)
        rx_x = np.random.uniform(100, 2000)
        rx_z = np.random.uniform(-100, -1)
        return tx_x, tx_z, rx_x, rx_z
    
    @staticmethod
    def _vertical_array():
        tx_x, tx_z = 0.0, np.random.uniform(-50, -5)
        rx_x = np.random.uniform(200, 1000)
        rx_z = np.random.uniform(-50, -5)
        return tx_x, tx_z, rx_x, rx_z
    
    @staticmethod
    def _surface_bottom():
        if np.random.random() < 0.5:
            tx_x, tx_z = 0.0, -1
            rx_z = np.random.uniform(-200, -50)
        else:
            tx_x, tx_z = 0.0, np.random.uniform(-200, -50)
            rx_z = -1
        rx_x = np.random.uniform(500, 2000)
        return tx_x, tx_z, rx_x, rx_z
    
    @staticmethod
    def _long_range():
        tx_x, tx_z = 0.0, np.random.uniform(-100, -10)
        rx_x = np.random.uniform(5000, 20000)
        rx_z = np.random.uniform(-200, -10)
        return tx_x, tx_z, rx_x, rx_z
    
    @staticmethod
    def _random_geometry():
        tx_x = 0.0
        tx_z = np.random.uniform(-200, -1)
        rx_x = np.random.uniform(100, 5000)
        rx_z = np.random.uniform(-200, -1)
        return tx_x, tx_z, rx_x, rx_z


class PositionAdjuster:
    """Position adjustment for model constraints"""
    
    @staticmethod
    def adjust_positions_for_model(model_name: str, tx_pos: Tuple[float, float], 
                                 rx_pos: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Adjust positions based on propagation model constraints"""
        if model_name in ['Kraken', 'Orca']:
            return PositionAdjuster._adjust_for_kraken_orca(tx_pos, rx_pos)
        return tx_pos, rx_pos
    
    @staticmethod
    def _adjust_for_kraken_orca(tx_pos: Tuple[float, float], 
                              rx_pos: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Adjust positions for Kraken/Orca (transmitter at origin, receiver in right half-plane)"""
        original_range = np.sqrt((tx_pos[0] - rx_pos[0])**2 + 
                               (tx_pos[1] - rx_pos[1])**2)
        
        tx_adjusted = (0.0, tx_pos[1])  # Keep depth
        rx_adjusted = (original_range, rx_pos[1])  # Keep depth
        
        return tx_adjusted, rx_adjusted


class EnhancedTOADatasetGenerator:
    """Enhanced Generator for diverse TOA dataset using UnderwaterAcoustics.jl"""

    def __init__(self, output_dir: str = "toa_dataset_enhanced", 
                 config: Optional[SimulationConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.signals_dir = self.output_dir / "signals"
        self.signals_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = config or SimulationConfig()
        
        # Components
        self.propagation_factory = PropagationModelFactory()
        
        # Dataset storage
        self.dataset = []

    def generate_scenario_config(self) -> Dict:
        """Generate random scenario configuration"""
        return {
            # Environment diversity
            'bathymetry_type': np.random.choice([
                'shallow', 'medium', 'deep', 'very_deep'
            ]),
            'soundspeed_profile': np.random.choice([
                'iso_velocity', 'linear', 'munk'
            ]),
            'use_elastic_seabed': np.random.random() < 0.3,
            'include_surface_effects': np.random.random() < 0.4,
            
            # Geometry diversity
            'geometry_type': np.random.choice([
                'fixed_source', 'fixed_receiver', 'both_moving',
                'vertical_array', 'surface_bottom', 'long_range', 'random'
            ]),
            
            # Signal diversity
            'signal_type': np.random.choice([
                'pulse', 'chirp', 'msequence', 'ofdm', 'tone'
            ]),
            'signal_duration': np.random.uniform(0.001, 0.01),
            'signal_bandwidth': 500,
            
            # Propagation model: PekerisRayTracer, PekerisModeSolver, Kraken, Orca, Bellhop, RaySolver
            'propagation_model': np.random.choice([
                'PekerisRayTracer', 'RaySolver'
            ]),
            'max_bounces': np.random.randint(1, 15),
            
            # Noise model
            'noise_type': np.random.choice(['white', 'red']),
            'noise_level': np.random.uniform(1e-6, 1e5)
        }

    def generate_single_sample(self, sample_id: int) -> Optional[Dict]:
        """Generate a single diverse dataset sample and immediately append to CSV/JSON"""
        for retry in range(self.config.max_retries):
            try:
                sample = self._attempt_sample_generation(sample_id, retry)
                if sample is not None:
                    # 信号文件已在 _create_sample_dict 内部保存
                    # 追加到 JSON Lines
                    json_path = self.output_dir / "complete_dataset.json"
                    if sample_id == 0:
                        with open(json_path, 'w') as f:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    else:
                        with open(json_path, 'a') as f:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    # 追加到 CSV
                    import pandas as pd
                    csv_path = self.output_dir / "metadata.csv"
                    csv_row = {
                        'sample_id': sample['sample_id'],
                        'rx_signal_file': sample['rx_signal_file'],
                        'tx_signal_file': sample['tx_signal_file'],
                        'tx_x': sample['tx_position'][0],
                        'tx_z': sample['tx_position'][1],
                        'rx_x': sample['rx_position'][0],
                        'rx_z': sample['rx_position'][1],
                        'distance': sample['distance'],
                        'primary_toa': sample['primary_toa'],
                        'num_multipath': sample['num_multipath'],
                        'delay_spread': sample['delay_spread'],
                        'sampling_frequency': sample['sampling_frequency'],
                        'bathymetry': sample['environment']['bathymetry'],
                        'seabed_type': sample['environment']['seabed_type'],
                        'surface_type': sample['environment']['surface_type'],
                        'temperature': sample['environment']['temperature'],
                        'salinity': sample['environment']['salinity'],
                        'soundspeed_profile': sample['environment']['soundspeed_profile'],
                        'signal_type': sample['signal']['type'],
                        'signal_frequency': sample['signal']['frequency'],
                        'signal_bandwidth': sample['signal']['bandwidth'],
                        'signal_duration': sample['signal']['duration'],
                        'geometry_type': sample['geometry']['geometry_type'],
                        'source_frequency': sample['geometry']['frequency'],
                        'spl': sample['geometry']['spl'],
                        'propagation_model': sample['propagation_model']['type'],
                        'noise_type': sample['noise']['type'],
                        'noise_level': sample['noise']['level']
                    }
                    # 首次写入表头
                    if sample_id == 0:
                        df = pd.DataFrame([csv_row])
                        df.to_csv(csv_path, index=False, mode='w')
                    else:
                        df = pd.DataFrame([csv_row])
                        df.to_csv(csv_path, index=False, mode='a', header=False)
                    self.dataset.append(sample)  # 仅用于统计
                    return sample
            except Exception as e:
                logger.debug(f"Attempt {retry + 1} failed for sample {sample_id}: {str(e)}", stacklevel=2, exc_info=True)
                if retry == self.config.max_retries - 1:
                    logger.error(f"Error generating sample {sample_id}: {str(e)}", stacklevel=2, exc_info=True)
                    return None
        return None

    def _attempt_sample_generation(self, sample_id: int, retry: int) -> Optional[Dict]:
        """Single attempt at sample generation"""
        # Generate scenario configuration
        scenario_config = self.generate_scenario_config()
        
        # Generate environment
        env, env_params = EnvironmentGenerator.generate_environment(scenario_config)
        
        # Generate transceiver geometry
        tx, rx, geometry_params = GeometryGenerator.generate_transceiver_geometry(scenario_config, env_params.bathymetry)
        
        # Get positions and validate geometry
        tx_location = jl.location(tx)
        rx_location = jl.location(rx)
        tx_pos = (float(tx_location.x), float(tx_location.z))
        rx_pos = (float(rx_location.x), float(rx_location.z))
        
        distance = np.sqrt((tx_pos[0] - rx_pos[0])**2 + (tx_pos[1] - rx_pos[1])**2)
        
        if not self._validate_geometry(distance, retry):
            return None
        
        # Select and create propagation model
        model_name, model_instance = self.propagation_factory.select_model(
            env_params, scenario_config, geometry_params)
        
        try:
            pm = model_instance.create_model(env, scenario_config)
            model_params = {'type': model_name, **scenario_config}
        except Exception as e:
            logger.debug(f"{model_name} failed: {e}", stacklevel=2, exc_info=True)
            pm = jl.PekerisRayTracer(env)
            model_params = {'type': 'PekerisRayTracer'}
            model_name = 'PekerisRayTracer'
        
        # Adjust positions for model constraints
        tx_pos, rx_pos = PositionAdjuster.adjust_positions_for_model(model_name, tx_pos, rx_pos)
        
        # # Recreate transceiver if needed
        # if model_name in ['Kraken', 'Orca']:
        #     tx = jl.AcousticSource(tx_pos[0], tx_pos[1], 
        #                          geometry_params.frequency, 
        #                          spl=geometry_params.spl)
        #     rx = jl.AcousticReceiver(rx_pos[0], rx_pos[1])
        #     distance = np.sqrt((tx_pos[0] - rx_pos[0])**2 + (tx_pos[1] - rx_pos[1])**2)
        
        # Generate noise model
        noise = self._create_noise_model(scenario_config)
        
        # Create channel and get arrivals
        ch, rays = self._create_channel_and_arrivals(pm, tx, rx, noise, model_name, retry)
        if ch is None or rays is None:
            logger.debug(f"Channel or arrivals not found for sample {sample_id}", stacklevel=2)
            return None
        
        # Generate signal and extract features
        signal_params = SignalParams(
            type=scenario_config['signal_type'],
            duration=scenario_config['signal_duration'],
            frequency=int(geometry_params.frequency),
            bandwidth=int(scenario_config['signal_bandwidth'])
        )

        toa_array, cir, tx_signal, features = self._extract_toa_and_cir(
            rays, ch, distance, signal_params, env_params
        )
        
        if not self._validate_results(toa_array, features, retry):
            logger.debug(f"Invalid results for sample {sample_id}", stacklevel=2)
            return None
        
        # Save and return sample
        return self._create_sample_dict(
            sample_id, toa_array, cir, features, tx_pos, rx_pos, distance,
            env_params, geometry_params, signal_params, model_params, 
            scenario_config, tx_signal
        )

    def _validate_geometry(self, distance: float, retry: int) -> bool:
        """Validate geometry constraints"""
        if distance < 10 or distance > 50000:
            if retry < self.config.max_retries - 1:
                return False
            return False
        return True

    def _create_noise_model(self, config: Dict):
        """Create noise model"""
        noise_type = config.get('noise_type', 'red')
        noise_level = config.get('noise_level', np.random.uniform(1e-8, 1e-3))
        
        if noise_type == 'white':
            return jl.WhiteGaussianNoise(noise_level)
        else:
            return jl.RedGaussianNoise(noise_level)

    def _create_channel_and_arrivals(self, pm, tx, rx, noise, model_name: str, retry: int):
        """Create channel and get ray arrivals with timeout protection"""
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Channel creation timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.config.timeout_seconds)
            
            try:
                ch = jl.channel(pm, tx, rx, self.config.fs, noise=noise)
                rays = jl.arrivals(pm, tx, rx)
                signal.alarm(0)
                
                if len(rays) == 0:
                    logger.debug("No ray arrivals found")
                    return None, None
                
                return ch, rays
                
            except TimeoutError:
                logger.debug(f"Channel creation timed out for {model_name}")
                signal.alarm(0)
                return None, None
                
        except Exception as e:
            signal.alarm(0)
            logger.debug(f"Channel creation failed: {e}")
            return None, None

    def _extract_toa_and_cir(self, rays, ch, distance: float, 
                           signal_params: SignalParams, env_params: EnvironmentParams):
        """Extract TOA and CIR data with enhanced signal processing"""
        # Extract ray information
        ray_times, ray_amplitudes, ray_phases = self._process_ray_arrivals(rays)
        
        # Generate transmitted signal
        tx_signal, _ = SignalGenerator.generate_signal(signal_params, self.config.fs)
        
        # Calculate primary TOA
        primary_toa = min(ray_times) if ray_times else 0.0

        start = max(0, int(primary_toa * self.config.fs) - 100)
        # end = start + int(self.config.signal_duration * self.config.fs)
        
        # Generate CIR using Julia's channel simulation
        try:
            received_signal = jl.transmit(ch, tx_signal, fs=self.config.fs, abstime=True)
            cir = np.abs(np.array(received_signal).squeeze()) # Use absolute value to avoid complex issues
            cir = cir[max(0, start):min(len(cir), start + int(self.config.signal_duration * self.config.fs))]
            primary_toa = (primary_toa * self.config.fs - start) / self.config.fs
        except Exception as e:
            logger.warning(f"Julia transmit failed: {e}, using manual CIR construction")
            cir = self._manual_cir_construction(ray_times, ray_amplitudes, ray_phases, tx_signal)
        
        # Calculate features
        features = self._calculate_features(ray_times, ray_amplitudes, primary_toa, tx_signal, cir)
        
        return np.array(ray_times), cir, tx_signal, features

    def _process_ray_arrivals(self, rays) -> Tuple[List[float], List[float], List[float]]:
        """Process ray arrivals to extract times, amplitudes, and phases"""
        ray_times, ray_amplitudes, ray_phases = [], [], []
        
        for i, ray in enumerate(rays):
            try:
                # Extract arrival time
                if hasattr(ray, 'time'):
                    arrival_time = float(ray.time)
                elif hasattr(ray, 't'):
                    arrival_time = float(ray.t)
                else:
                    continue
                
                # Extract phasor information
                if hasattr(ray, 'phasor'):
                    phasor = ray.phasor
                elif hasattr(ray, 'amplitude'):
                    amp = getattr(ray, 'amplitude', 1.0)
                    phase = getattr(ray, 'phase', 0.0)
                    phasor = amp * np.exp(1j * phase)
                else:
                    phasor = 1.0 + 0j
                
                amplitude_db = float(abs(phasor))
                amplitude_linear = 10**(amplitude_db/20.0) if amplitude_db > 0 else 1e-10
                phase = float(np.angle(phasor))
                
                ray_times.append(arrival_time)
                ray_amplitudes.append(amplitude_linear)
                ray_phases.append(phase)
                
            except Exception as e:
                logger.debug(f"Skipping arrival {i}: {e}")
                continue
        
        return ray_times, ray_amplitudes, ray_phases

    def _manual_cir_construction(self, ray_times: List[float], ray_amplitudes: List[float], 
                               ray_phases: List[float], tx_signal: np.ndarray) -> np.ndarray:
        """Manual CIR construction fallback"""
        total_length = int(self.config.fs * self.config.signal_duration)
        cir = np.zeros(total_length, dtype=complex)
        
        for t, amp, phase in zip(ray_times, ray_amplitudes, ray_phases):
            if t * self.config.fs < total_length:
                delay_samples = int(t * self.config.fs)
                end_idx = min(delay_samples + len(tx_signal), total_length)
                actual_signal_samples = end_idx - delay_samples
                
                if actual_signal_samples > 0:
                    signal_part = tx_signal[:actual_signal_samples]
                    phase_factor = np.exp(1j * phase)
                    delayed_signal = signal_part * amp * phase_factor
                    cir[delay_samples:end_idx] += delayed_signal
        
        # Add noise and convert to real
        noise_level = 0.01 * np.max(np.abs(cir)) if np.max(np.abs(cir)) > 0 else 0.001
        noise = noise_level * (np.random.randn(total_length) + 
                             1j * np.random.randn(total_length))
        cir += noise
        return np.real(cir)

    def _calculate_features(self, ray_times: List[float], ray_amplitudes: List[float], 
                          primary_toa: float, tx_signal: np.ndarray, cir: np.ndarray) -> Dict:
        """Calculate signal features"""
        return {
            'num_rays': len(ray_times),
            'primary_toa': primary_toa,
            'max_amplitude': max(ray_amplitudes) if ray_amplitudes else 0.0,
            'rms_amplitude': (np.sqrt(np.mean(np.array(ray_amplitudes)**2))
                              if ray_amplitudes else 0.0),
            'delay_spread': (max(ray_times) - min(ray_times)
                             if len(ray_times) > 1 else 0.0),
            'actual_duration': len(tx_signal) / self.config.fs,
            'signal_duration': len(cir) / self.config.fs
        }

    def _validate_results(self, toa_array: np.ndarray, features: Dict, retry: int) -> bool:
        """Validate extraction results"""
        if len(toa_array) == 0 or features['primary_toa'] <= 0:
            if retry < self.config.max_retries - 1:
                return False
            return False
        return True

    def _create_sample_dict(self, sample_id: int, toa_array: np.ndarray, cir: np.ndarray,
                          features: Dict, tx_pos: Tuple[float, float], rx_pos: Tuple[float, float],
                          distance: float, env_params: EnvironmentParams, 
                          geometry_params: GeometryParams, signal_params: SignalParams,
                          model_params: Dict, scenario_config: Dict, tx_signal: np.ndarray = None) -> Dict:
        """Create final sample dictionary"""
        # Save CIR signal
        rx_signal_filename = f"rx_signal_{sample_id:06d}.npy"
        rx_signal_path = self.signals_dir / rx_signal_filename
        np.save(rx_signal_path, cir)
        # Save TX signal
        tx_signal_filename = f"tx_signal_{sample_id:06d}.npy"
        tx_signal_path = self.signals_dir / tx_signal_filename
        if tx_signal is not None:
            np.save(tx_signal_path, tx_signal)
        return {
            'sample_id': sample_id,
            'rx_signal_file': rx_signal_filename,
            'tx_signal_file': tx_signal_filename,
            'tx_position': tx_pos,
            'rx_position': rx_pos,
            'distance': distance,
            'toa_values': toa_array.tolist(),
            'primary_toa': features['primary_toa'],
            'num_multipath': features['num_rays'],
            'delay_spread': features['delay_spread'],
            'max_amplitude': features['max_amplitude'],
            'rms_amplitude': features['rms_amplitude'],
            'sampling_frequency': self.config.fs,
            'signal_duration': features['signal_duration'],
            'actual_duration': features['actual_duration'],
            # Structured parameters
            'environment': env_params.__dict__,
            'geometry': geometry_params.__dict__,
            'signal': signal_params.__dict__,
            'propagation_model': model_params,
            'noise': {
                'type': scenario_config['noise_type'],
                'level': scenario_config['noise_level']
            },
            'timestamp': time.time(),
            'config': scenario_config
        }

    def generate_dataset(self, num_samples: int = 1000) -> None:
        """Generate complete diverse dataset, saving each sample immediately"""
        logger.info(f"Generating {num_samples} diverse samples...")
        for i in tqdm(range(num_samples), desc="Generating enhanced TOA dataset"):
            self.generate_single_sample(i)
        logger.info(f"Successfully generated {len(self.dataset)} samples")
        self._create_readme()
        logger.info(f"Enhanced dataset saved to {self.output_dir}")
        logger.info(f"- {len(self.dataset)} samples")

    def save_dataset(self) -> None:
        """Save dataset with comprehensive metadata"""
        # Save complete dataset as JSON
        dataset_file = self.output_dir / "complete_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        
        # Create CSV metadata if pandas available
        try:
            self._save_csv_metadata()
        except ImportError:
            logger.warning("pandas not available, skipping CSV export")
        
        # Create README
        self._create_readme()
        
        logger.info(f"Enhanced dataset saved to {self.output_dir}")
        logger.info(f"- {len(self.dataset)} samples")

    def _save_csv_metadata(self):
        """Save CSV metadata using pandas"""
        import pandas as pd
        
        csv_data = []
        for sample in self.dataset:
            csv_row = {
                'sample_id': sample['sample_id'],
                'rx_signal_file': sample['rx_signal_file'],
                'tx_signal_file': sample['tx_signal_file'],
                'tx_x': sample['tx_position'][0],
                'tx_z': sample['tx_position'][1],
                'rx_x': sample['rx_position'][0],
                'rx_z': sample['rx_position'][1],
                'distance': sample['distance'],
                'primary_toa': sample['primary_toa'],
                'num_multipath': sample['num_multipath'],
                'delay_spread': sample['delay_spread'],
                'sampling_frequency': sample['sampling_frequency'],
                
                # Environmental
                'bathymetry': sample['environment']['bathymetry'],
                'seabed_type': sample['environment']['seabed_type'],
                'surface_type': sample['environment']['surface_type'],
                'temperature': sample['environment']['temperature'],
                'salinity': sample['environment']['salinity'],
                'soundspeed_profile': sample['environment']['soundspeed_profile'],
                
                # Signal
                'signal_type': sample['signal']['type'],
                'signal_frequency': sample['signal']['frequency'],
                'signal_bandwidth': sample['signal']['bandwidth'],
                'signal_duration': sample['signal']['duration'],
                
                # Geometry
                'geometry_type': sample['geometry']['geometry_type'],
                'source_frequency': sample['geometry']['frequency'],
                'spl': sample['geometry']['spl'],
                
                # Propagation
                'propagation_model': sample['propagation_model']['type'],
                
                # Noise
                'noise_type': sample['noise']['type'],
                'noise_level': sample['noise']['level']
            }
            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / "metadata.csv"
        df.to_csv(csv_file, index=False)

    def _create_readme(self):
        """Create comprehensive README"""
        readme_content = f"""# Enhanced TOA Dataset (Refactored)

## Description
This dataset contains diverse Channel Impulse Response (CIR) and Time of Arrival (TOA)
data for underwater acoustic communication scenarios, generated using UnderwaterAcoustics.jl
with maximum parameter diversity.

## Code Architecture

### Key Improvements
- **Modular Design**: Separated concerns into specialized classes
- **Type Safety**: Added dataclasses and type hints for better code clarity
- **Extensibility**: Abstract base classes allow easy addition of new propagation models
- **Error Handling**: Robust error handling with proper logging
- **Configuration**: Centralized configuration management

### Class Structure
- `PropagationModel`: Abstract base for propagation models
- `PropagationModelFactory`: Factory pattern for model selection
- `SignalGenerator`: Static methods for various signal types
- `EnvironmentGenerator`: Environment creation utilities
- `GeometryGenerator`: Transceiver positioning utilities
- `PositionAdjuster`: Model-specific position adjustments
- `EnhancedTOADatasetGenerator`: Main orchestrator class

## Dataset Statistics
- Total samples: {len(self.dataset)}
- Sampling frequency: {self.config.fs} Hz
- Base signal duration: {self.config.signal_duration} s
- Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Usage Example
```python
from enhanced_toa_generator_refactored import EnhancedTOADatasetGenerator, SimulationConfig

# Custom configuration
config = SimulationConfig(
    fs=96000,
    signal_duration=0.5,
    timeout_seconds=60
)

# Generate dataset
generator = EnhancedTOADatasetGenerator('my_dataset', config)
generator.generate_dataset(num_samples=100)
generator.save_dataset()
```

## Extension Points

### Adding New Propagation Models
```python
class MyNewModel(PropagationModel):
    def is_compatible(self, env_params: EnvironmentParams) -> bool:
        # Define compatibility constraints
        return True
    
    def create_model(self, env, config: Dict):
        # Create model instance
        return jl.MyNewModel(env)
    
    def get_name(self) -> str:
        return 'MyNewModel'

# Register in factory
factory.models['MyNewModel'] = MyNewModel()
```

### Adding New Signal Types
```python
@staticmethod
def _generate_my_signal(t, frequency, bandwidth, duration, samples):
    # Implement signal generation
    return signal

# Add to SignalGenerator.generate_signal generators dict
```
"""
        
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)


def main():
    """Main function with improved argument parsing"""
    parser = argparse.ArgumentParser(
        description="Generate enhanced diverse TOA dataset using UnderwaterAcoustics.jl (Refactored)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="toa_dataset_test",
                        help="Output directory for dataset")
    parser.add_argument("--fs", type=int, default=32000,
                        help="Sampling frequency (Hz)")
    parser.add_argument("--signal_duration", type=float, default=0.1,
                        help="Base signal duration (s)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout for channel creation (seconds)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Maximum retry attempts per sample")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create configuration
    config = SimulationConfig(
        fs=args.fs,
        signal_duration=args.signal_duration,
        max_retries=args.retries,
        timeout_seconds=args.timeout
    )

    # Create generator and run
    generator = EnhancedTOADatasetGenerator(args.output_dir, config)
    generator.generate_dataset(num_samples=args.num_samples)
    # generator.save_dataset()

    print("Enhanced TOA dataset generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {len(generator.dataset)} samples")
    print("Refactored code provides better maintainability and extensibility")


if __name__ == "__main__":
    main()

# TODO: CIR全部转为正值，TOA索引需要随机化，不能固定在100个点