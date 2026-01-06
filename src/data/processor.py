"""Data processing utilities for downsampling, feature extraction, and windowing."""

import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from scipy import signal as scipy_signal


@dataclass
class ProcessedData:
    """Container for processed data."""
    data: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


class DataProcessor:
    """Process raw simulation data: downsample, derive features, create windows."""

    def __init__(self,
                 source_rate: int,
                 target_rate: int,
                 derived_features: Dict[str, List[str]],
                 window_size: int,
                 window_stride: int,
                 input_variables: List[str]):
        """
        Initialize DataProcessor.

        Args:
            source_rate: Source sampling rate (Hz)
            target_rate: Target sampling rate (Hz)
            derived_features: Dict mapping variable names to list of features to derive
            window_size: Window size for creating sequences
            window_stride: Stride for sliding window
            input_variables: List of variables to include
        """
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.derived_features = derived_features
        self.window_size = window_size
        self.window_stride = window_stride
        self.input_variables = input_variables

        self.downsample_factor = source_rate // target_rate
        if source_rate % target_rate != 0:
            raise ValueError(f"Source rate {source_rate} must be divisible by target rate {target_rate}")

    def process(self, raw_data: Dict[str, np.ndarray]) -> ProcessedData:
        """
        Process raw data: downsample and derive features.

        Args:
            raw_data: Dictionary with 'time', 'voltage', 'current', 'angular_velocity'

        Returns:
            ProcessedData object
        """
        # Downsample and derive features
        processed_dict = {}
        feature_names = []

        for var_name in self.input_variables:
            if var_name not in raw_data:
                raise ValueError(f"Variable {var_name} not found in raw data")

            # Get raw variable
            raw_var = raw_data[var_name]

            # Check if this variable has derived features
            has_derived_features = var_name in self.derived_features and len(self.derived_features[var_name]) > 0

            if has_derived_features:
                # For variables with derived features (e.g., high-freq current),
                # ONLY include derived features, NOT the base downsampled value
                for feature_type in self.derived_features[var_name]:
                    feature_values = self._derive_feature(raw_var, feature_type)
                    feature_name = f"{var_name}_{feature_type}"
                    processed_dict[feature_name] = feature_values
                    feature_names.append(feature_name)
            else:
                # For variables without derived features (e.g., low-freq voltage, velocity),
                # include the downsampled base variable
                downsampled = self._downsample(raw_var)
                processed_dict[var_name] = downsampled
                feature_names.append(var_name)

        # Stack into array: (n_samples, n_features)
        data_arrays = [processed_dict[name] for name in feature_names]
        data = np.column_stack(data_arrays)

        metadata = {
            'source_rate': self.source_rate,
            'target_rate': self.target_rate,
            'downsample_factor': self.downsample_factor,
            'n_samples': data.shape[0],
            'n_features': data.shape[1]
        }

        return ProcessedData(
            data=data,
            feature_names=feature_names,
            metadata=metadata
        )

    def _downsample(self, data: np.ndarray) -> np.ndarray:
        """
        Downsample data by averaging over downsample windows.

        Args:
            data: Input data array

        Returns:
            Downsampled array
        """
        n_samples = len(data)
        n_windows = n_samples // self.downsample_factor

        # Reshape and average
        reshaped = data[:n_windows * self.downsample_factor].reshape(n_windows, self.downsample_factor)
        downsampled = reshaped.mean(axis=1)

        return downsampled

    def _derive_feature(self, data: np.ndarray, feature_type: str) -> np.ndarray:
        """
        Derive feature from raw data over downsample windows.

        Args:
            data: Raw data array
            feature_type: Type of feature ('rms', 'peak_to_peak', 'variance', etc.)

        Returns:
            Feature array (downsampled)
        """
        n_samples = len(data)
        n_windows = n_samples // self.downsample_factor

        # Reshape into windows
        reshaped = data[:n_windows * self.downsample_factor].reshape(n_windows, self.downsample_factor)

        if feature_type == 'rms':
            # Root mean square
            feature = np.sqrt(np.mean(reshaped ** 2, axis=1))

        elif feature_type == 'peak_to_peak':
            # Peak-to-peak (max - min)
            feature = np.max(reshaped, axis=1) - np.min(reshaped, axis=1)

        elif feature_type == 'variance':
            # Variance
            feature = np.var(reshaped, axis=1)

        elif feature_type == 'mean':
            # Mean
            feature = np.mean(reshaped, axis=1)

        elif feature_type == 'max':
            # Maximum
            feature = np.max(reshaped, axis=1)

        elif feature_type == 'min':
            # Minimum
            feature = np.min(reshaped, axis=1)

        elif feature_type == 'slope':
            # Linear regression slope (trend)
            feature = np.array([np.polyfit(np.arange(len(window)), window, 1)[0]
                               for window in reshaped])

        elif feature_type == 'zero_crossing_rate':
            # Zero crossing rate
            feature = np.array([np.sum(np.diff(np.sign(window)) != 0) / len(window)
                               for window in reshaped])

        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        return feature

    def create_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Create sliding windows from data.

        Args:
            data: Input data array (n_samples, n_features)

        Returns:
            Windowed data (n_windows, window_size, n_features)
        """
        n_samples, n_features = data.shape

        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // self.window_stride + 1

        if n_windows <= 0:
            raise ValueError(f"Not enough samples for windowing. Need at least {self.window_size} samples.")

        # Create windows
        windows = np.zeros((n_windows, self.window_size, n_features))

        for i in range(n_windows):
            start_idx = i * self.window_stride
            end_idx = start_idx + self.window_size
            windows[i] = data[start_idx:end_idx]

        return windows

    def save(self, processed_data: ProcessedData, filepath: str) -> None:
        """
        Save processed data to file.

        Args:
            processed_data: ProcessedData object
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save as compressed numpy archive
        np.savez_compressed(
            filepath,
            data=processed_data.data,
            feature_names=processed_data.feature_names,
            metadata=processed_data.metadata
        )

    @staticmethod
    def load(filepath: str) -> ProcessedData:
        """
        Load processed data from file.

        Args:
            filepath: Path to file

        Returns:
            ProcessedData object
        """
        loaded = np.load(filepath, allow_pickle=True)

        return ProcessedData(
            data=loaded['data'],
            feature_names=list(loaded['feature_names']),
            metadata=loaded['metadata'].item()
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DataProcessor':
        """
        Create DataProcessor from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            DataProcessor instance
        """
        proc_config = config.get('data_processing', {})
        sim_config = config.get('simulation', {})

        source_rate = sim_config.get('sampling_rate_hz', 20000)
        target_rate = proc_config.get('target_sampling_rate_hz', 500)
        derived_features = proc_config.get('derived_features', {})
        window_size = proc_config.get('window_size', 64)
        window_stride = proc_config.get('window_stride', 32)
        input_variables = proc_config.get('input_variables', ['current', 'angular_velocity', 'voltage'])

        return cls(
            source_rate=source_rate,
            target_rate=target_rate,
            derived_features=derived_features,
            window_size=window_size,
            window_stride=window_stride,
            input_variables=input_variables
        )

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names.

        Returns:
            List of feature names
        """
        feature_names = []

        for var_name in self.input_variables:
            # Check if this variable has derived features
            has_derived_features = var_name in self.derived_features and len(self.derived_features[var_name]) > 0

            if has_derived_features:
                # Only include derived features
                for feature_type in self.derived_features[var_name]:
                    feature_names.append(f"{var_name}_{feature_type}")
            else:
                # Include base variable
                feature_names.append(var_name)

        return feature_names
