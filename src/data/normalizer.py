"""Data normalization utilities."""

import numpy as np
import json
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class Normalizer:
    """Normalize data and store statistics for denormalization."""

    def __init__(self,
                 method: str = 'minmax',
                 feature_range: Tuple[float, float] = (-1, 1)):
        """
        Initialize Normalizer.

        Args:
            method: Normalization method ('minmax', 'standard', 'robust')
            feature_range: Target range for minmax scaling
        """
        self.method = method
        self.feature_range = feature_range
        self.statistics: Optional[Dict[str, Any]] = None
        self.scaler = None

        self._create_scaler()

    def _create_scaler(self):
        """Create sklearn scaler based on method."""
        if self.method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
        elif self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def fit(self, data: np.ndarray) -> 'Normalizer':
        """
        Fit normalizer to data.

        Args:
            data: Input data (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            Self for method chaining
        """
        # Handle 3D data (windowed)
        original_shape = data.shape
        if data.ndim == 3:
            n_windows, seq_len, n_features = data.shape
            data = data.reshape(-1, n_features)

        # Fit scaler
        self.scaler.fit(data)

        # Store statistics
        self.statistics = self._extract_statistics()

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalizer.

        Args:
            data: Input data (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            Normalized data
        """
        if self.scaler is None or self.statistics is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        # Handle 3D data
        original_shape = data.shape
        if data.ndim == 3:
            n_windows, seq_len, n_features = data.shape
            data_2d = data.reshape(-1, n_features)
            normalized_2d = self.scaler.transform(data_2d)
            normalized = normalized_2d.reshape(n_windows, seq_len, n_features)
        else:
            normalized = self.scaler.transform(data)

        return normalized

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit normalizer and transform data.

        Args:
            data: Input data

        Returns:
            Normalized data
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data.

        Args:
            data: Normalized data (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            Original scale data
        """
        if self.scaler is None or self.statistics is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        # Handle 3D data
        original_shape = data.shape
        if data.ndim == 3:
            n_windows, seq_len, n_features = data.shape
            data_2d = data.reshape(-1, n_features)
            denormalized_2d = self.scaler.inverse_transform(data_2d)
            denormalized = denormalized_2d.reshape(n_windows, seq_len, n_features)
        else:
            denormalized = self.scaler.inverse_transform(data)

        return denormalized

    def _extract_statistics(self) -> Dict[str, Any]:
        """
        Extract statistics from fitted scaler.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'method': self.method,
            'feature_range': self.feature_range
        }

        if isinstance(self.scaler, MinMaxScaler):
            stats['min'] = self.scaler.data_min_.tolist()
            stats['max'] = self.scaler.data_max_.tolist()
            stats['scale'] = self.scaler.scale_.tolist()
            stats['data_range'] = self.scaler.data_range_.tolist()

        elif isinstance(self.scaler, StandardScaler):
            stats['mean'] = self.scaler.mean_.tolist()
            stats['std'] = self.scaler.scale_.tolist()
            stats['var'] = self.scaler.var_.tolist()

        elif isinstance(self.scaler, RobustScaler):
            stats['center'] = self.scaler.center_.tolist()
            stats['scale'] = self.scaler.scale_.tolist()

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get normalization statistics.

        Returns:
            Statistics dictionary
        """
        if self.statistics is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        return self.statistics

    def save_statistics(self, filepath: str) -> None:
        """
        Save normalization statistics to JSON file.

        Args:
            filepath: Path to save file
        """
        if self.statistics is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.statistics, f, indent=2)

    def load_statistics(self, filepath: str) -> 'Normalizer':
        """
        Load normalization statistics from JSON file.

        Args:
            filepath: Path to statistics file

        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            self.statistics = json.load(f)

        # Recreate scaler from statistics
        self.method = self.statistics['method']
        self.feature_range = tuple(self.statistics['feature_range'])
        self._create_scaler()

        # Set scaler attributes
        if isinstance(self.scaler, MinMaxScaler):
            self.scaler.data_min_ = np.array(self.statistics['min'])
            self.scaler.data_max_ = np.array(self.statistics['max'])
            self.scaler.scale_ = np.array(self.statistics['scale'])
            self.scaler.data_range_ = np.array(self.statistics['data_range'])
            self.scaler.min_ = -self.scaler.data_min_ * self.scaler.scale_
            self.scaler.n_features_in_ = len(self.scaler.data_min_)
            self.scaler.n_samples_seen_ = 1  # Dummy value

        elif isinstance(self.scaler, StandardScaler):
            self.scaler.mean_ = np.array(self.statistics['mean'])
            self.scaler.scale_ = np.array(self.statistics['std'])
            self.scaler.var_ = np.array(self.statistics['var'])
            self.scaler.n_features_in_ = len(self.scaler.mean_)
            self.scaler.n_samples_seen_ = 1  # Dummy value

        elif isinstance(self.scaler, RobustScaler):
            self.scaler.center_ = np.array(self.statistics['center'])
            self.scaler.scale_ = np.array(self.statistics['scale'])
            self.scaler.n_features_in_ = len(self.scaler.center_)

        return self

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Normalizer':
        """
        Create Normalizer from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Normalizer instance
        """
        norm_config = config.get('normalization', {})

        method = norm_config.get('method', 'minmax')
        feature_range = tuple(norm_config.get('feature_range', [-1, 1]))

        return cls(method=method, feature_range=feature_range)
