"""TensorFlow dataset creation utilities."""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
from pathlib import Path


class DatasetBuilder:
    """Create TensorFlow datasets for training."""

    def __init__(self,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 seed: int = 42):
        """
        Initialize DatasetBuilder.

        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def build(self,
              windows: np.ndarray,
              split_ratios: List[float] = [0.7, 0.15, 0.15]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Build train, validation, and test datasets from windowed data.

        Args:
            windows: Windowed data (n_windows, window_size, n_features)
            split_ratios: [train_ratio, val_ratio, test_ratio]

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        n_windows = len(windows)

        # Shuffle indices if requested
        if self.shuffle:
            np.random.seed(self.seed)
            indices = np.random.permutation(n_windows)
            windows = windows[indices]

        # Calculate split indices
        train_end = int(n_windows * split_ratios[0])
        val_end = train_end + int(n_windows * split_ratios[1])

        # Split data
        train_data = windows[:train_end]
        val_data = windows[train_end:val_end]
        test_data = windows[val_end:]

        # Create datasets (autoencoder: input = output)
        train_dataset = self._create_dataset(train_data, train_data, is_training=True)
        val_dataset = self._create_dataset(val_data, val_data, is_training=False)
        test_dataset = self._create_dataset(test_data, test_data, is_training=False)

        return train_dataset, val_dataset, test_dataset

    def _create_dataset(self,
                       inputs: np.ndarray,
                       outputs: np.ndarray,
                       is_training: bool) -> tf.data.Dataset:
        """
        Create TensorFlow dataset.

        Args:
            inputs: Input data
            outputs: Output data (targets)
            is_training: Whether this is training data

        Returns:
            TensorFlow dataset
        """
        # Convert to float32
        inputs = inputs.astype(np.float32)
        outputs = outputs.astype(np.float32)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

        # Shuffle if training
        if is_training and self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(inputs), seed=self.seed)

        # Batch
        dataset = dataset.batch(self.batch_size)

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def from_processed_file(self,
                           filepath: str,
                           window_size: int,
                           window_stride: int,
                           split_ratios: List[float] = [0.7, 0.15, 0.15]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Build datasets directly from processed data file.

        Args:
            filepath: Path to processed data file (.npz)
            window_size: Window size
            window_stride: Window stride
            split_ratios: [train_ratio, val_ratio, test_ratio]

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load processed data
        loaded = np.load(filepath, allow_pickle=True)
        data = loaded['data']

        # Create windows
        windows = self._create_windows(data, window_size, window_stride)

        # Build datasets
        return self.build(windows, split_ratios)

    def _create_windows(self, data: np.ndarray, window_size: int, window_stride: int) -> np.ndarray:
        """
        Create sliding windows from data.

        Args:
            data: Input data (n_samples, n_features)
            window_size: Window size
            window_stride: Window stride

        Returns:
            Windowed data (n_windows, window_size, n_features)
        """
        n_samples, n_features = data.shape
        n_windows = (n_samples - window_size) // window_stride + 1

        if n_windows <= 0:
            raise ValueError(f"Not enough samples for windowing. Need at least {window_size} samples.")

        windows = np.zeros((n_windows, window_size, n_features))

        for i in range(n_windows):
            start_idx = i * window_stride
            end_idx = start_idx + window_size
            windows[i] = data[start_idx:end_idx]

        return windows

    @classmethod
    def from_config(cls, config: dict) -> 'DatasetBuilder':
        """
        Create DatasetBuilder from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            DatasetBuilder instance
        """
        training_config = config.get('training', {})
        proc_config = config.get('data_processing', {})

        batch_size = training_config.get('batch_size', 32)
        shuffle = True  # Always shuffle for training
        seed = proc_config.get('shuffle_seed', 42)

        return cls(batch_size=batch_size, shuffle=shuffle, seed=seed)
