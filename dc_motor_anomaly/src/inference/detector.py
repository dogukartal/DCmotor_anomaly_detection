"""Anomaly detection inference."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    reconstruction_errors: np.ndarray
    is_anomaly: np.ndarray
    threshold: float
    anomaly_scores: np.ndarray
    metadata: Dict[str, Any]


class AnomalyDetector:
    """Load trained model, run inference, detect anomalies."""

    def __init__(self, model: keras.Model, normalizer, threshold: Optional[float] = None):
        """
        Initialize AnomalyDetector.

        Args:
            model: Trained Keras model
            normalizer: Normalizer instance
            threshold: Anomaly threshold (optional, can be set later)
        """
        self.model = model
        self.normalizer = normalizer
        self.threshold = threshold

    def load_checkpoint(self, checkpoint_path: str) -> 'AnomalyDetector':
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Self for method chaining
        """
        self.model = keras.models.load_model(checkpoint_path)
        return self

    def set_threshold(self, method: str = 'percentile', **params) -> float:
        """
        Set anomaly threshold using specified method.

        Methods:
            - 'percentile': Use percentile of training errors (param: percentile, errors)
            - 'std': Use mean + k*std (params: mean, std, k)
            - 'manual': Manually set threshold (param: value)

        Args:
            method: Threshold calculation method
            **params: Method-specific parameters

        Returns:
            Calculated threshold
        """
        if method == 'percentile':
            errors = params.get('errors')
            percentile = params.get('percentile', 95)
            if errors is None:
                raise ValueError("'errors' parameter required for percentile method")
            self.threshold = np.percentile(errors, percentile)

        elif method == 'std':
            mean = params.get('mean')
            std = params.get('std')
            k = params.get('k', 3)
            if mean is None or std is None:
                raise ValueError("'mean' and 'std' parameters required for std method")
            self.threshold = mean + k * std

        elif method == 'manual':
            value = params.get('value')
            if value is None:
                raise ValueError("'value' parameter required for manual method")
            self.threshold = value

        else:
            raise ValueError(f"Unknown threshold method: {method}")

        print(f"Threshold set to: {self.threshold:.6f}")
        return self.threshold

    def compute_reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for data.

        Args:
            data: Input data (n_samples, seq_len, n_features)

        Returns:
            Reconstruction errors per sample (n_samples,)
        """
        # Predict
        reconstructed = self.model.predict(data, verbose=0)

        # Compute error (MSE per sample)
        errors = np.mean(np.square(data - reconstructed), axis=(1, 2))

        return errors

    def detect(self, data: np.ndarray, threshold: Optional[float] = None) -> AnomalyResult:
        """
        Detect anomalies in data.

        Args:
            data: Input data (n_samples, seq_len, n_features)
            threshold: Override threshold (optional)

        Returns:
            AnomalyResult object
        """
        if threshold is not None:
            self.threshold = threshold

        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first or provide threshold parameter.")

        # Compute reconstruction errors
        errors = self.compute_reconstruction_error(data)

        # Detect anomalies
        is_anomaly = errors > self.threshold

        # Compute anomaly scores (normalized by threshold)
        anomaly_scores = errors / self.threshold

        # Metadata
        metadata = {
            'n_samples': len(data),
            'n_anomalies': int(np.sum(is_anomaly)),
            'anomaly_rate': float(np.mean(is_anomaly)),
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors))
        }

        result = AnomalyResult(
            reconstruction_errors=errors,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            anomaly_scores=anomaly_scores,
            metadata=metadata
        )

        return result

    @classmethod
    def from_experiment(cls, experiment_dir: str, checkpoint: str = 'best_model') -> 'AnomalyDetector':
        """
        Load detector from experiment directory.

        Args:
            experiment_dir: Path to experiment directory
            checkpoint: Checkpoint to load ('best_model' or 'last_model')

        Returns:
            AnomalyDetector instance
        """
        experiment_dir = Path(experiment_dir)

        # Load model
        checkpoint_path = experiment_dir / 'checkpoints' / checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = keras.models.load_model(checkpoint_path)

        # Load normalizer
        from ..data.normalizer import Normalizer

        normalizer_path = experiment_dir / 'normalizer_stats.json'
        if normalizer_path.exists():
            normalizer = Normalizer()
            normalizer.load_statistics(str(normalizer_path))
        else:
            print("Warning: Normalizer statistics not found. Normalizer will be None.")
            normalizer = None

        # Load threshold if available
        results_path = experiment_dir / 'results.json'
        threshold = None
        if results_path.exists():
            import json
            with open(results_path, 'r') as f:
                results = json.load(f)
                threshold = results.get('threshold')

        detector = cls(model=model, normalizer=normalizer, threshold=threshold)

        return detector

    def save_results(self, result: AnomalyResult, filepath: str) -> None:
        """
        Save anomaly detection results.

        Args:
            result: AnomalyResult object
            filepath: Path to save results
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save as npz
        np.savez(
            filepath,
            reconstruction_errors=result.reconstruction_errors,
            is_anomaly=result.is_anomaly,
            threshold=result.threshold,
            anomaly_scores=result.anomaly_scores,
            metadata=result.metadata
        )

    @staticmethod
    def load_results(filepath: str) -> AnomalyResult:
        """
        Load anomaly detection results.

        Args:
            filepath: Path to results file

        Returns:
            AnomalyResult object
        """
        loaded = np.load(filepath, allow_pickle=True)

        return AnomalyResult(
            reconstruction_errors=loaded['reconstruction_errors'],
            is_anomaly=loaded['is_anomaly'],
            threshold=float(loaded['threshold']),
            anomaly_scores=loaded['anomaly_scores'],
            metadata=loaded['metadata'].item()
        )
