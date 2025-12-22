"""Visualization utilities for plotting simulation, training, and anomaly results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class Plotter:
    """Generate all visualizations for the project."""

    def __init__(self,
                 save_format: str = 'png',
                 dpi: int = 150,
                 figsize: tuple = (12, 6)):
        """
        Initialize Plotter.

        Args:
            save_format: Format for saving figures ('png', 'pdf', 'svg')
            dpi: DPI for saved figures
            figsize: Default figure size
        """
        self.save_format = save_format
        self.dpi = dpi
        self.figsize = figsize

    def plot_input_voltage(self,
                          voltage: np.ndarray,
                          time: np.ndarray,
                          title: str = "Input Voltage Signal") -> plt.Figure:
        """
        Plot input voltage signal.

        Args:
            voltage: Voltage array
            time: Time array
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(time, voltage, linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_simulation(self,
                       time: np.ndarray,
                       voltage: np.ndarray,
                       current: np.ndarray,
                       angular_velocity: np.ndarray,
                       title: str = "DC Motor Simulation Results") -> plt.Figure:
        """
        Plot simulation results (voltage, current, angular velocity).

        Args:
            time: Time array
            voltage: Voltage array
            current: Current array
            angular_velocity: Angular velocity array
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))

        # Voltage
        axes[0].plot(time, voltage, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Voltage (V)')
        axes[0].set_title(title)
        axes[0].grid(True, alpha=0.3)

        # Current
        axes[1].plot(time, current, 'r-', linewidth=1.5)
        axes[1].set_ylabel('Current (A)')
        axes[1].grid(True, alpha=0.3)

        # Angular velocity
        axes[2].plot(time, angular_velocity, 'g-', linewidth=1.5)
        axes[2].set_ylabel('Angular Velocity (rad/s)')
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_features(self,
                     data: np.ndarray,
                     feature_names: List[str],
                     title: str = "Processed Features") -> plt.Figure:
        """
        Plot processed features.

        Args:
            data: Data array (n_samples, n_features)
            feature_names: List of feature names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], 4 * n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
            ax.plot(data[:, i], linewidth=1)
            ax.set_ylabel(feature_name)
            ax.set_xlabel('Sample')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(title)
        plt.tight_layout()
        return fig

    def plot_training_history(self,
                             history: Dict[str, List[float]],
                             title: str = "Training History") -> plt.Figure:
        """
        Plot training history (loss curves).

        Args:
            history: Training history dictionary
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot loss
        if 'loss' in history:
            ax.plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_reconstruction(self,
                           original: np.ndarray,
                           reconstructed: np.ndarray,
                           idx: int = 0,
                           feature_names: Optional[List[str]] = None,
                           title: str = "Reconstruction Comparison") -> plt.Figure:
        """
        Plot original vs reconstructed sequence.

        Args:
            original: Original data (batch_size, seq_len, n_features)
            reconstructed: Reconstructed data (batch_size, seq_len, n_features)
            idx: Index of sample to plot
            feature_names: List of feature names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        n_features = original.shape[2]
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]

        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], 4 * n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
            ax.plot(original[idx, :, i], 'b-', label='Original', linewidth=2, alpha=0.7)
            ax.plot(reconstructed[idx, :, i], 'r--', label='Reconstructed', linewidth=2, alpha=0.7)
            ax.set_ylabel(feature_name)
            ax.set_xlabel('Time Step')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"{title} (Sample {idx})")
        plt.tight_layout()
        return fig

    def plot_reconstruction_error(self,
                                  errors: np.ndarray,
                                  title: str = "Reconstruction Error Over Time") -> plt.Figure:
        """
        Plot reconstruction error over samples.

        Args:
            errors: Error array (n_samples,)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(errors, linewidth=1.5)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_error_distribution(self,
                               errors: np.ndarray,
                               threshold: Optional[float] = None,
                               title: str = "Reconstruction Error Distribution") -> plt.Figure:
        """
        Plot histogram of reconstruction errors.

        Args:
            errors: Error array (n_samples,)
            threshold: Optional threshold line to plot
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(title)

        if threshold is not None:
            ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
            ax.legend()

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

    def plot_threshold_analysis(self,
                               errors: np.ndarray,
                               thresholds: np.ndarray,
                               title: str = "Threshold Analysis") -> plt.Figure:
        """
        Plot threshold analysis (e.g., percentiles).

        Args:
            errors: Error array (n_samples,)
            thresholds: Array of thresholds to visualize
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot error distribution
        ax.hist(errors, bins=50, alpha=0.5, edgecolor='black', label='Errors')

        # Plot threshold lines
        for i, threshold in enumerate(thresholds):
            percentile = np.sum(errors <= threshold) / len(errors) * 100
            ax.axvline(threshold, linestyle='--', linewidth=2,
                      label=f'P{percentile:.1f}: {threshold:.4f}')

        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, filepath: str) -> None:
        """
        Save figure to file.

        Args:
            fig: Matplotlib figure
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add extension if not present
        if not filepath.suffix:
            filepath = filepath.with_suffix(f'.{self.save_format}')

        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Plotter':
        """
        Create Plotter from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Plotter instance
        """
        plot_config = config.get('plotting', {})

        save_format = plot_config.get('save_format', 'png')
        dpi = plot_config.get('dpi', 150)
        figsize = tuple(plot_config.get('figsize', [12, 6]))

        return cls(save_format=save_format, dpi=dpi, figsize=figsize)
