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
        Plot training history with multiple subplots for different metrics.

        Args:
            history: Training history dictionary
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Determine number of subplots needed
        has_physics = 'val_physics_loss' in history
        has_lr = 'lr' in history

        # Calculate number of rows needed
        n_rows = 1  # Main loss plot
        if has_physics:
            n_rows += 1  # Physics loss components
        if has_lr:
            n_rows += 1  # Learning rate

        fig, axes = plt.subplots(n_rows, 1, figsize=(self.figsize[0], 4 * n_rows))

        # Ensure axes is always a list
        if n_rows == 1:
            axes = [axes]

        current_ax_idx = 0

        # Plot 1: Main Loss
        ax = axes[current_ax_idx]
        current_ax_idx += 1

        if 'loss' in history:
            ax.plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)

        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Physics Loss Components (if available)
        if has_physics:
            ax = axes[current_ax_idx]
            current_ax_idx += 1

            if 'val_physics_loss' in history:
                ax.plot(history['val_physics_loss'], label='Total Physics Loss', linewidth=2, color='purple')
            if 'val_electrical_loss' in history:
                ax.plot(history['val_electrical_loss'], label='Electrical Loss', linewidth=2, color='blue', alpha=0.7)
            if 'val_mechanical_loss' in history:
                ax.plot(history['val_mechanical_loss'], label='Mechanical Loss', linewidth=2, color='green', alpha=0.7)

            ax.set_ylabel('Physics Loss')
            ax.set_title('Physics Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Learning Rate (if available)
        if has_lr:
            ax = axes[current_ax_idx]
            current_ax_idx += 1

            ax.plot(history['lr'], label='Learning Rate', linewidth=2, color='red')
            ax.set_ylabel('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')  # Log scale for better visibility
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Add xlabel to the last plot
            axes[-1].set_xlabel('Epoch')

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

    def plot_multi_sample_reconstruction(self,
                                        original: np.ndarray,
                                        reconstructed: np.ndarray,
                                        sample_indices: np.ndarray,
                                        feature_names: Optional[List[str]] = None,
                                        title: str = "Multi-Sample Reconstruction") -> plt.Figure:
        """
        Plot multiple samples in a grid layout (rows=features, cols=samples).

        Args:
            original: Original data (num_samples, seq_len, n_features)
            reconstructed: Reconstructed data (num_samples, seq_len, n_features)
            sample_indices: Indices of samples being plotted
            feature_names: List of feature names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        num_samples = original.shape[0]
        n_features = original.shape[2]

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]

        # Create grid: rows=features, cols=samples
        fig, axes = plt.subplots(n_features, num_samples,
                                figsize=(4 * num_samples, 3 * n_features))

        # Handle single feature or single sample cases
        if n_features == 1 and num_samples == 1:
            axes = np.array([[axes]])
        elif n_features == 1:
            axes = axes.reshape(1, -1)
        elif num_samples == 1:
            axes = axes.reshape(-1, 1)

        # Plot each feature-sample combination
        for feature_idx in range(n_features):
            for sample_idx in range(num_samples):
                ax = axes[feature_idx, sample_idx]

                # Plot original and reconstructed
                ax.plot(original[sample_idx, :, feature_idx], 'b-',
                       label='Original', linewidth=1.5, alpha=0.7)
                ax.plot(reconstructed[sample_idx, :, feature_idx], 'r--',
                       label='Reconstructed', linewidth=1.5, alpha=0.7)

                # Add labels
                if sample_idx == 0:
                    ax.set_ylabel(feature_names[feature_idx], fontsize=10)
                if feature_idx == 0:
                    ax.set_title(f'Sample {sample_indices[sample_idx]}', fontsize=10)
                if feature_idx == n_features - 1:
                    ax.set_xlabel('Time Step', fontsize=9)

                # Add legend only to the first plot
                if feature_idx == 0 and sample_idx == 0:
                    ax.legend(fontsize=8, loc='upper right')

                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

        fig.suptitle(title, fontsize=12, y=0.995)
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
