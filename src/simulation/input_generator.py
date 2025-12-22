"""Voltage input signal generation for DC motor simulation."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path
from scipy import signal as scipy_signal


class VoltageInputGenerator:
    """Generate concatenated voltage input signals for DC motor simulation."""

    def __init__(self, sampling_rate: int = 20000, save_format: str = "npy"):
        """
        Initialize VoltageInputGenerator.

        Args:
            sampling_rate: Sampling rate in Hz
            save_format: Save format ("npy" or "csv")
        """
        self.sampling_rate = sampling_rate
        self.save_format = save_format
        self.signals: List[Dict[str, Any]] = []
        self._generated_voltage: Optional[np.ndarray] = None
        self._time: Optional[np.ndarray] = None

    def add_signal(self, signal_type: str, duration: float, **params) -> 'VoltageInputGenerator':
        """
        Add a signal to the sequence.

        Args:
            signal_type: Type of signal ('step', 'ramp', 'sinusoidal', 'sawtooth', 'triangular', 'chirp')
            duration: Duration in seconds
            **params: Signal-specific parameters

        Returns:
            Self for method chaining
        """
        signal_config = {
            'type': signal_type,
            'duration_sec': duration,
            **params
        }
        self.signals.append(signal_config)
        return self

    def clear_signals(self) -> 'VoltageInputGenerator':
        """
        Clear all signals.

        Returns:
            Self for method chaining
        """
        self.signals = []
        self._generated_voltage = None
        self._time = None
        return self

    def generate(self) -> np.ndarray:
        """
        Generate concatenated voltage signal.

        Returns:
            Generated voltage array
        """
        if not self.signals:
            raise ValueError("No signals added. Use add_signal() first.")

        voltage_segments = []
        time_segments = []
        current_time = 0.0

        for signal_config in self.signals:
            signal_type = signal_config['type']
            duration = signal_config['duration_sec']
            n_samples = int(duration * self.sampling_rate)
            t = np.linspace(0, duration, n_samples, endpoint=False)

            if signal_type == 'step':
                v = self._generate_step(t, signal_config)
            elif signal_type == 'ramp':
                v = self._generate_ramp(t, signal_config)
            elif signal_type == 'sinusoidal':
                v = self._generate_sinusoidal(t, signal_config)
            elif signal_type == 'sawtooth':
                v = self._generate_sawtooth(t, signal_config)
            elif signal_type == 'triangular':
                v = self._generate_triangular(t, signal_config)
            elif signal_type == 'chirp':
                v = self._generate_chirp(t, signal_config)
            else:
                raise ValueError(f"Unknown signal type: {signal_type}")

            voltage_segments.append(v)
            time_segments.append(t + current_time)
            current_time += duration

        self._generated_voltage = np.concatenate(voltage_segments)
        self._time = np.concatenate(time_segments)

        return self._generated_voltage

    def _generate_step(self, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate step signal."""
        amplitude = config.get('amplitude', 12.0)
        return np.ones_like(t) * amplitude

    def _generate_ramp(self, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate ramp signal."""
        start_amplitude = config.get('start_amplitude', 0.0)
        end_amplitude = config.get('end_amplitude', 24.0)
        return np.linspace(start_amplitude, end_amplitude, len(t))

    def _generate_sinusoidal(self, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate sinusoidal signal."""
        amplitude = config.get('amplitude', 12.0)
        frequency = config.get('frequency_hz', 5.0)
        offset = config.get('offset', 0.0)
        phase = config.get('phase', 0.0)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    def _generate_sawtooth(self, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate sawtooth signal."""
        amplitude = config.get('amplitude', 24.0)
        frequency = config.get('frequency_hz', 2.0)
        offset = config.get('offset', 0.0)
        return amplitude * scipy_signal.sawtooth(2 * np.pi * frequency * t) + offset

    def _generate_triangular(self, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate triangular signal."""
        amplitude = config.get('amplitude', 24.0)
        frequency = config.get('frequency_hz', 2.0)
        offset = config.get('offset', 0.0)
        return amplitude * scipy_signal.sawtooth(2 * np.pi * frequency * t, width=0.5) + offset

    def _generate_chirp(self, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate chirp signal (frequency sweep)."""
        amplitude = config.get('amplitude', 12.0)
        start_freq = config.get('start_frequency_hz', 1.0)
        end_freq = config.get('end_frequency_hz', 20.0)
        offset = config.get('offset', 0.0)
        method = config.get('method', 'linear')
        return amplitude * scipy_signal.chirp(t, start_freq, t[-1], end_freq, method=method) + offset

    def save(self, filepath: str) -> None:
        """
        Save generated voltage signal to file.

        Args:
            filepath: Path to save file
        """
        if self._generated_voltage is None:
            raise ValueError("No signal generated. Call generate() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.save_format == 'npy':
            # Save as numpy binary
            data = {
                'voltage': self._generated_voltage,
                'time': self._time,
                'sampling_rate': self.sampling_rate,
                'signals': self.signals
            }
            np.save(filepath, data, allow_pickle=True)

        elif self.save_format == 'csv':
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame({
                'time': self._time,
                'voltage': self._generated_voltage
            })
            df.to_csv(filepath, index=False)

        else:
            raise ValueError(f"Unknown save format: {self.save_format}")

    def plot(self, figsize: tuple = (12, 6), title: str = "Voltage Input Signal") -> plt.Figure:
        """
        Plot generated voltage signal.

        Args:
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if self._generated_voltage is None:
            raise ValueError("No signal generated. Call generate() first.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self._time, self._generated_voltage, linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add vertical lines to separate signal segments
        current_time = 0.0
        for i, signal_config in enumerate(self.signals):
            if i > 0:
                ax.axvline(current_time, color='r', linestyle='--', alpha=0.5)
            current_time += signal_config['duration_sec']

        plt.tight_layout()
        return fig

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VoltageInputGenerator':
        """
        Create VoltageInputGenerator from configuration.

        Args:
            config: Configuration dictionary with 'input_generator' section

        Returns:
            VoltageInputGenerator instance
        """
        input_config = config.get('input_generator', {})
        sampling_rate = input_config.get('sampling_rate_hz', 20000)
        save_format = input_config.get('save_format', 'npy')

        generator = cls(sampling_rate=sampling_rate, save_format=save_format)

        # Add signals from config
        signals = input_config.get('signals', [])
        for signal_config in signals:
            signal_type = signal_config['type']
            duration = signal_config['duration_sec']

            # Extract signal-specific parameters
            params = {k: v for k, v in signal_config.items()
                     if k not in ['type', 'duration_sec']}

            generator.add_signal(signal_type, duration, **params)

        return generator
