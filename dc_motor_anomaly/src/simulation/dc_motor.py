"""DC Motor simulation using electrical and mechanical equations."""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import pandas as pd


@dataclass
class MotorParams:
    """DC Motor parameters."""
    R: float  # Armature resistance (Ohm)
    L: float  # Armature inductance (H)
    Kt: float  # Torque constant (Nm/A)
    Ke: float  # Back-EMF constant (V·s/rad)
    J: float  # Moment of inertia (kg·m²)
    B: float  # Viscous friction coefficient (Nm·s/rad)


@dataclass
class SimulationResult:
    """Container for simulation results."""
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    angular_velocity: np.ndarray
    metadata: Dict[str, Any]


class DCMotorSimulator:
    """
    Simulate DC motor dynamics using electrical and mechanical equations.

    Equations:
        Electrical: V = R*i + L*(di/dt) + Ke*ω
        Mechanical: J*(dω/dt) = Kt*i - B*ω - T_load
    """

    def __init__(self,
                 motor_params: MotorParams,
                 sampling_rate: int = 20000,
                 initial_conditions: Optional[Dict[str, float]] = None):
        """
        Initialize DCMotorSimulator.

        Args:
            motor_params: Motor parameters
            sampling_rate: Sampling rate in Hz
            initial_conditions: Initial current and angular_velocity
        """
        self.motor_params = motor_params
        self.sampling_rate = sampling_rate

        if initial_conditions is None:
            initial_conditions = {'current': 0.0, 'angular_velocity': 0.0}

        self.initial_conditions = initial_conditions

    def simulate(self,
                 voltage_input: np.ndarray,
                 external_torque: float = 0.0) -> SimulationResult:
        """
        Simulate DC motor response to voltage input.

        Args:
            voltage_input: Input voltage array
            external_torque: External load torque (Nm)

        Returns:
            SimulationResult containing time, voltage, current, angular_velocity
        """
        # Time array
        n_samples = len(voltage_input)
        dt = 1.0 / self.sampling_rate
        time = np.arange(n_samples) * dt

        # Initial state: [current, angular_velocity]
        y0 = [
            self.initial_conditions['current'],
            self.initial_conditions['angular_velocity']
        ]

        # Integrate using odeint
        # We need to solve for each time step with the corresponding voltage
        current = np.zeros(n_samples)
        angular_velocity = np.zeros(n_samples)

        current[0] = y0[0]
        angular_velocity[0] = y0[1]

        for i in range(1, n_samples):
            # Time span for this step
            t_span = [time[i-1], time[i]]

            # Average voltage over this interval
            v = voltage_input[i-1]

            # Solve ODE for this time step
            solution = odeint(
                self._derivatives,
                [current[i-1], angular_velocity[i-1]],
                t_span,
                args=(v, external_torque)
            )

            current[i] = solution[-1, 0]
            angular_velocity[i] = solution[-1, 1]

        # Create result
        result = SimulationResult(
            time=time,
            voltage=voltage_input,
            current=current,
            angular_velocity=angular_velocity,
            metadata={
                'motor_params': {
                    'R': self.motor_params.R,
                    'L': self.motor_params.L,
                    'Kt': self.motor_params.Kt,
                    'Ke': self.motor_params.Ke,
                    'J': self.motor_params.J,
                    'B': self.motor_params.B
                },
                'sampling_rate': self.sampling_rate,
                'external_torque': external_torque,
                'initial_conditions': self.initial_conditions
            }
        )

        return result

    def _derivatives(self, y: np.ndarray, t: float, V: float, T_load: float) -> np.ndarray:
        """
        Compute derivatives for ODE solver.

        Args:
            y: State vector [current, angular_velocity]
            t: Time (not used, but required by odeint)
            V: Input voltage
            T_load: External load torque

        Returns:
            Derivatives [di/dt, dω/dt]
        """
        i, omega = y

        # Electrical equation: di/dt = (V - R*i - Ke*ω) / L
        di_dt = (V - self.motor_params.R * i - self.motor_params.Ke * omega) / self.motor_params.L

        # Mechanical equation: dω/dt = (Kt*i - B*ω - T_load) / J
        domega_dt = (self.motor_params.Kt * i - self.motor_params.B * omega - T_load) / self.motor_params.J

        return [di_dt, domega_dt]

    def save_results(self, result: SimulationResult, filepath: str, format: str = 'npy') -> None:
        """
        Save simulation results to file.

        Args:
            result: SimulationResult object
            filepath: Path to save file
            format: Save format ('npy' or 'csv')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npy':
            # Save as numpy binary
            data = {
                'time': result.time,
                'voltage': result.voltage,
                'current': result.current,
                'angular_velocity': result.angular_velocity,
                'metadata': result.metadata
            }
            np.save(filepath, data, allow_pickle=True)

        elif format == 'csv':
            # Save as CSV
            df = pd.DataFrame({
                'time': result.time,
                'voltage': result.voltage,
                'current': result.current,
                'angular_velocity': result.angular_velocity
            })
            df.to_csv(filepath, index=False)

            # Save metadata separately
            metadata_path = filepath.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(result.metadata, f, indent=2)

        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DCMotorSimulator':
        """
        Create DCMotorSimulator from configuration.

        Args:
            config: Configuration dictionary with 'simulation' section

        Returns:
            DCMotorSimulator instance
        """
        sim_config = config.get('simulation', {})

        # Extract motor parameters
        motor_params_dict = sim_config.get('motor_params', {})
        motor_params = MotorParams(
            R=motor_params_dict.get('R', 1.0),
            L=motor_params_dict.get('L', 0.5),
            Kt=motor_params_dict.get('Kt', 0.01),
            Ke=motor_params_dict.get('Ke', 0.01),
            J=motor_params_dict.get('J', 0.01),
            B=motor_params_dict.get('B', 0.1)
        )

        sampling_rate = sim_config.get('sampling_rate_hz', 20000)
        initial_conditions = sim_config.get('initial_conditions', None)

        return cls(
            motor_params=motor_params,
            sampling_rate=sampling_rate,
            initial_conditions=initial_conditions
        )

    @staticmethod
    def load_results(filepath: str) -> SimulationResult:
        """
        Load simulation results from file.

        Args:
            filepath: Path to results file

        Returns:
            SimulationResult object
        """
        filepath = Path(filepath)

        if filepath.suffix == '.npy':
            data = np.load(filepath, allow_pickle=True).item()
            return SimulationResult(
                time=data['time'],
                voltage=data['voltage'],
                current=data['current'],
                angular_velocity=data['angular_velocity'],
                metadata=data['metadata']
            )

        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)

            # Load metadata
            metadata_path = filepath.with_suffix('.json')
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return SimulationResult(
                time=df['time'].values,
                voltage=df['voltage'].values,
                current=df['current'].values,
                angular_velocity=df['angular_velocity'].values,
                metadata=metadata
            )

        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")
