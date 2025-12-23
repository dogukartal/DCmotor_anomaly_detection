"""Physics-informed loss function for DC motor dynamics."""

import tensorflow as tf
from typing import Dict, Any, Optional
import numpy as np


class PhysicsInformedLoss:
    """
    Custom loss combining reconstruction and physics constraints.

    Physics Constraints (DC Motor):
        Electrical: V ≈ R*i + L*(di/dt) + Ke*ω
        Mechanical: J*(dω/dt) ≈ Kt*i - B*ω
    """

    def __init__(self,
                 motor_params: Dict[str, float],
                 physics_weight: float = 0.1,
                 reconstruction_loss: str = 'mse',
                 start_epoch: int = 0,
                 enabled: bool = True,
                 normalizer = None,
                 dt: float = 0.002):
        """
        Initialize PhysicsInformedLoss.

        Args:
            motor_params: Motor parameters (R, L, Kt, Ke, J, B)
            physics_weight: Weight for physics loss component
            reconstruction_loss: Type of reconstruction loss ('mse' or 'mae')
            start_epoch: Epoch to start applying physics loss
            enabled: Whether physics loss is enabled
            normalizer: Normalizer for inverse transform
            dt: Time step for derivative calculation (1/sampling_rate)
        """
        self.motor_params = motor_params
        self.physics_weight = physics_weight
        self.reconstruction_loss_type = reconstruction_loss
        self.start_epoch = start_epoch
        self.enabled = enabled
        self.normalizer = normalizer
        self.dt = dt

        self.current_epoch = 0

        # Extract motor parameters
        self.R = float(motor_params['R'])
        self.L = float(motor_params['L'])
        self.Kt = float(motor_params['Kt'])
        self.Ke = float(motor_params['Ke'])
        self.J = float(motor_params['J'])
        self.B = float(motor_params['B'])

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute combined loss.

        Args:
            y_true: True values (batch_size, seq_len, n_features)
            y_pred: Predicted values (batch_size, seq_len, n_features)

        Returns:
            Combined loss tensor
        """
        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(y_true, y_pred)

        # Physics loss (only if enabled and past start epoch)
        if self.enabled and self.current_epoch >= self.start_epoch:
            physics_loss = self.compute_physics_loss(y_pred)
            total_loss = recon_loss + self.physics_weight * physics_loss
        else:
            total_loss = recon_loss

        return total_loss

    def compute_reconstruction_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute reconstruction loss.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Reconstruction loss
        """
        if self.reconstruction_loss_type == 'mse':
            return tf.reduce_mean(tf.square(y_true - y_pred))
        elif self.reconstruction_loss_type == 'mae':
            return tf.reduce_mean(tf.abs(y_true - y_pred))
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss_type}")

    def compute_physics_loss(self, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute physics constraint loss.

        Args:
            y_pred: Predicted values (batch_size, seq_len, n_features)
                   Assumes features are [current, angular_velocity, voltage, ...]

        Returns:
            Physics loss
        """
        # Denormalize predictions to physical units before computing physics residuals
        if self.normalizer is not None:
            # Convert tensor to numpy for denormalization
            y_pred_np = y_pred.numpy()
            y_pred_denorm_np = self.normalizer.inverse_transform(y_pred_np)
            # Convert back to tensor
            y_pred_physical = tf.convert_to_tensor(y_pred_denorm_np, dtype=tf.float32)
        else:
            # If no normalizer, assume predictions are already in physical units
            y_pred_physical = y_pred

        # Extract variables (assuming order: current, angular_velocity, voltage, ...)
        # Shape: (batch_size, seq_len)
        i_pred = y_pred_physical[:, :, 0]  # Current (A)
        omega_pred = y_pred_physical[:, :, 1]  # Angular velocity (rad/s)
        V_pred = y_pred_physical[:, :, 2]  # Voltage (V)

        # Compute derivatives using finite differences in physical units
        # di/dt ≈ (i[t+1] - i[t]) / dt (A/s)
        di_dt = (i_pred[:, 1:] - i_pred[:, :-1]) / self.dt
        domega_dt = (omega_pred[:, 1:] - omega_pred[:, :-1]) / self.dt  # (rad/s^2)

        # Match shapes (use values at t for physics equations)
        i_t = i_pred[:, :-1]
        omega_t = omega_pred[:, :-1]
        V_t = V_pred[:, :-1]

        # Electrical equation: V = R*i + L*(di/dt) + Ke*ω
        # All terms in Volts
        electrical_residual = V_t - (self.R * i_t + self.L * di_dt + self.Ke * omega_t)

        # Mechanical equation: J*(dω/dt) = Kt*i - B*ω
        # All terms in N⋅m (torque)
        mechanical_residual = self.J * domega_dt - (self.Kt * i_t - self.B * omega_t)

        # Compute losses
        electrical_loss = tf.reduce_mean(tf.square(electrical_residual))
        mechanical_loss = tf.reduce_mean(tf.square(mechanical_residual))

        # Combined physics loss
        physics_loss = electrical_loss + mechanical_loss

        return physics_loss

    def compute_loss_components(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Compute loss components separately for tracking.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with 'reconstruction', 'electrical', 'mechanical', 'physics' losses
        """
        recon_loss = self.compute_reconstruction_loss(y_true, y_pred)

        components = {
            'reconstruction': recon_loss,
            'electrical': tf.constant(0.0),
            'mechanical': tf.constant(0.0),
            'physics': tf.constant(0.0)
        }

        # Only compute physics components if enabled and past start epoch
        if self.enabled and self.current_epoch >= self.start_epoch:
            # Denormalize predictions
            if self.normalizer is not None:
                y_pred_np = y_pred.numpy()
                y_pred_denorm_np = self.normalizer.inverse_transform(y_pred_np)
                y_pred_physical = tf.convert_to_tensor(y_pred_denorm_np, dtype=tf.float32)
            else:
                y_pred_physical = y_pred

            # Extract variables
            i_pred = y_pred_physical[:, :, 0]
            omega_pred = y_pred_physical[:, :, 1]
            V_pred = y_pred_physical[:, :, 2]

            # Compute derivatives
            di_dt = (i_pred[:, 1:] - i_pred[:, :-1]) / self.dt
            domega_dt = (omega_pred[:, 1:] - omega_pred[:, :-1]) / self.dt

            # Match shapes
            i_t = i_pred[:, :-1]
            omega_t = omega_pred[:, :-1]
            V_t = V_pred[:, :-1]

            # Compute residuals
            electrical_residual = V_t - (self.R * i_t + self.L * di_dt + self.Ke * omega_t)
            mechanical_residual = self.J * domega_dt - (self.Kt * i_t - self.B * omega_t)

            # Compute component losses
            electrical_loss = tf.reduce_mean(tf.square(electrical_residual))
            mechanical_loss = tf.reduce_mean(tf.square(mechanical_residual))
            physics_loss = electrical_loss + mechanical_loss

            components['electrical'] = electrical_loss
            components['mechanical'] = mechanical_loss
            components['physics'] = physics_loss

        return components

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for physics loss scheduling.

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch

    @classmethod
    def from_config(cls,
                   config: Dict[str, Any],
                   normalizer = None,
                   sampling_rate: int = 500) -> 'PhysicsInformedLoss':
        """
        Create PhysicsInformedLoss from configuration.

        Args:
            config: Configuration dictionary
            normalizer: Normalizer object for denormalization
            sampling_rate: Sampling rate for dt calculation

        Returns:
            PhysicsInformedLoss instance
        """
        physics_config = config.get('physics_loss', {})
        training_config = config.get('training', {})

        enabled = physics_config.get('enabled', True)
        weight = physics_config.get('weight', 0.1)
        start_epoch = physics_config.get('start_epoch', 10)

        # Get motor parameters
        motor_params_source = physics_config.get('motor_params', 'from_simulation')
        if motor_params_source == 'from_simulation':
            motor_params = config['simulation']['motor_params']
        else:
            motor_params = motor_params_source

        reconstruction_loss = training_config.get('loss', 'mse')

        # Calculate dt
        dt = 1.0 / sampling_rate

        return cls(
            motor_params=motor_params,
            physics_weight=weight,
            reconstruction_loss=reconstruction_loss,
            start_epoch=start_epoch,
            enabled=enabled,
            normalizer=normalizer,
            dt=dt
        )
