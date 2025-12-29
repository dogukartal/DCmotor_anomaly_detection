"""Custom callbacks for training."""

import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
from typing import Dict, Any, Optional


class PhysicsLossScheduler(keras.callbacks.Callback):
    """Callback to enable physics loss after a specified epoch."""

    def __init__(self, physics_loss, start_epoch: int = 10):
        """
        Initialize PhysicsLossScheduler.

        Args:
            physics_loss: PhysicsInformedLoss instance
            start_epoch: Epoch to start applying physics loss
        """
        super().__init__()
        self.physics_loss = physics_loss
        self.start_epoch = start_epoch

    def on_epoch_begin(self, epoch, logs=None):
        """Update physics loss epoch counter."""
        self.physics_loss.set_epoch(epoch)

        if epoch == self.start_epoch:
            print(f"\nPhysics loss enabled at epoch {epoch}")


class TrainingLogger(keras.callbacks.Callback):
    """Callback to log training metrics to JSON file."""

    def __init__(self, log_filepath: str):
        """
        Initialize TrainingLogger.

        Args:
            log_filepath: Path to log file
        """
        super().__init__()
        self.log_filepath = Path(log_filepath)
        self.log_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        logs = logs or {}
        epoch_log = {'epoch': epoch}
        epoch_log.update(logs)
        self.history.append(epoch_log)

        # Save to file
        with open(self.log_filepath, 'w') as f:
            json.dump(str(self.history), f, indent=2)


class PlotCallback(keras.callbacks.Callback):
    """Callback to generate plots at specified intervals."""

    def __init__(self,
                 plot_interval: int,
                 plot_dir: str,
                 val_data: tf.data.Dataset,
                 plotter,
                 feature_names: list,
                 num_samples: int = 1):
        """
        Initialize PlotCallback.

        Args:
            plot_interval: Interval (in epochs) to generate plots
            plot_dir: Directory to save plots
            val_data: Validation dataset for generating plots
            plotter: Plotter instance
            feature_names: List of feature names
            num_samples: Number of samples to plot (randomly selected and reused)
        """
        super().__init__()
        self.plot_interval = plot_interval
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.val_data = val_data
        self.plotter = plotter
        self.feature_names = feature_names
        self.num_samples = num_samples
        self.sample_indices = None
        self.sample_data = None

    def on_train_begin(self, logs=None):
        """Initialize sample data at the start of training."""
        import numpy as np

        # Get a batch from validation data
        for x_val, y_val in self.val_data.take(1):
            batch_size = x_val.shape[0]

            # Randomly select sample indices
            self.sample_indices = np.random.choice(
                batch_size,
                size=min(self.num_samples, batch_size),
                replace=False
            )

            # Store the selected samples
            self.sample_data = x_val.numpy()[self.sample_indices]
            print(f"\nSelected {len(self.sample_indices)} random samples for reconstruction plots (indices: {self.sample_indices.tolist()})")
            break

    def on_epoch_end(self, epoch, logs=None):
        """Generate plots at specified intervals."""
        if (epoch + 1) % self.plot_interval == 0:
            # Use stored sample data
            if self.sample_data is not None:
                y_pred = self.model.predict(self.sample_data, verbose=0)

                # Plot all samples in a single grid figure
                fig = self.plotter.plot_multi_sample_reconstruction(
                    self.sample_data,
                    y_pred,
                    sample_indices=self.sample_indices,
                    feature_names=self.feature_names,
                    title=f"Reconstruction at Epoch {epoch+1}"
                )
                self.plotter.save_figure(
                    fig,
                    self.plot_dir / f'reconstruction_epoch_{epoch+1:03d}.png'
                )


class LearningRateLogger(keras.callbacks.Callback):
    """Callback to log learning rate."""

    def on_epoch_end(self, epoch, logs=None):
        """Log current learning rate."""
        logs = logs or {}
        if hasattr(self.model.optimizer, 'lr'):
            lr = float(keras.backend.get_value(self.model.optimizer.lr))
            logs['lr'] = lr


class PhysicsLossTracker(keras.callbacks.Callback):
    """Callback to track physics loss components during training."""

    def __init__(self, physics_loss, val_data: tf.data.Dataset):
        """
        Initialize PhysicsLossTracker.

        Args:
            physics_loss: PhysicsInformedLoss instance
            val_data: Validation dataset for computing physics loss
        """
        super().__init__()
        self.physics_loss = physics_loss
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        """Track physics loss components at end of epoch."""
        if logs is None:
            logs = {}

        # Only compute if physics loss is enabled and past start epoch
        if self.physics_loss.enabled and epoch >= self.physics_loss.start_epoch:
            # Get a batch from validation data
            for x_val, y_val in self.val_data.take(1):
                y_pred = self.model.predict(x_val, verbose=0)
                y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
                y_val_tf = tf.convert_to_tensor(y_val, dtype=tf.float32)

                # Compute loss components
                components = self.physics_loss.compute_loss_components(y_val_tf, y_pred_tf)

                # Log components
                logs['val_physics_loss'] = float(components['physics'].numpy())
                logs['val_electrical_loss'] = float(components['electrical'].numpy())
                logs['val_mechanical_loss'] = float(components['mechanical'].numpy())
                logs['val_reconstruction_loss'] = float(components['reconstruction'].numpy())

                break


def create_callbacks(config: Dict[str, Any],
                    experiment_paths,
                    physics_loss = None,
                    val_data: Optional[tf.data.Dataset] = None,
                    plotter = None,
                    feature_names: Optional[list] = None) -> list:
    """
    Create list of callbacks from configuration.

    Args:
        config: Configuration dictionary
        experiment_paths: ExperimentPaths object
        physics_loss: PhysicsInformedLoss instance (optional)
        val_data: Validation dataset (optional, for PlotCallback)
        plotter: Plotter instance (optional, for PlotCallback)
        feature_names: List of feature names (optional, for PlotCallback)

    Returns:
        List of callbacks
    """
    callbacks = []

    training_config = config.get('training', {})

    # Early stopping
    early_stopping_config = training_config.get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 1e-5),
            restore_best_weights=early_stopping_config.get('restore_best_weights', True),
            verbose=1
        )
        callbacks.append(early_stopping)

    # Learning rate scheduler
    lr_scheduler_config = training_config.get('lr_scheduler', {})
    if lr_scheduler_config.get('enabled', True):
        scheduler_type = lr_scheduler_config.get('type', 'reduce_on_plateau')

        if scheduler_type == 'reduce_on_plateau':
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=lr_scheduler_config.get('monitor', 'val_loss'),
                factor=lr_scheduler_config.get('factor', 0.5),
                patience=lr_scheduler_config.get('patience', 5),
                min_lr=lr_scheduler_config.get('min_lr', 1e-6),
                verbose=1
            )
            callbacks.append(lr_scheduler)

    # Checkpoints
    checkpoint_config = training_config.get('checkpoints', {})
    if checkpoint_config.get('save_best', True):
        best_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(Path(experiment_paths.checkpoints) / 'best_model.keras'),
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            save_best_only=True,
            verbose=1
        )
        callbacks.append(best_checkpoint)

    if checkpoint_config.get('save_last', True):
        last_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(Path(experiment_paths.checkpoints) / 'last_model.keras'),
            save_best_only=False,
            verbose=0
        )
        callbacks.append(last_checkpoint)

    # Physics loss scheduler
    if physics_loss is not None:
        physics_config = config.get('physics_loss', {})
        if physics_config.get('enabled', True):
            physics_scheduler = PhysicsLossScheduler(
                physics_loss=physics_loss,
                start_epoch=physics_config.get('start_epoch', 10)
            )
            callbacks.append(physics_scheduler)

    # Training logger
    log_filepath = Path(experiment_paths.logs) / 'training_log.json'
    training_logger = TrainingLogger(log_filepath=str(log_filepath))
    callbacks.append(training_logger)

    # Learning rate logger
    lr_logger = LearningRateLogger()
    callbacks.append(lr_logger)

    # Physics loss tracker
    if physics_loss is not None and val_data is not None:
        physics_config = config.get('physics_loss', {})
        if physics_config.get('enabled', True):
            physics_tracker = PhysicsLossTracker(
                physics_loss=physics_loss,
                val_data=val_data
            )
            callbacks.append(physics_tracker)

    # Plot callback (optional)
    if val_data is not None and plotter is not None and feature_names is not None:
        plotting_config = config.get('plotting', {})
        plot_interval = plotting_config.get('plot_interval', 10)
        num_samples = plotting_config.get('num_samples', 1)
        plot_callback = PlotCallback(
            plot_interval=plot_interval,
            plot_dir=experiment_paths.plots,
            val_data=val_data,
            plotter=plotter,
            feature_names=feature_names,
            num_samples=num_samples
        )
        callbacks.append(plot_callback)

    return callbacks
