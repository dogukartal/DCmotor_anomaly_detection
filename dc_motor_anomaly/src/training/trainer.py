"""Training manager for LSTM Autoencoder."""

import tensorflow as tf
from tensorflow import keras
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..training.callbacks import create_callbacks


class Trainer:
    """Manage training loop with custom callbacks."""

    def __init__(self,
                 model,
                 config: Dict[str, Any],
                 experiment_paths,
                 physics_loss = None,
                 normalizer = None):
        """
        Initialize Trainer.

        Args:
            model: LSTMAutoencoder instance
            config: Training configuration
            experiment_paths: ExperimentPaths object
            physics_loss: PhysicsInformedLoss instance (optional)
            normalizer: Normalizer instance (optional)
        """
        self.model = model
        self.config = config
        self.experiment_paths = experiment_paths
        self.physics_loss = physics_loss
        self.normalizer = normalizer
        self.history = None
        self.callbacks_list = []

    def setup_callbacks(self,
                       val_data: Optional[tf.data.Dataset] = None,
                       plotter = None,
                       feature_names: Optional[List[str]] = None) -> List[keras.callbacks.Callback]:
        """
        Setup training callbacks.

        Args:
            val_data: Validation dataset (for PlotCallback)
            plotter: Plotter instance (for PlotCallback)
            feature_names: List of feature names (for PlotCallback)

        Returns:
            List of callbacks
        """
        self.callbacks_list = create_callbacks(
            config=self.config,
            experiment_paths=self.experiment_paths,
            physics_loss=self.physics_loss,
            val_data=val_data,
            plotter=plotter,
            feature_names=feature_names
        )

        return self.callbacks_list

    def train(self,
             train_ds: tf.data.Dataset,
             val_ds: tf.data.Dataset,
             plotter = None,
             feature_names: Optional[List[str]] = None) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            plotter: Plotter instance (optional)
            feature_names: List of feature names (optional)

        Returns:
            Training history
        """
        training_config = self.config.get('training', {})

        # Setup callbacks
        if not self.callbacks_list:
            self.setup_callbacks(val_data=val_ds, plotter=plotter, feature_names=feature_names)

        # Train
        print("Starting training...")
        print(f"Epochs: {training_config.get('epochs', 100)}")
        print(f"Batch size: {training_config.get('batch_size', 32)}")

        self.history = self.model.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_config.get('epochs', 100),
            callbacks=self.callbacks_list,
            verbose=1
        )

        print("\nTraining completed!")

        # Save history
        self.save_history()

        return self.history

    def save_history(self) -> None:
        """Save training history to JSON file."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        history_path = Path(self.experiment_paths.root) / 'training_history.json'

        # Convert history to serializable format
        history_dict = {key: [float(val) for val in values]
                       for key, values in self.history.history.items()}

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"Training history saved to {history_path}")

    @staticmethod
    def load_history(filepath: str) -> Dict[str, List[float]]:
        """
        Load training history from JSON file.

        Args:
            filepath: Path to history file

        Returns:
            History dictionary
        """
        with open(filepath, 'r') as f:
            history = json.load(f)

        return history

    @classmethod
    def from_config(cls,
                   config: Dict[str, Any],
                   model,
                   experiment_paths,
                   physics_loss = None,
                   normalizer = None) -> 'Trainer':
        """
        Create Trainer from configuration.

        Args:
            config: Configuration dictionary
            model: LSTMAutoencoder instance
            experiment_paths: ExperimentPaths object
            physics_loss: PhysicsInformedLoss instance (optional)
            normalizer: Normalizer instance (optional)

        Returns:
            Trainer instance
        """
        return cls(
            model=model,
            config=config,
            experiment_paths=experiment_paths,
            physics_loss=physics_loss,
            normalizer=normalizer
        )
