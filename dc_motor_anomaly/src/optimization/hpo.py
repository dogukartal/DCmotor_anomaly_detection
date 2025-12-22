"""Hyperparameter optimization using Optuna."""

import optuna
from optuna.trial import Trial
import json
from typing import Dict, Any, Optional
from pathlib import Path
import tensorflow as tf
from ..utils.config import ConfigManager


class HyperparameterOptimizer:
    """Optuna-based hyperparameter search."""

    def __init__(self,
                 base_config: Dict[str, Any],
                 param_space: Dict[str, Dict[str, Any]],
                 n_trials: int = 50,
                 metric: str = 'val_loss',
                 direction: str = 'minimize',
                 study_name: Optional[str] = None):
        """
        Initialize HyperparameterOptimizer.

        Args:
            base_config: Base configuration dictionary
            param_space: Parameter space definition
            n_trials: Number of optimization trials
            metric: Metric to optimize
            direction: Optimization direction ('minimize' or 'maximize')
            study_name: Name for the optuna study
        """
        self.base_config = base_config
        self.param_space = param_space
        self.n_trials = n_trials
        self.metric = metric
        self.direction = direction
        self.study_name = study_name or 'hpo_study'
        self.study = None
        self.best_config = None

    def objective(self, trial: Trial, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial
            train_ds: Training dataset
            val_ds: Validation dataset

        Returns:
            Metric value to optimize
        """
        # Create trial config by modifying base config
        trial_config = self._create_trial_config(trial)

        # Import here to avoid circular imports
        from ..models.lstm_autoencoder import LSTMAutoencoder
        from ..models.physics_loss import PhysicsInformedLoss
        from ..data.normalizer import Normalizer
        from ..training.trainer import Trainer
        from ..utils.experiment import ExperimentManager

        # Get input shape from validation dataset
        for x, y in val_ds.take(1):
            input_shape = (x.shape[1], x.shape[2])
            break

        # Build model
        autoencoder = LSTMAutoencoder.from_config(trial_config, input_shape=input_shape)
        autoencoder.build()

        # Compile model
        training_config = trial_config['training']
        autoencoder.compile(
            optimizer=training_config['optimizer']['type'],
            learning_rate=training_config['optimizer']['learning_rate'],
            loss=training_config['loss']
        )

        # Create temporary experiment directory
        temp_exp_dir = Path('experiments') / 'hpo_temp' / f'trial_{trial.number}'
        temp_exp_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal experiment paths
        class TempExpPaths:
            def __init__(self, root):
                self.root = str(root)
                self.checkpoints = str(root / 'checkpoints')
                self.plots = str(root / 'plots')
                self.logs = str(root / 'logs')
                Path(self.checkpoints).mkdir(parents=True, exist_ok=True)
                Path(self.plots).mkdir(parents=True, exist_ok=True)
                Path(self.logs).mkdir(parents=True, exist_ok=True)

        experiment_paths = TempExpPaths(temp_exp_dir)

        # Setup physics loss if enabled
        physics_loss = None
        if trial_config.get('physics_loss', {}).get('enabled', True):
            normalizer = Normalizer.from_config(trial_config)
            physics_loss = PhysicsInformedLoss.from_config(trial_config, normalizer)

        # Train
        trainer = Trainer.from_config(
            trial_config,
            autoencoder,
            experiment_paths,
            physics_loss=physics_loss
        )

        # Reduce epochs for HPO
        trial_config['training']['epochs'] = min(trial_config['training']['epochs'], 50)

        history = trainer.train(train_ds, val_ds)

        # Get metric value
        metric_value = min(history.history[self.metric])

        return metric_value

    def _create_trial_config(self, trial: Trial) -> Dict[str, Any]:
        """
        Create configuration for trial by sampling from parameter space.

        Args:
            trial: Optuna trial

        Returns:
            Trial configuration
        """
        trial_config = self.base_config.copy()

        for param_path, param_def in self.param_space.items():
            param_type = param_def['type']

            if param_type == 'categorical':
                value = trial.suggest_categorical(param_path, param_def['choices'])
            elif param_type == 'int':
                value = trial.suggest_int(
                    param_path,
                    param_def['low'],
                    param_def['high'],
                    step=param_def.get('step', 1)
                )
            elif param_type == 'uniform':
                value = trial.suggest_uniform(
                    param_path,
                    param_def['low'],
                    param_def['high']
                )
            elif param_type == 'loguniform':
                value = trial.suggest_loguniform(
                    param_path,
                    param_def['low'],
                    param_def['high']
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            # Set value in config using dot notation
            ConfigManager.set_nested(trial_config, param_path, value)

        return trial_config

    def optimize(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset

        Returns:
            Optuna study object
        """
        print(f"Starting hyperparameter optimization...")
        print(f"Number of trials: {self.n_trials}")
        print(f"Optimizing metric: {self.metric} ({self.direction})")

        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction
        )

        # Create objective function with datasets
        def objective_with_data(trial):
            return self.objective(trial, train_ds, val_ds)

        self.study.optimize(objective_with_data, n_trials=self.n_trials)

        print("\nOptimization completed!")
        print(f"Best trial: {self.study.best_trial.number}")
        print(f"Best {self.metric}: {self.study.best_value}")

        # Create best config
        self.best_config = self._create_trial_config(self.study.best_trial)

        return self.study

    def get_best_config(self) -> Dict[str, Any]:
        """
        Get best configuration from optimization.

        Returns:
            Best configuration dictionary
        """
        if self.best_config is None:
            raise ValueError("Optimization not run. Call optimize() first.")

        return self.best_config

    def save_study(self, directory: str) -> None:
        """
        Save study results and best config.

        Args:
            directory: Directory to save results
        """
        if self.study is None:
            raise ValueError("Optimization not run. Call optimize() first.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save best config
        config_path = directory / 'best_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.best_config, f, indent=2)

        # Save study summary
        summary = {
            'best_trial': self.study.best_trial.number,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'metric': self.metric,
            'direction': self.direction
        }

        summary_path = directory / 'study_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Study results saved to {directory}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HyperparameterOptimizer':
        """
        Create HyperparameterOptimizer from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            HyperparameterOptimizer instance
        """
        hpo_config = config.get('hyperparameter_optimization', {})

        if not hpo_config.get('enabled', False):
            raise ValueError("Hyperparameter optimization not enabled in config")

        param_space = hpo_config.get('parameters', {})
        n_trials = hpo_config.get('n_trials', 50)
        metric = hpo_config.get('metric', 'val_loss')
        direction = hpo_config.get('direction', 'minimize')

        return cls(
            base_config=config,
            param_space=param_space,
            n_trials=n_trials,
            metric=metric,
            direction=direction
        )
