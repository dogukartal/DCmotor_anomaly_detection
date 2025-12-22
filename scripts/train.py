#!/usr/bin/env python3
"""Train LSTM Autoencoder for anomaly detection."""

import argparse
import sys
from pathlib import Path
import numpy as np
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.experiment import ExperimentManager
from src.data.dataset import DatasetBuilder
from src.data.normalizer import Normalizer
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.physics_loss import PhysicsInformedLoss
from src.training.trainer import Trainer
from src.visualization.plotter import Plotter


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data (.npz)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (optional)')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = ConfigManager.load(args.config)
    ConfigManager.validate(config)

    # Create experiment
    print("Creating experiment...")
    exp_base_path = Path(config['paths']['experiments'])
    exp_manager = ExperimentManager(base_path=str(exp_base_path), config=config)
    exp_paths = exp_manager.create_experiment()
    print(f"Experiment directory: {exp_paths.root}")

    # Load processed data
    print(f"Loading processed data from {args.data}")
    loaded_data = np.load(args.data, allow_pickle=True)
    windows = loaded_data['windows']
    feature_names = list(loaded_data['feature_names'])

    print(f"Data shape: {windows.shape}")
    print(f"Features: {', '.join(feature_names)}")

    # Create datasets
    print("Creating datasets...")
    dataset_builder = DatasetBuilder.from_config(config)
    split_ratios = config['data_processing']['train_val_test_split']
    train_ds, val_ds, test_ds = dataset_builder.build(windows, split_ratios=split_ratios)

    # Get input shape
    input_shape = (windows.shape[1], windows.shape[2])
    print(f"Input shape: {input_shape}")

    # Build model
    print("Building model...")
    autoencoder = LSTMAutoencoder.from_config(config, input_shape=input_shape)
    autoencoder.build()
    autoencoder.summary()

    # Compile model
    training_config = config['training']
    autoencoder.compile(
        optimizer=training_config['optimizer']['type'],
        learning_rate=training_config['optimizer']['learning_rate'],
        loss=training_config['loss']
    )

    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        autoencoder.load(args.resume)

    # Setup physics loss
    physics_loss = None
    if config.get('physics_loss', {}).get('enabled', True):
        print("Setting up physics-informed loss...")
        # Load normalizer
        normalizer_path = Path(args.data).parent / 'normalizer_stats.json'
        if normalizer_path.exists():
            normalizer = Normalizer()
            normalizer.load_statistics(str(normalizer_path))
        else:
            print("Warning: Normalizer not found, physics loss will not denormalize")
            normalizer = None

        target_rate = config['data_processing']['target_sampling_rate_hz']
        physics_loss = PhysicsInformedLoss.from_config(config, normalizer, sampling_rate=target_rate)

    # Create plotter
    plotter = Plotter.from_config(config)

    # Setup trainer
    print("Setting up trainer...")
    trainer = Trainer.from_config(
        config=config,
        model=autoencoder,
        experiment_paths=exp_paths,
        physics_loss=physics_loss,
        normalizer=normalizer if physics_loss else None
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    history = trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        plotter=plotter,
        feature_names=feature_names
    )

    # Plot training history
    print("\nGenerating training plots...")
    fig = plotter.plot_training_history(history.history, title="Training History")
    plotter.save_figure(fig, Path(exp_paths.plots) / 'training_history')

    # Save normalizer stats to experiment
    if normalizer_path.exists():
        shutil.copy(normalizer_path, Path(exp_paths.root) / 'normalizer_stats.json')

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)
    print(f"\nExperiment directory: {exp_paths.root}")
    print(f"Best model checkpoint: {Path(exp_paths.checkpoints) / 'best_model'}")
    print(f"Training history: {Path(exp_paths.root) / 'training_history.json'}")


if __name__ == '__main__':
    main()
