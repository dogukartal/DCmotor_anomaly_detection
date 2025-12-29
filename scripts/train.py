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
    parser.add_argument('--model-config', type=str, help='Model config ID or path (default: default)')
    parser.add_argument('--sim-id', type=str, help='Simulation ID to use data from (default: from model config)')
    parser.add_argument('--data', type=str, help='Path to processed data (optional, auto-detected from sim-id)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (optional)')
    args = parser.parse_args()

    # Load model configuration
    if args.model_config:
        if args.model_config.endswith('.json'):
            # Full path provided
            print(f"Loading model configuration from {args.model_config}")
            model_config = ConfigManager.load(args.model_config)
            ConfigManager.validate(model_config, config_type='model')
        else:
            # ID provided
            print(f"Loading model configuration: {args.model_config}")
            model_config = ConfigManager.load_model_config(args.model_config)
    else:
        # Use default
        print("Loading default model configuration")
        model_config = ConfigManager.load_model_config('default')

    # Determine simulation ID
    simulation_id = args.sim_id if args.sim_id else model_config.get('simulation_id', 'default')
    print(f"Simulation ID: {simulation_id}")

    # Load simulation configuration (needed for physics loss motor params)
    print(f"Loading simulation configuration: {simulation_id}")
    simulation_config = ConfigManager.load_simulation_config(simulation_id)

    # Merge configs for backward compatibility with existing code
    config = ConfigManager.merge_configs(simulation_config, model_config)

    # Determine data path
    if args.data:
        data_path = Path(args.data)
    else:
        # Auto-detect from simulation_id
        processed_dir = ConfigManager.get_simulation_data_path(simulation_id, 'processed')
        data_path = processed_dir / 'processed_data.npz'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_path}\n"
            f"Please run process_data.py first with --sim-id {simulation_id}"
        )

    # Create experiment
    print("Creating experiment...")
    exp_base_path = Path(config['paths']['experiments'])
    exp_manager = ExperimentManager(base_path=str(exp_base_path), config=config)
    exp_paths = exp_manager.create_experiment()
    print(f"Experiment directory: {exp_paths.root}")

    # Load processed data
    print(f"Loading processed data from {data_path}")
    loaded_data = np.load(data_path, allow_pickle=True)
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

    # Setup physics loss BEFORE compilation (needed for loss function)
    physics_loss = None
    normalizer = None
    normalizer_path = data_path.parent / 'normalizer_stats.json'

    if config.get('physics_loss', {}).get('enabled', True):
        print("\nSetting up physics-informed loss...")
        # Load normalizer
        if normalizer_path.exists():
            normalizer = Normalizer()
            normalizer.load_statistics(str(normalizer_path))
            print("  ✓ Loaded normalizer for physics loss denormalization")
        else:
            print("  ⚠ Warning: Normalizer not found, physics loss will not denormalize")

        target_rate = config['data_processing']['target_sampling_rate_hz']
        physics_loss = PhysicsInformedLoss.from_config(config, normalizer, sampling_rate=target_rate)
        print(f"  ✓ Physics loss enabled with weight={physics_loss.physics_weight}")
        print(f"  ✓ Physics loss will activate at epoch {physics_loss.start_epoch}")

    # Compile model (with physics loss if enabled, otherwise use standard loss)
    training_config = config['training']
    loss_function = physics_loss if physics_loss is not None else training_config['loss']

    print("\nCompiling model...")
    print(f"  Optimizer: {training_config['optimizer']['type']}")
    print(f"  Learning rate: {training_config['optimizer']['learning_rate']}")
    if physics_loss is not None:
        print(f"  Loss function: PhysicsInformedLoss (reconstruction + physics constraints)")
    else:
        print(f"  Loss function: {training_config['loss']}")

    autoencoder.compile(
        optimizer=training_config['optimizer']['type'],
        learning_rate=training_config['optimizer']['learning_rate'],
        loss=loss_function  # Now using PhysicsInformedLoss if enabled!
    )

    # Load checkpoint if resuming
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        autoencoder.load(args.resume)

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
