#!/usr/bin/env python3
"""Run hyperparameter optimization using Optuna."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.data.dataset import DatasetBuilder
from src.optimization.hpo import HyperparameterOptimizer


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file with HPO settings')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data (.npz)')
    parser.add_argument('--output', type=str, help='Output directory (default: experiments/hpo_study)')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = ConfigManager.load(args.config)
    # Note: HPO config has different structure (references base configs), so no validation needed

    # Check if HPO is enabled
    if not config.get('hyperparameter_optimization', {}).get('enabled', False):
        print("ERROR: Hyperparameter optimization is not enabled in the configuration.")
        print("Set 'hyperparameter_optimization.enabled' to true in your config file.")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output) if args.output else Path('experiments/hpo_study')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    print(f"Loading processed data from {args.data}")
    loaded_data = np.load(args.data, allow_pickle=True)
    windows = loaded_data['windows']

    print(f"Data shape: {windows.shape}")

    # Create datasets (use only train and val for HPO)
    print("Creating datasets...")
    dataset_builder = DatasetBuilder.from_config(config)
    split_ratios = config['data_processing']['train_val_test_split']
    train_ds, val_ds, test_ds = dataset_builder.build(windows, split_ratios=split_ratios)

    # Create optimizer
    print("\nInitializing hyperparameter optimizer...")
    optimizer = HyperparameterOptimizer.from_config(config)

    hpo_config = config['hyperparameter_optimization']
    print(f"Number of trials: {hpo_config['n_trials']}")
    print(f"Metric to optimize: {hpo_config['metric']} ({hpo_config['direction']})")
    print(f"\nParameter space:")
    for param_name, param_def in hpo_config['parameters'].items():
        print(f"  {param_name}: {param_def}")

    # Run optimization
    print("\n" + "=" * 50)
    print("Starting hyperparameter optimization...")
    print("=" * 50 + "\n")

    study = optimizer.optimize(train_ds, val_ds)

    # Save results
    print("\nSaving optimization results...")
    optimizer.save_study(str(output_dir))

    # Display best parameters
    print("\n" + "=" * 50)
    print("Optimization completed!")
    print("=" * 50)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best {hpo_config['metric']}: {study.best_value:.6f}")
    print("\nBest parameters:")
    for param_name, param_value in study.best_params.items():
        print(f"  {param_name}: {param_value}")

    print(f"\nResults saved to {output_dir}")
    print(f"Best configuration: {output_dir / 'best_config.json'}")
    print(f"Study summary: {output_dir / 'study_summary.json'}")

    print("\nTo train with the best configuration, run:")
    print(f"  python scripts/train.py --config {output_dir / 'best_config.json'} --data {args.data}")


if __name__ == '__main__':
    main()
