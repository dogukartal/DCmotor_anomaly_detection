#!/usr/bin/env python3
"""Evaluate trained model on test data."""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.detector import AnomalyDetector
from src.data.dataset import DatasetBuilder
from src.visualization.plotter import Plotter
from src.utils.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--experiment', type=str, required=True, help='Path to experiment directory')
    parser.add_argument('--data', type=str, help='Path to processed data (optional, uses experiment data if not provided)')
    parser.add_argument('--checkpoint', type=str, default='best_model', help='Checkpoint to use (best_model or last_model)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    print(f"Evaluating experiment: {exp_dir.name}")

    # Load configuration
    config_path = exp_dir / 'config.json'
    config = ConfigManager.load(str(config_path))

    # Load detector
    print(f"Loading model from checkpoint: {args.checkpoint}")
    detector = AnomalyDetector.from_experiment(str(exp_dir), checkpoint=args.checkpoint)

    # Load data
    if args.data:
        data_path = Path(args.data)
    else:
        # Try to find data in experiment or default location
        data_path = Path(config['paths']['processed_data']) / 'processed_data.npz'

    print(f"Loading data from {data_path}")
    loaded_data = np.load(data_path, allow_pickle=True)
    windows = loaded_data['windows']
    feature_names = list(loaded_data['feature_names'])

    # Create datasets
    dataset_builder = DatasetBuilder.from_config(config)
    split_ratios = config['data_processing']['train_val_test_split']
    train_ds, val_ds, test_ds = dataset_builder.build(windows, split_ratios=split_ratios)

    # Get test data as numpy array
    print("Evaluating on test set...")
    test_x = []
    test_y = []
    for x, y in test_ds:
        test_x.append(x.numpy())
        test_y.append(y.numpy())
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    print(f"Test set size: {test_x.shape[0]} samples")

    # Compute reconstruction errors on test set
    test_errors = detector.compute_reconstruction_error(test_x)

    # Set threshold using percentile method
    print("Computing threshold...")
    threshold = detector.set_threshold(method='percentile', errors=test_errors, percentile=95)

    # Detect anomalies
    result = detector.detect(test_x)

    print(f"\nEvaluation Results:")
    print(f"  Threshold: {result.threshold:.6f}")
    print(f"  Mean error: {result.metadata['mean_error']:.6f}")
    print(f"  Max error: {result.metadata['max_error']:.6f}")
    print(f"  Min error: {result.metadata['min_error']:.6f}")
    print(f"  Anomalies detected: {result.metadata['n_anomalies']} / {result.metadata['n_samples']}")
    print(f"  Anomaly rate: {result.metadata['anomaly_rate']*100:.2f}%")

    # Save results
    results_dict = {
        'threshold': float(result.threshold),
        'mean_error': float(result.metadata['mean_error']),
        'max_error': float(result.metadata['max_error']),
        'min_error': float(result.metadata['min_error']),
        'n_anomalies': int(result.metadata['n_anomalies']),
        'n_samples': int(result.metadata['n_samples']),
        'anomaly_rate': float(result.metadata['anomaly_rate'])
    }

    results_path = exp_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating evaluation plots...")
    plotter = Plotter.from_config(config)

    # Plot reconstruction error
    fig = plotter.plot_reconstruction_error(result.reconstruction_errors)
    plotter.save_figure(fig, exp_dir / 'plots' / 'reconstruction_error')

    # Plot error distribution
    fig = plotter.plot_error_distribution(result.reconstruction_errors, threshold=result.threshold)
    plotter.save_figure(fig, exp_dir / 'plots' / 'error_distribution')

    # Plot reconstruction comparison for a few samples
    predictions = detector.model.predict(test_x[:5], verbose=0)
    fig = plotter.plot_reconstruction(
        test_x[:5],
        predictions,
        idx=0,
        feature_names=feature_names,
        title="Test Set Reconstruction"
    )
    plotter.save_figure(fig, exp_dir / 'plots' / 'test_reconstruction')

    print(f"\nPlots saved to {exp_dir / 'plots'}")
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
