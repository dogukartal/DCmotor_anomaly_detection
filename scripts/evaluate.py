#!/usr/bin/env python3
"""Evaluate trained model on test data."""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

    # Load model
    checkpoint_path = exp_dir / 'checkpoints' / f'{args.checkpoint}.keras'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = keras.models.load_model(checkpoint_path)
    print("Model loaded successfully")

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
    print("Extracting test set...")
    test_x = []
    test_y = []
    for x, y in test_ds:
        test_x.append(x.numpy())
        test_y.append(y.numpy())
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    print(f"Test set size: {test_x.shape[0]} samples")

    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_loss = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_loss:.6f}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_x, verbose=1)

    # Compute reconstruction errors (MSE per sample)
    print("\nComputing reconstruction errors...")
    reconstruction_errors = np.mean(np.square(test_x - predictions), axis=(1, 2))

    # Compute statistics
    mean_error = float(np.mean(reconstruction_errors))
    std_error = float(np.std(reconstruction_errors))
    max_error = float(np.max(reconstruction_errors))
    min_error = float(np.min(reconstruction_errors))
    median_error = float(np.median(reconstruction_errors))

    # Compute threshold using percentile method (95th percentile)
    threshold_95 = float(np.percentile(reconstruction_errors, 95))
    threshold_99 = float(np.percentile(reconstruction_errors, 99))

    # Count anomalies at different thresholds
    n_anomalies_95 = int(np.sum(reconstruction_errors > threshold_95))
    n_anomalies_99 = int(np.sum(reconstruction_errors > threshold_99))

    print(f"\nEvaluation Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Mean reconstruction error: {mean_error:.6f}")
    print(f"  Std reconstruction error: {std_error:.6f}")
    print(f"  Median reconstruction error: {median_error:.6f}")
    print(f"  Max reconstruction error: {max_error:.6f}")
    print(f"  Min reconstruction error: {min_error:.6f}")
    print(f"\nThreshold Analysis:")
    print(f"  95th percentile threshold: {threshold_95:.6f}")
    print(f"    Anomalies detected: {n_anomalies_95} / {len(reconstruction_errors)} ({n_anomalies_95/len(reconstruction_errors)*100:.2f}%)")
    print(f"  99th percentile threshold: {threshold_99:.6f}")
    print(f"    Anomalies detected: {n_anomalies_99} / {len(reconstruction_errors)} ({n_anomalies_99/len(reconstruction_errors)*100:.2f}%)")

    # Save evaluation results
    results_dict = {
        'test_loss': test_loss,
        'mean_error': mean_error,
        'std_error': std_error,
        'median_error': median_error,
        'max_error': max_error,
        'min_error': min_error,
        'threshold_95': threshold_95,
        'threshold_99': threshold_99,
        'n_samples': int(len(reconstruction_errors)),
        'n_anomalies_95': n_anomalies_95,
        'n_anomalies_99': n_anomalies_99,
        'anomaly_rate_95': float(n_anomalies_95 / len(reconstruction_errors)),
        'anomaly_rate_99': float(n_anomalies_99 / len(reconstruction_errors))
    }

    eval_results_path = exp_dir / 'evaluation_results.json'
    with open(eval_results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nEvaluation results saved to {eval_results_path}")

    # Load training history if available
    history_path = exp_dir / 'training_history.json'
    history = None
    if history_path.exists():
        print(f"\nLoading training history from {history_path}")
        with open(history_path, 'r') as f:
            history = json.load(f)

    # Generate plots
    print("\nGenerating evaluation plots...")
    plotter = Plotter.from_config(config)
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Training history (if available)
    if history is not None:
        print("  - Training history plot")
        fig = plotter.plot_training_history(history, title="Training History")
        plotter.save_figure(fig, plots_dir / 'training_history')

    # Plot 2: Reconstruction error over samples
    print("  - Reconstruction error plot")
    fig = plotter.plot_reconstruction_error(reconstruction_errors, title="Test Set Reconstruction Errors")
    plotter.save_figure(fig, plots_dir / 'test_reconstruction_error')

    # Plot 3: Error distribution with thresholds
    print("  - Error distribution plot")
    fig = plotter.plot_error_distribution(
        reconstruction_errors,
        threshold=threshold_95,
        title="Test Set Error Distribution (95th percentile threshold)"
    )
    plotter.save_figure(fig, plots_dir / 'test_error_distribution')

    # Plot 4: Threshold analysis
    print("  - Threshold analysis plot")
    fig = plotter.plot_threshold_analysis(
        reconstruction_errors,
        thresholds=[threshold_95, threshold_99],
        title="Threshold Analysis"
    )
    plotter.save_figure(fig, plots_dir / 'test_threshold_analysis')

    # Plot 5: Sample reconstructions
    print("  - Sample reconstruction comparisons")
    n_samples_to_plot = min(5, test_x.shape[0])
    for i in range(n_samples_to_plot):
        fig = plotter.plot_reconstruction(
            test_x[i:i+1],
            predictions[i:i+1],
            idx=0,
            feature_names=feature_names,
            title=f"Test Sample {i+1} Reconstruction (Error: {reconstruction_errors[i]:.6f})"
        )
        plotter.save_figure(fig, plots_dir / f'test_reconstruction_sample_{i+1}')

    print(f"\nAll plots saved to {plots_dir}")
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
