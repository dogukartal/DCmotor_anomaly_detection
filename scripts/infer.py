#!/usr/bin/env python3
"""Run anomaly detection inference on new data."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.detector import AnomalyDetector
from src.visualization.plotter import Plotter
from src.utils.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description='Run anomaly detection inference')
    parser.add_argument('--experiment', type=str, required=True, help='Path to experiment directory')
    parser.add_argument('--input', type=str, required=True, help='Path to processed input data (.npz)')
    parser.add_argument('--threshold', type=float, help='Anomaly threshold (optional, auto-compute if not provided)')
    parser.add_argument('--checkpoint', type=str, default='best_model', help='Checkpoint to use (best_model or last_model)')
    parser.add_argument('--output', type=str, help='Output directory (default: experiment/inference)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    print(f"Running inference using experiment: {exp_dir.name}")

    # Load configuration
    config_path = exp_dir / 'config.json'
    config = ConfigManager.load(str(config_path))

    # Setup output directory
    output_dir = Path(args.output) if args.output else (exp_dir / 'inference')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detector
    print(f"Loading model from checkpoint: {args.checkpoint}")
    detector = AnomalyDetector.from_experiment(str(exp_dir), checkpoint=args.checkpoint)

    # Load input data
    print(f"Loading input data from {args.input}")
    loaded_data = np.load(args.input, allow_pickle=True)
    windows = loaded_data['windows']
    feature_names = list(loaded_data['feature_names'])

    print(f"Input data shape: {windows.shape}")
    print(f"Number of samples: {windows.shape[0]}")

    # Set threshold
    if args.threshold:
        print(f"Using provided threshold: {args.threshold}")
        detector.threshold = args.threshold
    elif detector.threshold is None:
        print("No threshold provided and no threshold found in experiment.")
        print("Computing threshold from input data (95th percentile)...")
        errors = detector.compute_reconstruction_error(windows)
        detector.set_threshold(method='percentile', errors=errors, percentile=95)

    # Run inference
    print("\nRunning anomaly detection...")
    result = detector.detect(windows)

    print(f"\nInference Results:")
    print(f"  Threshold: {result.threshold:.6f}")
    print(f"  Mean error: {result.metadata['mean_error']:.6f}")
    print(f"  Max error: {result.metadata['max_error']:.6f}")
    print(f"  Min error: {result.metadata['min_error']:.6f}")
    print(f"  Anomalies detected: {result.metadata['n_anomalies']} / {result.metadata['n_samples']}")
    print(f"  Anomaly rate: {result.metadata['anomaly_rate']*100:.2f}%")

    # Save results
    results_path = output_dir / 'anomaly_results.npz'
    detector.save_results(result, str(results_path))
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plotter = Plotter.from_config(config)

    # Plot reconstruction error
    fig = plotter.plot_reconstruction_error(result.reconstruction_errors)
    plotter.save_figure(fig, output_dir / 'reconstruction_error')

    # Plot error distribution
    fig = plotter.plot_error_distribution(result.reconstruction_errors, threshold=result.threshold)
    plotter.save_figure(fig, output_dir / 'error_distribution')

    # Plot reconstruction for first few samples
    predictions = detector.model.predict(windows[:5], verbose=0)
    fig = plotter.plot_reconstruction(
        windows[:5],
        predictions,
        idx=0,
        feature_names=feature_names,
        title="Inference Reconstruction"
    )
    plotter.save_figure(fig, output_dir / 'reconstruction_sample')

    # If anomalies found, plot some examples
    if result.metadata['n_anomalies'] > 0:
        anomaly_indices = np.where(result.is_anomaly)[0]
        n_examples = min(3, len(anomaly_indices))

        for i in range(n_examples):
            idx = anomaly_indices[i]
            sample_data = windows[idx:idx+1]
            sample_pred = detector.model.predict(sample_data, verbose=0)

            fig = plotter.plot_reconstruction(
                sample_data,
                sample_pred,
                idx=0,
                feature_names=feature_names,
                title=f"Anomaly Example {i+1} (Error: {result.reconstruction_errors[idx]:.6f})"
            )
            plotter.save_figure(fig, output_dir / f'anomaly_example_{i+1}')

    print(f"\nPlots saved to {output_dir}")

    # Save anomaly indices
    if result.metadata['n_anomalies'] > 0:
        anomaly_indices = np.where(result.is_anomaly)[0]
        anomaly_file = output_dir / 'anomaly_indices.txt'
        np.savetxt(anomaly_file, anomaly_indices, fmt='%d')
        print(f"Anomaly indices saved to {anomaly_file}")

    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
