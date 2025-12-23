#!/usr/bin/env python3
"""Process raw simulation data: downsample, extract features, create windows."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.data.processor import DataProcessor
from src.data.normalizer import Normalizer
from src.visualization.plotter import Plotter


def main():
    parser = argparse.ArgumentParser(description='Process raw simulation data')
    parser.add_argument('--model-config', type=str, help='Model config ID or path (default: default)')
    parser.add_argument('--sim-id', type=str, help='Simulation ID to process data from (default: from model config)')
    parser.add_argument('--input', type=str, help='Path to raw simulation data (optional, auto-detected from sim-id)')
    parser.add_argument('--output', type=str, help='Output directory (optional, default: data/processed/{simulation_id})')
    args = parser.parse_args()

    # Load model configuration (contains data processing settings)
    if args.model_config:
        if args.model_config.endswith('.json'):
            # Full path provided
            print(f"Loading model configuration from {args.model_config}")
            config = ConfigManager.load(args.model_config)
            ConfigManager.validate(config, config_type='model')
        else:
            # ID provided
            print(f"Loading model configuration: {args.model_config}")
            config = ConfigManager.load_model_config(args.model_config)
    else:
        # Use default
        print("Loading default model configuration")
        config = ConfigManager.load_model_config('default')

    # Determine simulation ID
    simulation_id = args.sim_id if args.sim_id else config.get('simulation_id', 'default')
    print(f"Simulation ID: {simulation_id}")

    # Setup directories
    raw_data_dir = ConfigManager.get_simulation_data_path(simulation_id, 'raw')
    output_dir = Path(args.output) if args.output else ConfigManager.get_simulation_data_path(simulation_id, 'processed')

    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        # Auto-detect from simulation_id
        input_file = raw_data_dir / 'simulation_result.npy'

    if not input_file.exists():
        raise FileNotFoundError(
            f"Raw simulation data not found: {input_file}\n"
            f"Please run simulate.py first with --sim-config {simulation_id}"
        )

    # Load raw simulation data
    print(f"Loading raw data from {input_file}")
    raw_data = np.load(input_file, allow_pickle=True).item()

    # Create processor
    print("Processing data...")
    processor = DataProcessor.from_config(config)

    # Process data
    processed_data = processor.process(raw_data)
    print(f"Processed data shape: {processed_data.data.shape}")
    print(f"Features: {', '.join(processed_data.feature_names)}")

    # Normalize data
    print("Normalizing data...")
    normalizer = Normalizer.from_config(config)
    normalized_data = normalizer.fit_transform(processed_data.data)

    # Save normalizer statistics
    normalizer_stats_path = output_dir / 'normalizer_stats.json'
    normalizer.save_statistics(str(normalizer_stats_path))
    print(f"Normalizer statistics saved to {normalizer_stats_path}")

    # Create windows
    print("Creating sliding windows...")
    windows = processor.create_windows(normalized_data)
    print(f"Windows shape: {windows.shape}")

    # Save processed data
    output_file = output_dir / 'processed_data.npz'
    np.savez_compressed(
        output_file,
        windows=windows,
        feature_names=processed_data.feature_names,
        metadata=processed_data.metadata,
        normalization=normalizer.get_statistics()
    )
    print(f"Processed data saved to {output_file}")

    # Plot features
    print("Generating plots...")
    plotter = Plotter.from_config(config)

    # Plot first 1000 samples of processed features
    sample_size = min(1000, processed_data.data.shape[0])
    fig = plotter.plot_features(
        data=processed_data.data[:sample_size],
        feature_names=processed_data.feature_names,
        title=f"Processed Features - {simulation_id} (First 1000 Samples)"
    )
    plotter.save_figure(fig, output_dir / 'processed_features')

    print("\nData processing completed successfully!")
    print(f"Simulation ID: {simulation_id}")
    print(f"Total windows: {windows.shape[0]}")
    print(f"Window size: {windows.shape[1]}")
    print(f"Features per timestep: {windows.shape[2]}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
