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
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True, help='Path to raw simulation data (.npy)')
    parser.add_argument('--output', type=str, help='Output directory (default: data/processed)')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = ConfigManager.load(args.config)
    ConfigManager.validate(config)

    # Setup output directory
    output_dir = Path(args.output) if args.output else Path(config['paths']['processed_data'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw simulation data
    print(f"Loading raw data from {args.input}")
    raw_data = np.load(args.input, allow_pickle=True).item()

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
        title="Processed Features (First 1000 Samples)"
    )
    plotter.save_figure(fig, output_dir / 'processed_features')

    print("\nData processing completed successfully!")
    print(f"Total windows: {windows.shape[0]}")
    print(f"Window size: {windows.shape[1]}")
    print(f"Features per timestep: {windows.shape[2]}")


if __name__ == '__main__':
    main()
