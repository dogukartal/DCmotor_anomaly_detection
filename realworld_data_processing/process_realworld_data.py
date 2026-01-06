#!/usr/bin/env python3
"""
Standalone Real-World Data Processor for DC Motor Anomaly Detection

This script processes real-world data collected from LabView:
  - High-frequency current measurements (e.g., 10 kHz)
  - Lower-frequency communication variables (e.g., 500 Hz)

Output format is compatible with the infer.py script from the main repository.
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


class RealWorldDataProcessor:
    """Processes real-world DC motor data into format compatible with inference."""

    def __init__(self, config_path: str):
        """
        Initialize processor with configuration.

        Args:
            config_path: Path to configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.current_data = None
        self.comm_data = None
        self.processed_data = None
        self.normalizer_stats = {}

    def load_current_data(self, file_path: str) -> pd.DataFrame:
        """
        Load high-frequency current measurement data.

        Args:
            file_path: Path to current data txt file

        Returns:
            DataFrame with time and current columns
        """
        print(f"Loading current data from: {file_path}")

        config = self.config['data_sources']['current_data']
        delimiter = config['delimiter']

        # Read the file
        data = np.loadtxt(file_path, delimiter=delimiter)

        # Create DataFrame
        df = pd.DataFrame({
            'time': data[:, config['columns']['time']],
            'current': data[:, config['columns']['current']]
        })

        print(f"  Loaded {len(df)} samples at {config['sampling_rate_hz']} Hz")
        print(f"  Time range: {df['time'].min():.4f} - {df['time'].max():.4f} seconds")

        return df

    def load_communication_data(self, file_path: str) -> pd.DataFrame:
        """
        Load communication variables data.

        Args:
            file_path: Path to communication data txt file

        Returns:
            DataFrame with communication variables
        """
        print(f"Loading communication data from: {file_path}")

        config = self.config['data_sources']['communication_data']
        delimiter = config['delimiter']

        # Read the file with pandas to handle headers properly
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            header=config['header_row'],
            skiprows=range(1, config['data_start_row']) if config['data_start_row'] > 1 else None
        )

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        print(f"  Loaded {len(df)} samples at {config['sampling_rate_hz']} Hz")
        print(f"  Available columns: {list(df.columns)}")

        # Extract and rename the columns we need based on mapping
        column_mapping = config['column_mapping']

        # Verify required columns exist
        for key, col_name in column_mapping.items():
            if col_name not in df.columns:
                print(f"  WARNING: Column '{col_name}' not found in data!")
                print(f"  Available columns: {list(df.columns)}")
                raise ValueError(f"Required column '{col_name}' not found in communication data")

        # Create new dataframe with renamed columns
        result_df = pd.DataFrame({
            'time': df[column_mapping['time']],
            'voltage': df[column_mapping['voltage']],
            'velocity': df[column_mapping['velocity']]
        })

        print(f"  Time range: {result_df['time'].min():.4f} - {result_df['time'].max():.4f} seconds")

        return result_df

    def downsample_current(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """
        Downsample current data to target sampling rate with feature extraction.

        Args:
            current_df: DataFrame with high-frequency current data

        Returns:
            DataFrame with downsampled current and derived features
        """
        print("Downsampling current data...")

        source_rate = self.config['data_sources']['current_data']['sampling_rate_hz']
        target_rate = self.config['processing']['target_sampling_rate_hz']
        downsample_factor = source_rate // target_rate

        print(f"  Downsampling from {source_rate} Hz to {target_rate} Hz (factor: {downsample_factor})")

        # Calculate number of complete windows
        n_samples = len(current_df)
        n_windows = n_samples // downsample_factor
        n_samples_used = n_windows * downsample_factor

        # Trim data to complete windows
        times = current_df['time'].values[:n_samples_used]
        currents = current_df['current'].values[:n_samples_used]

        # Reshape into windows for vectorized operations
        time_windows = times.reshape(n_windows, downsample_factor)
        current_windows = currents.reshape(n_windows, downsample_factor)

        # Calculate downsampled times
        downsampled_times = time_windows.mean(axis=1)

        # Get derived features configuration
        derived_features = self.config['processing']['derived_features'].get('current', [])

        result = {
            'time': downsampled_times
        }

        # Only compute derived features if specified
        # Current is high-frequency and should ONLY appear through derived features
        if len(derived_features) > 0:
            print(f"  Computing derived features: {derived_features}")
            for feature_type in derived_features:
                feature_values = self._compute_feature(current_windows, feature_type)
                result[f'current_{feature_type}'] = feature_values
        else:
            print("  No derived features specified for current - current will not appear in output")

        result_df = pd.DataFrame(result)
        n_features = len(result_df.columns) - 1  # Exclude 'time' column
        print(f"  Result: {len(result_df)} samples with {n_features} current features")

        return result_df

    def _compute_feature(self, windows: np.ndarray, feature_type: str) -> np.ndarray:
        """
        Compute derived feature for each window.

        Args:
            windows: Array of shape (n_windows, window_size)
            feature_type: Type of feature to compute

        Returns:
            Array of feature values (n_windows,)
        """
        if feature_type == 'rms':
            return np.sqrt(np.mean(windows**2, axis=1))
        elif feature_type == 'peak_to_peak':
            return np.ptp(windows, axis=1)
        elif feature_type == 'variance':
            return np.var(windows, axis=1)
        elif feature_type == 'mean':
            return np.mean(windows, axis=1)
        elif feature_type == 'max':
            return np.max(windows, axis=1)
        elif feature_type == 'min':
            return np.min(windows, axis=1)
        elif feature_type == 'slope':
            # Calculate linear trend for each window
            slopes = np.zeros(windows.shape[0])
            for i in range(windows.shape[0]):
                x = np.arange(windows.shape[1])
                coeffs = np.polyfit(x, windows[i], 1)
                slopes[i] = coeffs[0]
            return slopes
        elif feature_type == 'zero_crossing_rate':
            # Count sign changes
            signs = np.sign(windows)
            sign_changes = np.abs(np.diff(signs, axis=1))
            return np.sum(sign_changes > 0, axis=1) / windows.shape[1]
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def merge_data(self, current_df: pd.DataFrame, comm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge downsampled current data with communication data based on timestamps.

        Args:
            current_df: Downsampled current data with derived features
            comm_df: Communication data

        Returns:
            Merged DataFrame
        """
        print("Merging current and communication data...")

        # Use pandas merge_asof for time-based alignment
        # This finds the nearest timestamp match
        merged = pd.merge_asof(
            current_df.sort_values('time'),
            comm_df.sort_values('time'),
            on='time',
            direction='nearest',
            tolerance=0.01  # 10ms tolerance
        )

        # Remove any rows with NaN values
        initial_len = len(merged)
        merged = merged.dropna()
        final_len = len(merged)

        if initial_len > final_len:
            print(f"  Removed {initial_len - final_len} rows with missing values")

        print(f"  Merged data: {len(merged)} samples")
        print(f"  Features: {[col for col in merged.columns if col != 'time']}")

        return merged

    def create_windows(self, data_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create sliding windows from the merged data.

        Args:
            data_df: Merged DataFrame with all features

        Returns:
            Tuple of (windowed_data, feature_names)
            windowed_data shape: (n_windows, window_size, n_features)
        """
        print("Creating sliding windows...")

        window_size = self.config['processing']['window_size']
        window_stride = self.config['processing']['window_stride']

        # Get feature columns (exclude time)
        # base_features should only contain low-frequency variables from communication data
        # (e.g., voltage, velocity) - NOT high-frequency current
        base_features = self.config['processing']['base_features']
        derived_features = self.config['processing']['derived_features'].get('current', [])

        # Build feature list in order: base features (low-freq) + derived features (from high-freq current)
        feature_names = []

        # Add base low-frequency features
        for feature in base_features:
            feature_names.append(feature)

        # Add derived current features
        for feature_type in derived_features:
            feature_names.append(f'current_{feature_type}')

        # Extract feature data
        feature_data = []
        for feature_name in feature_names:
            if feature_name not in data_df.columns:
                raise ValueError(f"Feature '{feature_name}' not found in merged data. "
                               f"Available: {list(data_df.columns)}")
            feature_data.append(data_df[feature_name].values)

        feature_matrix = np.column_stack(feature_data)

        # Create sliding windows
        n_samples = len(feature_matrix)
        n_features = len(feature_names)

        if n_samples < window_size:
            raise ValueError(f"Not enough samples ({n_samples}) for window size ({window_size})")

        n_windows = (n_samples - window_size) // window_stride + 1

        windows = np.zeros((n_windows, window_size, n_features))

        for i in range(n_windows):
            start_idx = i * window_stride
            end_idx = start_idx + window_size
            windows[i] = feature_matrix[start_idx:end_idx]

        print(f"  Created {n_windows} windows of shape ({window_size}, {n_features})")
        print(f"  Feature order: {feature_names}")

        return windows, feature_names

    def normalize(self, windows: np.ndarray) -> np.ndarray:
        """
        Normalize the windowed data.

        Args:
            windows: Array of shape (n_windows, window_size, n_features)

        Returns:
            Normalized windows
        """
        print("Normalizing data...")

        method = self.config['processing']['normalization']['method']

        if method == 'minmax':
            feature_range = self.config['processing']['normalization']['feature_range']
            return self._minmax_normalize(windows, feature_range)
        elif method == 'standard':
            return self._standard_normalize(windows)
        elif method == 'robust':
            return self._robust_normalize(windows)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _minmax_normalize(self, windows: np.ndarray, feature_range: List[float]) -> np.ndarray:
        """MinMax normalization to specified range."""
        n_features = windows.shape[2]
        normalized = windows.copy()

        min_val, max_val = feature_range

        for i in range(n_features):
            feature_data = windows[:, :, i]
            data_min = feature_data.min()
            data_max = feature_data.max()

            # Store statistics
            self.normalizer_stats[i] = {
                'method': 'minmax',
                'data_min': float(data_min),
                'data_max': float(data_max),
                'feature_range': feature_range
            }

            # Normalize
            if data_max > data_min:
                normalized[:, :, i] = (feature_data - data_min) / (data_max - data_min)
                normalized[:, :, i] = normalized[:, :, i] * (max_val - min_val) + min_val
            else:
                print(f"  WARNING: Feature {i} has constant value, setting to middle of range")
                normalized[:, :, i] = (min_val + max_val) / 2

        print(f"  Normalized to range [{min_val}, {max_val}]")
        return normalized

    def _standard_normalize(self, windows: np.ndarray) -> np.ndarray:
        """Standard (z-score) normalization."""
        n_features = windows.shape[2]
        normalized = windows.copy()

        for i in range(n_features):
            feature_data = windows[:, :, i]
            mean = feature_data.mean()
            std = feature_data.std()

            self.normalizer_stats[i] = {
                'method': 'standard',
                'mean': float(mean),
                'std': float(std)
            }

            if std > 0:
                normalized[:, :, i] = (feature_data - mean) / std
            else:
                print(f"  WARNING: Feature {i} has zero std, setting to zero")
                normalized[:, :, i] = 0

        print("  Normalized using z-score")
        return normalized

    def _robust_normalize(self, windows: np.ndarray) -> np.ndarray:
        """Robust normalization using median and IQR."""
        n_features = windows.shape[2]
        normalized = windows.copy()

        for i in range(n_features):
            feature_data = windows[:, :, i]
            median = np.median(feature_data)
            q25 = np.percentile(feature_data, 25)
            q75 = np.percentile(feature_data, 75)
            iqr = q75 - q25

            self.normalizer_stats[i] = {
                'method': 'robust',
                'median': float(median),
                'q25': float(q25),
                'q75': float(q75),
                'iqr': float(iqr)
            }

            if iqr > 0:
                normalized[:, :, i] = (feature_data - median) / iqr
            else:
                print(f"  WARNING: Feature {i} has zero IQR, setting to zero")
                normalized[:, :, i] = 0

        print("  Normalized using median and IQR")
        return normalized

    def process(self, current_file: str, comm_file: str, output_dir: str, output_name: str = "processed_data"):
        """
        Main processing pipeline.

        Args:
            current_file: Path to current data txt file
            comm_file: Path to communication data txt file
            output_dir: Directory to save processed data
            output_name: Name for output files (without extension)
        """
        print("="*80)
        print("Starting Real-World Data Processing Pipeline")
        print("="*80)

        # Load data
        current_df = self.load_current_data(current_file)
        comm_df = self.load_communication_data(comm_file)

        # Downsample current data
        downsampled_current = self.downsample_current(current_df)

        # Merge data
        merged_df = self.merge_data(downsampled_current, comm_df)

        # Create windows
        windows, feature_names = self.create_windows(merged_df)

        # Normalize
        normalized_windows = self.normalize(windows)

        # Save results
        self._save_results(normalized_windows, feature_names, output_dir, output_name)

        print("="*80)
        print("Processing Complete!")
        print("="*80)

    def _save_results(self, windows: np.ndarray, feature_names: List[str],
                     output_dir: str, output_name: str):
        """Save processed data and metadata."""
        print(f"Saving results to: {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main data file
        npz_path = output_path / f"{output_name}.npz"
        np.savez_compressed(
            npz_path,
            windows=windows,
            feature_names=feature_names
        )
        print(f"  Saved: {npz_path}")
        print(f"    Shape: {windows.shape}")
        print(f"    Features: {feature_names}")

        # Save normalizer statistics
        if self.config['output']['save_normalizer_stats']:
            stats_path = output_path / f"{output_name}_normalizer_stats.json"
            stats_with_names = {
                feature_names[i]: self.normalizer_stats[i]
                for i in range(len(feature_names))
            }
            with open(stats_path, 'w') as f:
                json.dump(stats_with_names, f, indent=2)
            print(f"  Saved: {stats_path}")

        # Save metadata
        if self.config['output']['save_metadata']:
            metadata = {
                'config': self.config,
                'output_shape': list(windows.shape),
                'feature_names': feature_names,
                'processing_date': pd.Timestamp.now().isoformat()
            }
            metadata_path = output_path / f"{output_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  Saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process real-world DC motor data for anomaly detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process data with default config
  python process_realworld_data.py \\
      --current data/raw/current_20240101_120000.txt \\
      --comm data/raw/communication_20240101_120000.txt \\
      --output data/processed/test_20240101

  # Use custom config
  python process_realworld_data.py \\
      --current data/raw/current.txt \\
      --comm data/raw/comm.txt \\
      --output data/processed/my_test \\
      --config custom_config.json
        """
    )

    parser.add_argument(
        '--current',
        type=str,
        required=True,
        help='Path to current data txt file (high frequency)'
    )

    parser.add_argument(
        '--comm',
        type=str,
        required=True,
        help='Path to communication data txt file (lower frequency)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration JSON file (default: config.json)'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='processed_data',
        help='Name for output files (default: processed_data)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.current).exists():
        print(f"ERROR: Current data file not found: {args.current}")
        sys.exit(1)

    if not Path(args.comm).exists():
        print(f"ERROR: Communication data file not found: {args.comm}")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    # Process data
    try:
        processor = RealWorldDataProcessor(args.config)
        processor.process(args.current, args.comm, args.output, args.name)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
