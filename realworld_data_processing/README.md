# Real-World Data Processing for DC Motor Anomaly Detection

This standalone module processes real-world DC motor data collected from LabView into a format compatible with the anomaly detection inference pipeline.

## Overview

The system handles two types of data files collected at different sampling rates:
- **Current Data**: High-frequency current measurements (e.g., 10 kHz)
- **Communication Data**: Lower-frequency variables like voltage, velocity, position, etc. (e.g., 500 Hz)

The processor synchronizes these data sources, extracts features, creates sliding windows, normalizes, and outputs data ready for inference.

## ⚠️ CRITICAL: Feature Ordering for Inference

**Your real-world data MUST have the same feature order as your training data!**

When you train a model on simulation data and want to run inference on real-world data, the features must be in the **exact same order**. Otherwise, the model will receive features in wrong positions and produce garbage results.

### Default Configuration

The default configuration matches the simulation training data:

**Feature Order:**
1. current_rms
2. current_peak_to_peak
3. current_max
4. current_min
5. velocity (corresponds to angular_velocity in simulation)
6. voltage

**This is achieved by:**
```json
{
  "processing": {
    "input_variables": ["current", "velocity", "voltage"],
    "derived_features": {
      "current": ["rms", "peak_to_peak", "max", "min"]
    }
  }
}
```

### How to Match Your Training Data

1. **Check your simulation/training config** (`configs/model/default.json`):
   ```json
   "input_variables": ["current", "angular_velocity", "voltage"]
   ```

2. **Set real-world config to match** (using "velocity" instead of "angular_velocity"):
   ```json
   "input_variables": ["current", "velocity", "voltage"]
   ```

3. **Keep derived_features identical:**
   ```json
   "derived_features": {
     "current": ["rms", "peak_to_peak", "max", "min"]
   }
   ```

**Note:** "velocity" in real-world data corresponds to "angular_velocity" in simulation. They're the same physical quantity, just named differently based on the data source.

## Directory Structure

```
realworld_data_processing/
├── config.json                    # Configuration file (EDIT THIS)
├── process_realworld_data.py      # Main processing script
├── data/
│   ├── raw/                       # Place your input .txt files here
│   │   ├── example_current.txt
│   │   └── example_communication.txt
│   └── processed/                 # Processed output files (.npz)
└── README.md                      # This file
```

## Quick Start

### 1. Configure Your Data Sources

Edit `config.json` to match your data collection setup:

```json
{
  "data_sources": {
    "communication_data": {
      "column_mapping": {
        "time": "Test Time",
        "voltage": "CASTemperature",      // ← Change to your voltage column
        "velocity": "CASFinVelocityIE"     // ← Change to your velocity column
      }
    }
  }
}
```

**Important Configuration Points:**

1. **⚠️ Feature Order** (MOST IMPORTANT): Must match training data!
   ```json
   "input_variables": ["current", "velocity", "voltage"]
   ```
   - Controls the order features appear in output
   - Must match simulation config (use "velocity" for "angular_velocity")
   - See "Feature Ordering" section above

2. **Voltage Column**: Set `"voltage"` to the column name where you store the input voltage
   - Example: If voltage is in `CASTemperature`, use `"voltage": "CASTemperature"`

3. **Velocity Column**: Set `"velocity"` to your velocity column name
   - Example: `"velocity": "CASFinVelocityIE"`

4. **Sampling Rates**: Update if your rates differ from defaults
   ```json
   "current_data": {
     "sampling_rate_hz": 10000  // Your current sampling rate
   },
   "communication_data": {
     "sampling_rate_hz": 500    // Your communication sampling rate
   }
   ```

5. **Derived Features**: Configure what features to extract from current
   ```json
   "derived_features": {
     "current": ["rms", "peak_to_peak", "max", "min"]
   }
   ```
   Available: `rms`, `peak_to_peak`, `variance`, `mean`, `max`, `min`, `slope`, `zero_crossing_rate`

   ⚠️ **Order matters!** Feature extraction order must match training data.

### 2. Prepare Your Data Files

Place your data files in `data/raw/`:

**Current Data Format** (`<name>_current.txt`):
```
0.0000	0.0454
0.0001	0.1056
0.0002	0.0615
...
```
- Tab-separated
- Column 0: Time (seconds)
- Column 1: Current (amps)

**Communication Data Format** (`<name>_communication.txt`):
```
Test Time	Applied Command	CASModeExternalEcho	...	CASTemperature	...	CASFinVelocityIE	...
0.0020	0.0000	0.0000	...	12.5000	...	9.0000	...
0.0040	0.0000	0.0000	...	12.8000	...	10.5000	...
...
```
- Tab-separated
- First row: Headers (column names)
- Subsequent rows: Data
- Column names can have spaces (e.g., "Test Time")

### 3. Run Processing

```bash
cd realworld_data_processing

python process_realworld_data.py \
    --current data/raw/your_current_data.txt \
    --comm data/raw/your_communication_data.txt \
    --output data/processed/test_run_001
```

**Arguments:**
- `--current`: Path to current data file
- `--comm`: Path to communication data file
- `--output`: Output directory for processed files
- `--config`: (Optional) Custom config file path (default: `config.json`)
- `--name`: (Optional) Output filename prefix (default: `processed_data`)

### 4. Run Inference

Use the processed data with the main repository's inference script:

```bash
# From the main repository root
python scripts/infer.py \
    --experiment experiments/your_trained_model \
    --input realworld_data_processing/data/processed/test_run_001/processed_data.npz
```

## Output Files

After processing, you'll find in your output directory:

1. **`processed_data.npz`** - Main data file for inference
   - `windows`: numpy array of shape `(n_windows, window_size, n_features)`
   - `feature_names`: list of feature names in order

2. **`processed_data_normalizer_stats.json`** - Normalization statistics
   - Min/max values for each feature
   - Used for inverse transformation if needed

3. **`processed_data_metadata.json`** - Processing metadata
   - Configuration used
   - Processing timestamp
   - Output shape and feature information

## Configuration Reference

### Complete config.json Structure

```json
{
  "data_sources": {
    "current_data": {
      "description": "High-frequency current measurements",
      "sampling_rate_hz": 10000,        // Your current sampling rate
      "columns": {
        "time": 0,                       // Column index for time
        "current": 1                     // Column index for current
      },
      "delimiter": "\t"                  // Tab-separated
    },
    "communication_data": {
      "description": "Communication variables from LabView",
      "sampling_rate_hz": 500,           // Your communication sampling rate
      "delimiter": "\t",
      "header_row": 0,                   // Row with column names
      "data_start_row": 1,               // First data row
      "column_mapping": {
        "time": "Test Time",             // Time column name
        "voltage": "CASTemperature",     // ← EDIT: Your voltage column
        "velocity": "CASFinVelocityIE"   // ← EDIT: Your velocity column
      }
    }
  },
  "processing": {
    "target_sampling_rate_hz": 500,      // Target rate after downsampling
    "input_variables": ["current", "velocity", "voltage"],  // ⚠️ Order matters!
    "derived_features": {
      "current": ["rms", "peak_to_peak", "max", "min"]  // Features to compute
    },
    "window_size": 128,                  // Sliding window size
    "window_stride": 16,                 // Sliding window stride
    "normalization": {
      "method": "minmax",                // minmax, standard, or robust
      "feature_range": [-1, 1]           // For minmax method
    }
  },
  "output": {
    "format": "npz",
    "save_normalizer_stats": true,
    "save_metadata": true
  }
}
```

### Adding More Features

To include additional communication variables:

1. Add them to `input_variables` (⚠️ **order matters**):
   ```json
   "input_variables": ["current", "velocity", "voltage", "position"]
   ```

2. Add column mapping for the new variables:
   ```json
   "column_mapping": {
     "time": "Test Time",
     "voltage": "CASTemperature",
     "velocity": "CASFinVelocityIE",
     "position": "CASFinPositionIE"
   }
   ```

3. Update your training model's config to match the same order:
   ```json
   "input_variables": ["current", "angular_velocity", "voltage", "position"]
   ```

**Important:** The order in `input_variables` determines feature order in output. Make sure it matches your training configuration!

### Normalization Methods

- **minmax**: Scale to a range (default: [-1, 1])
  - Best for bounded features
  - Preserves zero and relationships

- **standard**: Z-score normalization (mean=0, std=1)
  - Good for normally distributed data
  - Sensitive to outliers

- **robust**: Median and IQR-based
  - Robust to outliers
  - Good for skewed distributions

## Processing Pipeline

The script performs these steps:

1. **Load Current Data**
   - Read high-frequency current measurements
   - Parse timestamp and current values

2. **Load Communication Data**
   - Read communication variables
   - Extract configured columns (voltage, velocity, etc.)

3. **Downsample Current**
   - Downsample from high rate (e.g., 10 kHz) to target rate (e.g., 500 Hz)
   - Compute derived features during downsampling (RMS, peak-to-peak, etc.)

4. **Merge Data**
   - Align current and communication data by timestamp
   - Use nearest-neighbor matching with tolerance

5. **Create Windows**
   - Apply sliding window to create sequences
   - Default: 128 timesteps with stride of 16

6. **Normalize**
   - Apply normalization (minmax, standard, or robust)
   - Store statistics for later use

7. **Save Results**
   - Save windowed data in .npz format
   - Save normalization statistics
   - Save metadata

## Troubleshooting

### Column Not Found Error

```
ValueError: Required column 'CASTemperature' not found in communication data
```

**Solution**: Check your column names in the data file and update `config.json`:
```bash
# View your column names
head -1 data/raw/your_communication_data.txt
```

Then update `column_mapping` to match exactly (including spaces).

### Timestamp Mismatch

```
Removed N rows with missing values
```

**Solution**: Check that:
1. Time ranges overlap between current and communication files
2. Time units are the same (both in seconds)
3. Tolerance in `merge_data` is appropriate (default: 0.01s)

### Not Enough Samples

```
ValueError: Not enough samples (X) for window size (128)
```

**Solution**: Either:
1. Collect more data
2. Reduce `window_size` in config.json
3. Check if data loading is working correctly

### Feature Not Found

```
ValueError: Feature 'voltage' not found in merged data
```

**Solution**: Check that:
1. Column mapping in config.json is correct
2. Column exists in communication data file
3. Feature name is in `base_features` list

## Examples

### Example 1: Basic Processing

```bash
python process_realworld_data.py \
    --current data/raw/test_20240315_current.txt \
    --comm data/raw/test_20240315_comm.txt \
    --output data/processed/test_20240315
```

### Example 2: Multiple Test Runs

```bash
# Process multiple test runs
for i in 001 002 003; do
    python process_realworld_data.py \
        --current data/raw/run_${i}_current.txt \
        --comm data/raw/run_${i}_comm.txt \
        --output data/processed/run_${i} \
        --name test_run_${i}
done
```

### Example 3: Custom Configuration

```bash
# Use a custom config for different sensor setup
python process_realworld_data.py \
    --current data/raw/sensor_v2_current.txt \
    --comm data/raw/sensor_v2_comm.txt \
    --output data/processed/sensor_v2 \
    --config configs/sensor_v2_config.json
```

## Integration with Main Repository

This module is **completely standalone** and does not import any classes from the main repository. However, its output is compatible with `scripts/infer.py`:

```bash
# 1. Process real-world data (from this directory)
cd realworld_data_processing
python process_realworld_data.py \
    --current data/raw/field_test.txt \
    --comm data/raw/field_comm.txt \
    --output data/processed/field_test

# 2. Run inference (from main repository root)
cd ..
python scripts/infer.py \
    --experiment experiments/my_trained_model \
    --input realworld_data_processing/data/processed/field_test/processed_data.npz \
    --threshold 0.05
```

## Notes

- **Time Alignment**: The processor uses nearest-neighbor timestamp matching with 10ms tolerance
- **Feature Order**: Features are ordered as: base features + derived features
- **Window Overlap**: Default stride (16) creates overlapping windows for smooth analysis
- **Normalization**: Use the same normalization method as used during model training
- **Missing Data**: Rows with any NaN values are automatically removed after merging

## Future Enhancements

To add more features in the future:

1. Add the column name to `column_mapping` in config.json
2. Add the feature name to `base_features` list
3. The processor will automatically include it in the output

No code changes needed for adding more columns from communication data!
