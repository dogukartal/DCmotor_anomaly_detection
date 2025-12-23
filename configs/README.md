# Configuration System

This directory contains configuration files organized by their purpose.

## Directory Structure

```
configs/
├── simulation/    # Simulation configurations
│   └── default.json
├── model/        # Model configurations
│   └── default.json
└── README.md     # This file
```

## Configuration Types

### Simulation Configs (`configs/simulation/`)

Simulation configs define:
- **simulation_id**: Unique identifier for this simulation configuration
- **input_generator**: Voltage signal generation parameters
- **simulation**: DC motor parameters and simulation settings
- **plotting**: Visualization settings
- **paths**: Data paths for inputs and raw data

**Example:**
```json
{
  "simulation_id": "default",
  "description": "Default DC motor simulation configuration",
  "input_generator": { ... },
  "simulation": {
    "motor_params": { "R": 1.0, "L": 0.5, ... }
  }
}
```

### Model Configs (`configs/model/`)

Model configs define:
- **model_id**: Unique identifier for this model configuration
- **simulation_id**: Which simulation config to use (for motor params in physics loss)
- **experiment**: Experiment naming settings
- **data_processing**: Feature extraction and windowing
- **normalization**: Data normalization settings
- **model**: LSTM autoencoder architecture
- **physics_loss**: Physics-informed loss settings
- **training**: Optimizer, epochs, callbacks
- **hyperparameter_optimization**: Optuna settings
- **plotting**: Visualization settings
- **paths**: Data paths for processed data and experiments

**Example:**
```json
{
  "model_id": "default",
  "simulation_id": "default",
  "description": "Default LSTM autoencoder model configuration",
  "model": {
    "encoder_units": [64, 32],
    ...
  },
  "training": { ... }
}
```

## Data Organization

The new config system organizes data by simulation ID:

```
data/
├── inputs/
│   └── {simulation_id}/
│       ├── voltage_input.npy
│       └── voltage_input.png
├── raw/
│   └── {simulation_id}/
│       ├── simulation_result.npy
│       └── simulation_results.png
└── processed/
    └── {simulation_id}/
        ├── processed_data.npz
        ├── normalizer_stats.json
        └── processed_features.png
```

This organization allows you to:
- Use 1-2 simulation configs with hundreds of model configs
- Easily track which data came from which simulation
- Avoid overwriting data files
- Keep the workspace organized

## Usage

### Running Simulations

```bash
# Use default simulation config
python scripts/simulate.py

# Use specific simulation config by ID
python scripts/simulate.py --sim-config default

# Use custom simulation config file
python scripts/simulate.py --sim-config path/to/custom.json
```

### Processing Data

```bash
# Use default model config, process data from default simulation
python scripts/process_data.py

# Use specific model config and simulation ID
python scripts/process_data.py --model-config default --sim-id default

# Explicit input/output paths
python scripts/process_data.py --input data/raw/default/simulation_result.npy
```

### Training Models

```bash
# Use default model config, auto-detect data from simulation_id
python scripts/train.py

# Use specific model config
python scripts/train.py --model-config default

# Use specific simulation data
python scripts/train.py --model-config default --sim-id default

# Explicit data path
python scripts/train.py --model-config default --data data/processed/default/processed_data.npz
```

## Creating Custom Configs

### Creating a New Simulation Config

1. Copy `configs/simulation/default.json` to `configs/simulation/your_name.json`
2. Update the `simulation_id` to match the filename
3. Modify motor parameters or signal generation as needed
4. Run simulation: `python scripts/simulate.py --sim-config your_name`

Example custom simulation config (`configs/simulation/high_friction.json`):
```json
{
  "simulation_id": "high_friction",
  "description": "High friction DC motor simulation",
  "simulation": {
    "motor_params": {
      "R": 1.0,
      "L": 0.5,
      "Kt": 0.01,
      "Ke": 0.01,
      "J": 0.01,
      "B": 0.5  // Increased friction
    }
  }
}
```

### Creating a New Model Config

1. Copy `configs/model/default.json` to `configs/model/your_name.json`
2. Update the `model_id` to match the filename
3. Set `simulation_id` to reference which simulation to use
4. Modify model architecture, training params as needed
5. Run training: `python scripts/train.py --model-config your_name`

Example custom model config (`configs/model/deep_encoder.json`):
```json
{
  "model_id": "deep_encoder",
  "simulation_id": "default",
  "description": "Deeper encoder architecture",
  "model": {
    "encoder_units": [128, 64, 32],
    "decoder_units": [32, 64, 128],
    "bottleneck": {
      "enabled": true,
      "units": 8,
      "activation": "relu"
    }
  }
}
```

## Workflow Example

Here's a complete workflow using the new config system:

```bash
# 1. Create a custom simulation with high friction
# Edit configs/simulation/high_friction.json

# 2. Run simulation
python scripts/simulate.py --sim-config high_friction

# 3. Process data with default model config
python scripts/process_data.py --model-config default --sim-id high_friction

# 4. Train multiple models on the same simulation data
python scripts/train.py --model-config default --sim-id high_friction
python scripts/train.py --model-config deep_encoder --sim-id high_friction

# 5. Compare experiment results in experiments/
```

## Benefits

1. **Separation of Concerns**: Simulation parameters separated from model parameters
2. **Reusability**: Run many model configs on the same simulation data
3. **Organization**: Data organized by simulation_id prevents overwriting
4. **Traceability**: Easy to track which simulation produced which data
5. **Experimentation**: Easily create and compare different configurations

## Backward Compatibility

The old `configs/default.json` file is still available for reference. The new system uses:
- `configs/simulation/default.json` - Simulation settings
- `configs/model/default.json` - Model settings

These two files together contain all the settings from the original `default.json`.
