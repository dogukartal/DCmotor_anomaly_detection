# Configuration System

This directory contains configuration files organized by their purpose.

## Directory Structure

```
configs/
├── simulation/    # Simulation configurations
│   └── default.json
├── model/        # Model configurations
│   └── default.json
├── hpo/          # Hyperparameter optimization configurations
│   └── hpo_config.json
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

### HPO Configs (`configs/hpo/`)

HPO (Hyperparameter Optimization) configs define:
- **hyperparameter_optimization.enabled**: Enable/disable Optuna HPO
- **hyperparameter_optimization.n_trials**: Number of trials to run
- **hyperparameter_optimization.metric**: Metric to optimize (e.g., 'val_loss')
- **hyperparameter_optimization.direction**: 'minimize' or 'maximize'
- **hyperparameter_optimization.parameters**: Parameter search space with optional constraints

**Parameter Types:**
- `categorical`: Choose from discrete options
- `int`: Integer values in a range
- `uniform`: Continuous values in a range
- `loguniform`: Log-scale continuous values

**Parameter Constraints:**

You can add constraints to ensure parameter dependencies are satisfied. For example, ensuring the bottleneck size doesn't exceed the last encoder layer:

```json
{
  "model.encoder_units": {
    "type": "categorical",
    "choices": [[32], [64], [64, 32], [128, 64]]
  },
  "model.bottleneck.units": {
    "type": "int",
    "low": 8,
    "high": 64,
    "step": 8,
    "constraint": {
      "type": "max_from_last",
      "parameter": "model.encoder_units",
      "multiplier": 1.0
    }
  }
}
```

**Constraint Types:**
- `max_from_last`: Ensures the parameter value doesn't exceed the last element of another parameter (useful for list parameters like encoder_units). The `multiplier` allows scaling (e.g., 0.5 for half the size).
- `max_ratio_of`: Ensures the parameter value doesn't exceed a specified ratio/fraction of another parameter value. The `ratio` parameter specifies the maximum fraction (e.g., 0.5 means the constrained parameter can be at most 50% of the reference parameter). This is useful for ensuring one parameter stays below another, such as `physics_loss.start_epoch` being at most half of `training.epochs`.

**Example with max_ratio_of:**
```json
{
  "training.epochs": {
    "type": "int",
    "low": 30,
    "high": 150,
    "step": 10
  },
  "physics_loss.start_epoch": {
    "type": "int",
    "low": 0,
    "high": 30,
    "step": 5,
    "constraint": {
      "type": "max_ratio_of",
      "parameter": "training.epochs",
      "ratio": 0.5
    }
  }
}
```

In this example, if `training.epochs` is sampled as 100, then `physics_loss.start_epoch` will be constrained to be at most 50 (100 * 0.5).

**Important:** When using constraints, the dependent parameter (e.g., `model.encoder_units` or `training.epochs`) must appear **before** the constrained parameter (e.g., `model.bottleneck.units` or `physics_loss.start_epoch`) in the parameters dictionary.

**Example HPO Config:**
```json
{
  "experiment_id": "hpo",
  "base_configs": {
    "simulation": "configs/simulation/default.json",
    "model": "configs/model/default.json"
  },
  "hyperparameter_optimization": {
    "enabled": true,
    "n_trials": 50,
    "metric": "val_loss",
    "direction": "minimize",
    "parameters": {
      "model.encoder_units": {
        "type": "categorical",
        "choices": [[32], [64], [64, 32], [128, 64]]
      },
      "model.bottleneck.units": {
        "type": "int",
        "low": 8,
        "high": 64,
        "step": 8,
        "constraint": {
          "type": "max_from_last",
          "parameter": "model.encoder_units"
        }
      },
      "training.epochs": {
        "type": "int",
        "low": 30,
        "high": 150,
        "step": 10
      },
      "training.lr_scheduler.patience": {
        "type": "int",
        "low": 3,
        "high": 15
      },
      "training.optimizer.learning_rate": {
        "type": "loguniform",
        "low": 1e-5,
        "high": 1e-2
      }
    }
  }
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

### Running Hyperparameter Optimization

```bash
# Run HPO with default config
python scripts/optimize.py

# Run HPO with custom config
python scripts/optimize.py --config configs/hpo/hpo_config.json

# Run HPO with specific simulation data
python scripts/optimize.py --sim-id default

# Specify number of trials (overrides config)
python scripts/optimize.py --n-trials 100
```

HPO results are saved to `experiments/hpo/` including:
- `best_config.json`: Best configuration found
- `study_summary.json`: Optimization summary and statistics
- Trial checkpoints and logs

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
