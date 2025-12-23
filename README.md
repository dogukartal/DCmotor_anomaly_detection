# DC Motor Anomaly Detection

A comprehensive physics-informed LSTM autoencoder system for detecting anomalies in DC motor behavior through simulation and deep learning.

## Features

- **Physics-based DC motor simulation** with configurable parameters
- **LSTM Autoencoder** for sequence-to-sequence reconstruction
- **Physics-informed loss function** incorporating DC motor equations
- **Flexible data processing pipeline** with derived features
- **Hyperparameter optimization** using Optuna
- **Comprehensive visualization** tools
- **Modular and extensible** architecture

## Project Structure

```
dc_motor_anomaly/
├── configs/
│   ├── simulation/
│   │   └── default.json             # DC motor simulation parameters
│   ├── model/
│   │   └── default.json             # Model, training, and data processing config
│   └── hpo/
│       └── hpo_config.json          # Hyperparameter optimization config
│
├── src/
│   ├── simulation/                  # DC motor simulation
│   ├── data/                        # Data processing
│   ├── models/                      # LSTM autoencoder and loss
│   ├── training/                    # Training pipeline
│   ├── optimization/                # Hyperparameter optimization
│   ├── visualization/               # Plotting utilities
│   ├── inference/                   # Anomaly detection
│   └── utils/                       # Config and experiment management
│
├── scripts/
│   ├── simulate.py                  # Run DC motor simulation
│   ├── process_data.py              # Process raw simulation data
│   ├── train.py                     # Train LSTM autoencoder
│   ├── evaluate.py                  # Evaluate trained model
│   ├── optimize.py                  # Hyperparameter optimization
│   └── infer.py                     # Run inference on new data
│
├── data/                            # Data storage (generated)
├── experiments/                     # Experiment outputs (generated)
└── requirements.txt                 # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Simulation

Generate DC motor simulation data using the default simulation configuration:

```bash
python scripts/simulate.py --sim-config default
```

This creates `data/raw/default/simulation_result.npy` containing voltage, current, and angular velocity time series.

### 2. Process Data

Downsample, extract features, and create windows using the model configuration:

```bash
python scripts/process_data.py --model-config default
```

This creates `data/processed/default/processed_data.npz` with windowed sequences ready for training.

### 3. Train Model

Train the LSTM autoencoder with the model configuration:

```bash
python scripts/train.py --model-config default
```

This creates an experiment directory (e.g., `experiments/exp_001_<timestamp>_<params>`) containing:
- Trained model checkpoints (best and last)
- Training history and logs
- Visualization plots
- Frozen configuration file

### 4. Evaluate Model

Evaluate the trained model on the test set:

```bash
python scripts/evaluate.py --experiment experiments/exp_001_<timestamp>_<params>
```

This generates evaluation metrics and reconstruction visualizations in the experiment directory.

### 5. Run Inference

Detect anomalies in new simulation data:

```bash
python scripts/infer.py \
  --experiment experiments/exp_001_<timestamp>_<params> \
  --input data/processed/default/processed_data.npz
```

This outputs anomaly scores and detection results.

## Workflows

### Workflow A: Complete First-Time Pipeline

Run the complete pipeline from simulation to evaluation:

```bash
# Step 1: Generate DC motor simulation data
python scripts/simulate.py --sim-config default
# Output: data/raw/default/simulation_result.npy

# Step 2: Process raw data (downsample, extract features, create windows)
python scripts/process_data.py --model-config default
# Output: data/processed/default/processed_data.npz
# Automatically detects data from simulation ID "default"

# Step 3: Train LSTM autoencoder
python scripts/train.py --model-config default
# Output: experiments/exp_001_<timestamp>_<params>/
# Automatically uses data from simulation ID "default"

# Step 4: Evaluate trained model on test set
python scripts/evaluate.py --experiment experiments/exp_001_<timestamp>_<params>
# Output: results.json and evaluation plots in experiment directory

# Step 5: Run inference on new data (optional)
python scripts/infer.py \
  --experiment experiments/exp_001_<timestamp>_<params> \
  --input data/processed/default/processed_data.npz
# Output: anomaly scores and detection results
```

### Workflow B: Hyperparameter Optimization

Find optimal hyperparameters using Optuna:

```bash
# Step 1: Ensure you have processed data ready
# (Follow Workflow A steps 1-2 if not already done)

# Step 2: Run hyperparameter optimization
python scripts/optimize.py \
  --config configs/hpo/hpo_config.json \
  --data data/processed/default/processed_data.npz
# This runs multiple trials exploring the parameter space
# Output: experiments/hpo_study/

# Step 3: Review optimization results
cat experiments/hpo_study/study_summary.json
# Contains best parameters and trial history

# Step 4: Train final model with best configuration
python scripts/train.py \
  --model-config experiments/hpo_study/best_config.json
# Output: experiments/exp_002_<timestamp>_<params>/

# Step 5: Evaluate optimized model
python scripts/evaluate.py --experiment experiments/exp_002_<timestamp>_<params>
```

**HPO Configuration Details:**

The `configs/hpo/hpo_config.json` defines:
- **n_trials**: Number of optimization trials (default: 50)
- **metric**: Metric to optimize (e.g., "val_loss")
- **direction**: "minimize" or "maximize"
- **parameters**: Search space for each hyperparameter

Optimizable parameters include:
- Model architecture (encoder units, bottleneck size)
- Training parameters (learning rate, batch size)
- Data processing (window size)
- Physics loss (weight, start epoch)

### Workflow C: Custom Experiment Configuration

Create custom configurations for specific experiments:

```bash
# Step 1: Copy and edit base configurations
cp configs/simulation/default.json configs/simulation/my_experiment.json
cp configs/model/default.json configs/model/my_experiment.json
# Edit my_experiment.json files:
# - Set "simulation_id": "my_experiment" in both files
# - Modify simulation parameters (motor params, load conditions, etc.)
# - Modify model architecture, training settings, etc.

# Step 2: Run simulation with custom config
python scripts/simulate.py --sim-config configs/simulation/my_experiment.json
# Output: data/raw/my_experiment/simulation_result.npy

# Step 3: Process data with custom model config
python scripts/process_data.py --model-config configs/model/my_experiment.json
# Output: data/processed/my_experiment/processed_data.npz
# Automatically uses sim-id from model config

# Step 4: Train with custom configuration
python scripts/train.py --model-config configs/model/my_experiment.json
# Output: experiments/exp_003_<timestamp>_<params>/

# Step 5: Evaluate
python scripts/evaluate.py --experiment experiments/exp_003_<timestamp>_<params>

# Alternative: Use config IDs instead of full paths
# After saving configs as configs/simulation/my_experiment.json
python scripts/simulate.py --sim-config my_experiment
python scripts/process_data.py --model-config my_experiment
python scripts/train.py --model-config my_experiment
```

## Configuration

The system uses separate configuration files for different aspects:

### Simulation Configuration (`configs/simulation/default.json`)

Controls DC motor simulation parameters:
- **experiment**: Experiment name and tracking
- **input_generator**: Voltage signal generation (step, ramp, sinusoidal, chirp, noise)
- **simulation**: DC motor parameters (resistance, inductance, torque constant, back-EMF constant, inertia, friction)
- **solver**: ODE solver settings (method, tolerances)

### Model Configuration (`configs/model/default.json`)

Controls data processing, model architecture, and training:
- **experiment**: Name, description, auto-increment ID
- **data_processing**: Downsampling rate, derived features, windowing parameters, train/val/test split
- **normalization**: Method (minmax, standard, robust) and feature range
- **model**: LSTM architecture (encoder/decoder layers, bottleneck, dropout, activation functions)
- **physics_loss**: Physics-informed loss weight and schedule
- **training**: Optimizer, learning rate, batch size, epochs, callbacks (early stopping, LR scheduler)
- **plotting**: Visualization settings

### HPO Configuration (`configs/hpo/hpo_config.json`)

Controls hyperparameter optimization:
- **base_configs**: References to simulation and model configs to use as base
- **hyperparameter_optimization**: Optuna settings including:
  - **enabled**: Must be `true` for HPO
  - **n_trials**: Number of optimization trials
  - **metric**: Metric to optimize (e.g., "val_loss")
  - **direction**: "minimize" or "maximize"
  - **parameters**: Search space definitions for each hyperparameter

Each parameter can be:
- **categorical**: Discrete choices (e.g., `[32, 64, 128]`)
- **int**: Integer range with step (e.g., `low: 8, high: 64, step: 8`)
- **uniform**: Continuous range (e.g., `low: 0.01, high: 1.0`)
- **loguniform**: Log-scale continuous range (e.g., `low: 1e-5, high: 1e-2`)

## Key Components

### DC Motor Equations

The physics-informed loss enforces:

- **Electrical**: `V = R*i + L*(di/dt) + Ke*ω`
- **Mechanical**: `J*(dω/dt) = Kt*i - B*ω - T_load`

### Derived Features

Configurable features computed over downsampling windows:

- RMS (Root Mean Square)
- Peak-to-peak
- Variance
- Mean, Max, Min
- Slope (trend)
- Zero crossing rate

### LSTM Autoencoder

Architecture:
```
Input → [Encoder LSTMs] → [Bottleneck] → [Decoder LSTMs] → Output
```

Supports:
- Multiple encoder/decoder layers
- Optional bottleneck layer
- Dropout and recurrent dropout
- Custom activations

## Experiment Naming

Experiments are automatically named with the format:
```
exp_{id}_{date}_{encoder_units}_{lr}_{batch}_{epochs}_{name}
```

Example: `exp_001_20241215_64-32_lr0.001_b32_e100_friction_test`

## Output Files

Each experiment generates:

```
experiments/exp_XXX_<...>/
├── config.json                    # Frozen configuration
├── checkpoints/
│   ├── best_model/                # Best model (lowest val_loss)
│   └── last_model/                # Final model
├── plots/
│   ├── training_history.png
│   ├── reconstruction_*.png
│   └── error_distribution.png
├── logs/
│   └── training_log.json
├── training_history.json
├── normalizer_stats.json
└── results.json                   # Evaluation metrics
```

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```
DC Motor Anomaly Detection System
Physics-Informed LSTM Autoencoder for Time Series Anomaly Detection
```
