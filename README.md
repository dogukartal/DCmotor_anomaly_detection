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
│   ├── default.json                 # Master default configuration
│   └── experiments/                 # User experiment configs
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

Generate DC motor simulation data:

```bash
python scripts/simulate.py --config configs/default.json
```

### 2. Process Data

Downsample, extract features, and create windows:

```bash
python scripts/process_data.py \
  --config configs/default.json \
  --input data/raw/simulation_result.npy
```

### 3. Train Model

Train LSTM autoencoder:

```bash
python scripts/train.py \
  --config configs/default.json \
  --data data/processed/processed_data.npz
```

### 4. Evaluate Model

Evaluate on test set:

```bash
python scripts/evaluate.py \
  --experiment experiments/exp_001_<...>
```

### 5. Run Inference

Detect anomalies in new data:

```bash
python scripts/infer.py \
  --experiment experiments/exp_001_<...> \
  --input data/processed/new_data.npz
```

## Workflows

### Workflow A: Complete First-Time Pipeline

```bash
# 1. Create custom config (optional)
cp configs/default.json configs/experiments/my_experiment.json

# 2. Generate simulation data
python scripts/simulate.py --config configs/experiments/my_experiment.json

# 3. Process data
python scripts/process_data.py \
  --config configs/experiments/my_experiment.json \
  --input data/raw/simulation_result.npy

# 4. Train model
python scripts/train.py \
  --config configs/experiments/my_experiment.json \
  --data data/processed/processed_data.npz

# 5. Evaluate
python scripts/evaluate.py --experiment experiments/exp_001_<...>
```

### Workflow B: Hyperparameter Optimization

```bash
# 1. Enable HPO in config
# Set hyperparameter_optimization.enabled = true in your config

# 2. Run optimization
python scripts/optimize.py \
  --config configs/experiments/hpo_config.json \
  --data data/processed/processed_data.npz

# 3. Train with best config
python scripts/train.py \
  --config experiments/hpo_study/best_config.json \
  --data data/processed/processed_data.npz
```

## Configuration

The system is highly configurable through JSON files. Key configuration sections:

- **experiment**: Name, description, experiment tracking
- **input_generator**: Voltage signal generation (step, ramp, sinusoidal, etc.)
- **simulation**: DC motor parameters (R, L, Kt, Ke, J, B)
- **data_processing**: Downsampling, feature extraction, windowing
- **normalization**: Data normalization (minmax, standard, robust)
- **model**: LSTM architecture (encoder, decoder, bottleneck)
- **physics_loss**: Physics-informed loss settings
- **training**: Optimizer, learning rate, callbacks
- **hyperparameter_optimization**: Optuna settings

See `configs/default.json` for full configuration schema.

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

## Example Use Cases

1. **Baseline Detection**: Train on normal operation, detect bearing wear
2. **Parameter Variation**: Simulate different motor parameters (friction, inertia)
3. **Load Changes**: Detect abnormal load conditions
4. **Voltage Anomalies**: Identify irregular voltage inputs
5. **Multi-fault Detection**: Train on multiple failure modes

## Advanced Features

### Custom Voltage Signals

Generate custom voltage inputs programmatically:

```python
from src.simulation.input_generator import VoltageInputGenerator

gen = VoltageInputGenerator(sampling_rate=20000)
gen.add_signal('step', duration=0.5, amplitude=12.0)
gen.add_signal('chirp', duration=1.0, amplitude=10.0,
               start_frequency_hz=1.0, end_frequency_hz=50.0)
voltage = gen.generate()
```

### Custom Features

Add custom derived features in `src/data/processor.py`:

```python
elif feature_type == 'my_custom_feature':
    feature = # your calculation here
```

## Performance Tips

- Use GPU for training: TensorFlow will auto-detect
- Adjust `batch_size` based on available memory
- Use `window_stride < window_size` for overlapping windows
- Start with smaller `n_trials` for HPO, then increase

## Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Reduce `batch_size` or `window_size`

**Issue**: Physics loss causes instability
- **Solution**: Reduce `physics_loss.weight` or increase `start_epoch`

**Issue**: Poor reconstruction
- **Solution**: Increase model capacity (encoder/decoder units) or training epochs

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```
DC Motor Anomaly Detection System
Physics-Informed LSTM Autoencoder for Time Series Anomaly Detection
```
