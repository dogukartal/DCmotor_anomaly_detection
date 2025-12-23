# Architecture Overview

This document explains the microservices-style architecture of the DC Motor Anomaly Detection system.

## Design Philosophy

The system follows **microservices principles**:
- **Single Responsibility**: Each component does one thing well
- **Loose Coupling**: Minimal dependencies between services
- **Clear Contracts**: Well-defined inputs and outputs
- **No Optimization of Baked-In Parameters**: Services don't optimize parameters that are already embedded in their inputs

## Service Pipeline

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────┐
│   simulate.py   │ ───> │   process.py     │ ───> │  train.py   │
│  (Raw Data)     │      │ (Processed Data) │      │  (Model)    │
└─────────────────┘      └──────────────────┘      └─────────────┘
                                  │
                                  ↓
                         ┌─────────────────┐
                         │  optimize.py    │
                         │ (HPO on Model)  │
                         └─────────────────┘
```

### Service 1: Data Generation (`simulate.py`)
- **Input**: Simulation configuration
- **Output**: Raw time series data (voltage, current, angular velocity)
- **Parameters**: Motor parameters, input signals, sampling rate
- **Responsibility**: Generate physically accurate DC motor simulation data

### Service 2: Data Processing (`process.py`)
- **Input**: Raw simulation data + Model configuration
- **Output**: Processed, windowed, normalized data ready for training
- **Parameters**: Window size, stride, downsampling rate, features
- **Responsibility**: Transform raw data into ML-ready format
- **Key Point**: Once processed, window_size is "baked in" - can't be changed

### Service 3: Model Training (`train.py`)
- **Input**: Processed data + Model configuration
- **Output**: Trained model with checkpoints and metrics
- **Parameters**: Model architecture, training hyperparameters, physics loss
- **Responsibility**: Train LSTM autoencoder on processed data

### Service 4: Hyperparameter Optimization (`optimize.py`)
- **Input**: Processed data + HPO configuration (which references base configs)
- **Output**: Best configuration and study results
- **Parameters**: Model architecture, training hyperparameters (NOT data processing)
- **Responsibility**: Find optimal model and training parameters
- **Key Constraint**: Can only optimize parameters that don't require re-processing data

## Configuration Architecture

### Three-Layer Config System

```
┌──────────────────────────┐
│  HPO Config              │  Layer 3: Optimization
│  configs/hpo/*.json      │
│                          │
│  References:             │
│  ├─> simulation config   │
│  └─> model config        │
└──────────────────────────┘
         │
         │ (merged at runtime)
         ↓
┌──────────────────────────┐
│  Simulation Config       │  Layer 1: Physics
│  configs/simulation/     │
│  - Motor parameters      │
│  - Input signals         │
└──────────────────────────┘

┌──────────────────────────┐
│  Model Config            │  Layer 2: ML Pipeline
│  configs/model/          │
│  - Data processing       │
│  - Model architecture    │
│  - Training params       │
└──────────────────────────┘
```

### Config Responsibilities

**Simulation Config** (`configs/simulation/default.json`)
- Defines physical system parameters
- Used by: `simulate.py`
- Independent of ML pipeline

**Model Config** (`configs/model/default.json`)
- References a simulation_id
- Defines data processing AND model parameters
- Used by: `process.py`, `train.py`, `evaluate.py`
- Contains ALL parameters needed for the ML pipeline

**HPO Config** (`configs/hpo/hpo_config.json`)
- References both simulation and model base configs
- Defines search space for optimization
- Used by: `optimize.py`
- Automatically merges base configs at runtime via `ConfigManager.load_hpo_config()`

## The Circular Dependency Problem (SOLVED)

### ❌ Original Problem

HPO configuration included `window_size` as an optimizable parameter:

```json
{
  "parameters": {
    "data_processing.window_size": {
      "type": "categorical",
      "choices": [32, 64, 128, 256]
    }
  }
}
```

But `optimize.py` expected pre-processed data:

```bash
python scripts/optimize.py \
  --config configs/hpo/hpo_config.json \
  --data data/processed/default/processed_data.npz  # Already has fixed window_size!
```

**This created a circular dependency:**
1. HPO wants to try window_size=64
2. But data is already windowed at window_size=128
3. To change window_size, we'd need to re-process data
4. But data processing is a separate service!

### ✅ Solution

**Principle**: *Don't optimize what's already baked into your inputs*

1. **Removed `window_size` from HPO parameters**
   - HPO now only optimizes model and training parameters
   - Data processing parameters are fixed before HPO

2. **If you want to optimize `window_size`:**
   - Run multiple experiments with different window sizes
   - Compare results across experiments
   - This is a separate, outer optimization loop

3. **Clear service boundaries:**
   ```
   Data Processing (window_size fixed) → HPO (model params only)
   ```

## Config Merging Strategy

### Before (BROKEN)

```python
config = ConfigManager.load('configs/hpo/hpo_config.json')
# Result: Only HPO config loaded, no base configs merged
# Error: KeyError: 'data_processing'
```

### After (WORKING)

```python
config = ConfigManager.load_hpo_config('configs/hpo/hpo_config.json')
# Result: Simulation + Model + HPO configs all merged
# Contains: simulation, data_processing, model, training, physics_loss, hyperparameter_optimization
```

The new `load_hpo_config()` method:
1. Loads HPO config
2. Reads `base_configs` section
3. Loads and validates simulation config
4. Loads and validates model config
5. Merges simulation + model configs
6. Adds HPO-specific sections
7. Returns complete, ready-to-use config

## Best Practices

### 1. Service Independence
Each script should be runnable independently with just its config:
```bash
# ✅ Good - clear dependencies
python scripts/simulate.py --sim-config default
python scripts/process_data.py --model-config default
python scripts/train.py --model-config default

# ❌ Bad - tight coupling
python scripts/train.py --sim-config X --model-config Y --data Z
```

### 2. Data as Contract
Once data is processed, it defines a contract:
- Window size is fixed
- Features are fixed
- Sampling rate is fixed
- Downstream services must accept this contract

### 3. Optimization Boundaries
Optimize at the right level:
- **Data processing level**: window_size, stride, features
- **Model level**: architecture, learning rate, batch size
- Don't mix levels in a single optimization run

### 4. Config Inheritance
Use the inheritance hierarchy:
```
HPO config (specific)
  ↓ inherits from
Model config (moderate)
  ↓ references
Simulation config (foundational)
```

## Example: Optimizing window_size

If you want to find the optimal window_size:

```bash
# Option 1: Manual grid search
for ws in 32 64 128 256; do
  # Update config
  sed -i "s/\"window_size\": [0-9]*/\"window_size\": $ws/" configs/model/default.json

  # Re-process data
  python scripts/process_data.py --model-config default

  # Train model
  python scripts/train.py --model-config default

  # Evaluate
  python scripts/evaluate.py --experiment experiments/exp_XXX_...
done

# Compare results across experiments
```

**Note**: This is intentionally separate from HPO because:
1. Data processing is expensive
2. Creates clear separation of concerns
3. Allows parallel experiments with different window sizes
4. Each experiment has its own processed data and models

## Summary

**Key Takeaways:**
1. ✅ Each service has clear input/output contracts
2. ✅ No circular dependencies
3. ✅ Config merging is automatic and correct
4. ✅ HPO optimizes only model/training parameters
5. ✅ Data processing parameters are fixed before HPO
6. ✅ If you need to optimize data processing, do it separately

This architecture ensures:
- **Maintainability**: Easy to understand and modify
- **Testability**: Each service can be tested independently
- **Scalability**: Can parallelize experiments easily
- **Correctness**: No circular dependencies or config conflicts
