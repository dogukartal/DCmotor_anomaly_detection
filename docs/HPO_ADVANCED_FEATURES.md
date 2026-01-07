# Advanced HPO Features

This document describes the advanced hyperparameter optimization features for layer architecture optimization.

## Overview

The HPO system now supports:
1. **Dynamic network depth** - Sample the number of layers (2 or 3)
2. **Integer-based layer sizes with gain constraints** - Ensure each layer follows architectural constraints
3. **Early stopping patience with intelligent constraints** - Optimize early stopping relative to LR scheduler patience
4. **Advanced constraint types** - `max_from_last`, `max_ratio_of`, and `min_ratio_of` for parameter dependencies
5. **Explicit TPE sampler** - Confirmed use of Tree-structured Parzen Estimator

## Features

### 1. Dynamic Network Depth with Layer Sequence

The `layer_sequence` parameter type allows sampling both the depth (number of layers) and the size of each layer with architectural constraints.

**Configuration Example:**
```json
{
  "model.encoder_units": {
    "type": "layer_sequence",
    "depth_choices": [2, 3],
    "low": 16,
    "high": 128,
    "step": 16,
    "gain": 0.5,
    "comment": "Dynamic encoder layers"
  }
}
```

**Parameters:**
- `depth_choices`: List of possible depths (e.g., [2, 3] means 2 or 3 layers)
- `low`: Minimum layer size
- `high`: Maximum layer size
- `step`: Step size for layer sampling (e.g., 16 means layers are multiples of 16)
- `gain`: Maximum ratio for subsequent layers (e.g., 0.5 means each layer ≤ previous * 0.5)

**How it works:**
- First layer: Sampled from full range [low, high] with step
- Subsequent layers: Constrained to be ≤ previous_layer * gain
- Ensures monotonically decreasing layer sizes for encoder

**Examples:**

With `low=16, high=128, step=16, gain=0.5`:

✓ **Valid configurations:**
- `[64, 32]` - ratio: 32/64 = 0.50
- `[96, 32]` - ratio: 32/96 = 0.33
- `[128, 64]` - ratio: 64/128 = 0.50
- `[48, 16, 8]` - ratios: 16/48 = 0.33, 8/16 = 0.50

✗ **Invalid configurations (will NOT be sampled):**
- `[64, 48]` - ratio: 48/64 = 0.75 > 0.5
- `[96, 128]` - wrong direction (increasing)
- `[128, 96]` - ratio: 96/128 = 0.75 > 0.5

### 2. Symmetric Decoder Layers

Decoder layers automatically mirror the encoder in reverse order for architectural symmetry.

**Configuration Example:**
```json
{
  "model.decoder_units": {
    "type": "layer_sequence",
    "mirror_from": "model.encoder_units",
    "comment": "Decoder mirrors encoder"
  }
}
```

**Example:**
- If encoder: `[96, 32, 16]`
- Then decoder: `[16, 32, 96]`

### 3. Early Stopping Patience with Constraint

Early stopping patience can now be optimized as a hyperparameter with intelligent constraints relative to the learning rate scheduler patience.

**Configuration Example:**
```json
{
  "training.lr_scheduler.patience": {
    "type": "int",
    "low": 3,
    "high": 15,
    "step": 1
  },
  "training.early_stopping.patience": {
    "type": "int",
    "low": 10,
    "high": 30,
    "step": 5,
    "constraint": {
      "type": "min_ratio_of",
      "parameter": "training.lr_scheduler.patience",
      "ratio": 2.0
    }
  }
}
```

**Why this constraint matters:**
- The LR scheduler should reduce the learning rate multiple times before early stopping
- Setting `early_stopping.patience >= lr_scheduler.patience * 2` ensures at least 2 LR reductions
- Example: if lr_scheduler.patience = 5, early_stopping.patience will be >= 10

This helps prevent overfitting while allowing the optimizer to explore different learning rates.

### 4. TPE Sampler

The HPO system explicitly uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for efficient hyperparameter optimization. TPE is a Bayesian optimization algorithm that is particularly effective for hyperparameter tuning.

## Complete Configuration Example

See `configs/hpo/hpo_config.json` for a complete working example with all features:

```json
{
  "hyperparameter_optimization": {
    "enabled": true,
    "n_trials": 50,
    "metric": "val_loss",
    "direction": "minimize",
    "parameters": {
      "model.encoder_units": {
        "type": "layer_sequence",
        "depth_choices": [2, 3],
        "low": 16,
        "high": 128,
        "step": 16,
        "gain": 0.5
      },
      "model.decoder_units": {
        "type": "layer_sequence",
        "mirror_from": "model.encoder_units"
      },
      "training.early_stopping.patience": {
        "type": "int",
        "low": 10,
        "high": 30,
        "step": 5
      }
    }
  }
}
```

## Usage

Run HPO with the new features:

```bash
python scripts/optimize.py \
  --config configs/hpo/hpo_config.json \
  --data data/processed/windows.npz \
  --output experiments/hpo_study
```

## Testing

Test the layer sequence sampling logic:

```bash
python scripts/test_hpo_sampling_unit.py
```

This will verify that:
- All sampled configurations satisfy the gain constraints
- Decoder layers are symmetric to encoder layers
- Depth is sampled correctly from choices
- Layer sizes respect min/max/step constraints

## Constraint Types

The HPO system supports multiple constraint types for parameter dependencies:

### 1. `max_from_last`
Constrains a parameter to be ≤ the last value from a list parameter.

**Example:** Bottleneck units ≤ last encoder layer
```json
{
  "model.bottleneck.units": {
    "constraint": {
      "type": "max_from_last",
      "parameter": "model.encoder_units",
      "multiplier": 1.0
    }
  }
}
```

### 2. `max_ratio_of`
Constrains a parameter to be ≤ reference_parameter × ratio.

**Example:** Physics loss start epoch ≤ 50% of total epochs
```json
{
  "physics_loss.start_epoch": {
    "constraint": {
      "type": "max_ratio_of",
      "parameter": "training.epochs",
      "ratio": 0.5
    }
  }
}
```

### 3. `min_ratio_of`
Constrains a parameter to be ≥ reference_parameter × ratio.

**Example:** Early stopping patience ≥ 2 × LR scheduler patience
```json
{
  "training.early_stopping.patience": {
    "constraint": {
      "type": "min_ratio_of",
      "parameter": "training.lr_scheduler.patience",
      "ratio": 2.0
    }
  }
}
```

## Implementation Details

### Layer Sequence Constraint Enforcement

The layer sequence sampler enforces constraints by:

1. **First layer**: Full range [low, high] with step
2. **Subsequent layers**:
   - Calculate: `constrained_max = int(previous_layer * gain)`
   - Round down to nearest valid step
   - If constrained_max < step, use raw constrained value
   - Otherwise, sample from [low, constrained_max] with step

This ensures that:
- Each layer is ≤ previous_layer * gain
- Layer sizes are valid (multiples of step when possible)
- No invalid configurations are sampled

### Parameter Order

**Important**: Ensure parameter order in config respects dependencies:

1. `model.encoder_units` (sampled first)
2. `model.decoder_units` (depends on encoder)
3. `model.bottleneck.units` (constrained by last encoder layer)

The HPO system processes parameters in order, so dependencies must be defined before they are referenced.

## Benefits

These advanced features provide:

1. **Better architecture exploration**: Systematically explore different network depths
2. **Architectural constraints**: Ensure sampled architectures follow best practices
3. **Reduced search space**: Eliminate invalid configurations
4. **Symmetric design**: Maintain encoder-decoder symmetry automatically
5. **Overfitting control**: Optimize early stopping for your specific dataset
6. **Efficient optimization**: TPE sampler focuses on promising regions of hyperparameter space
