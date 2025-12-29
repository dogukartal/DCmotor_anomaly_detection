# Physics-Informed Neural Network Gradient Flow Fix

## Problem Summary

The Physics-Informed Neural Network (PINN) training was not working because **gradients from physics constraints could not reach the model weights** during backpropagation. The physics loss was being computed and added to the total loss, but it had **zero impact on model optimization**.

## Root Cause

**Location**: `src/models/physics_loss.py:110` and `src/models/physics_loss.py:175`

The `PhysicsInformedLoss` class used `.numpy()` to convert TensorFlow tensors to NumPy arrays for denormalization:

```python
# BROKEN CODE (before fix)
if self.normalizer is not None:
    y_pred_np = y_pred.numpy()  # ← BREAKS GRADIENT TAPE
    y_pred_denorm_np = self.normalizer.inverse_transform(y_pred_np)
    y_pred_physical = tf.convert_to_tensor(y_pred_denorm_np, dtype=tf.float32)
```

### Why This Broke Training

1. **Forward pass**: Model predictions `y_pred` are tracked by TensorFlow's gradient tape ✓
2. **Denormalization**: `.numpy()` converts tensor to NumPy array → **gradient tape connection severed** ✗
3. **NumPy operations**: `normalizer.inverse_transform()` uses sklearn/NumPy → **no gradient tracking** ✗
4. **New tensor**: `tf.convert_to_tensor()` creates a brand new tensor → **disconnected from model** ✗
5. **Physics loss**: Computed using the new tensor → **gradients exist but don't reach model weights** ✗
6. **Backpropagation**: Gradients stop at the new tensor boundary → **only reconstruction loss trains the model** ✗

## The Fix

### Implementation

Added a new method `_denormalize_tf()` that performs denormalization using **pure TensorFlow operations**:

```python
def _denormalize_tf(self, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Denormalize predictions using TensorFlow operations to maintain gradient flow.

    CRITICAL: Uses pure TensorFlow ops instead of NumPy to ensure gradients
    can flow through denormalization during backpropagation.
    """
    if self.normalizer is None:
        return y_pred

    stats = self.normalizer.get_statistics()
    method = stats['method']

    if method == 'minmax':
        # MinMax inverse transform using TensorFlow
        data_min = tf.constant(stats['min'], dtype=tf.float32)
        data_range = tf.constant(stats['data_range'], dtype=tf.float32)
        feature_min, feature_max = stats['feature_range']

        y_pred_physical = (y_pred - feature_min) / (feature_max - feature_min) * data_range + data_min

    elif method == 'standard':
        # Standard inverse transform using TensorFlow
        mean = tf.constant(stats['mean'], dtype=tf.float32)
        std = tf.constant(stats['std'], dtype=tf.float32)
        y_pred_physical = y_pred * std + mean

    elif method == 'robust':
        # Robust inverse transform using TensorFlow
        center = tf.constant(stats['center'], dtype=tf.float32)
        scale = tf.constant(stats['scale'], dtype=tf.float32)
        y_pred_physical = y_pred * scale + center

    return y_pred_physical
```

### Changes Made

1. **Added** `_denormalize_tf()` method (line 96-139)
2. **Updated** `compute_physics_loss()` to use `_denormalize_tf()` (line 154)
3. **Updated** `compute_loss_components()` to use `_denormalize_tf()` (line 212)

## Gradient Flow After Fix

```
Model → y_pred → _denormalize_tf() (TF ops) → physics_loss → total_loss → Gradients → Update weights
        ↑                                                                              ↓
        └──────────────────────────── Backpropagation ──────────────────────────────────┘
                              (continuous gradient path) ✓
```

Now gradients flow smoothly from physics constraints all the way back to model weights!

## Impact

### Before Fix
- Physics loss was computed but **ignored during optimization**
- Model trained as a **regular autoencoder** (reconstruction loss only)
- Physics constraints had **zero effect** on learned representations
- Training was equivalent to `physics_weight = 0.0`

### After Fix
- Physics loss **actively influences** model training
- Gradients from electrical and mechanical equations **update model weights**
- Model learns representations that **respect DC motor physics**
- Physics weight parameter **actually works** as intended

## Verification

Run the gradient flow test:

```bash
python scripts/test_gradient_flow_fix.py
```

This test verifies:
1. ✓ Gradients flow through physics loss with normalizer
2. ✓ Physics weight affects gradients
3. ✓ Physics loss components are computed correctly

## Technical Details

### Why TensorFlow Operations?

TensorFlow tracks computational operations to build a **computational graph**. This graph enables:
- **Automatic differentiation**: Compute gradients of any operation
- **Backpropagation**: Chain rule applied automatically through the graph

When you use `.numpy()`:
- The computational graph is **severed**
- New tensors created with `tf.convert_to_tensor()` are **orphaned**
- Gradients cannot traverse the NumPy boundary

### Supported Normalization Methods

The fix supports all three normalization methods used by the `Normalizer` class:

1. **MinMax Scaling** (`-1` to `1` or custom range)
   - Inverse: `x = (x_scaled - min_range) / (max_range - min_range) * data_range + data_min`

2. **Standard Scaling** (z-score normalization)
   - Inverse: `x = x_scaled * std + mean`

3. **Robust Scaling** (median and IQR)
   - Inverse: `x = x_scaled * scale + center`

## Files Modified

- `src/models/physics_loss.py`: Core fix implementation
- `scripts/test_gradient_flow_fix.py`: Verification test (new)
- `GRADIENT_FLOW_FIX.md`: This documentation (new)

## Related Issues

This fix resolves the fundamental issue where physics-informed training appeared to be set up correctly but had no actual effect on model optimization. The bug was subtle because:

1. ✓ `PhysicsInformedLoss` was correctly passed to `model.compile()`
2. ✓ Loss values included physics components in the logs
3. ✓ Callbacks tracked physics loss correctly
4. ✗ **But gradients couldn't flow back to the model**

The issue was discovered through detailed gradient flow analysis showing that physics weight changes had no effect on model gradients.

## Future Considerations

### Performance Optimization

The current implementation recreates `tf.constant()` tensors on every forward pass. For better performance, consider:

1. **Cache constants** in `__init__()` after normalizer is set
2. **Reuse tensors** across forward passes
3. **Profile** to measure actual overhead

Example optimization:

```python
def __init__(self, ...):
    # ... existing code ...
    self._cached_denorm_params = None

def _cache_denormalization_params(self):
    """Cache denormalization parameters as TensorFlow constants."""
    if self.normalizer is None or self._cached_denorm_params is not None:
        return

    stats = self.normalizer.get_statistics()
    # Create cached TF constants
    # ...
```

### Alternative Approaches

Other potential solutions (not implemented):

1. **Custom TensorFlow normalizer layer**: Subclass `tf.keras.layers.Layer`
2. **Preprocessing layers**: Use `tf.keras.layers.Normalization`
3. **Graph mode execution**: Use `@tf.function` decoration

The current fix was chosen for:
- **Minimal code changes**
- **Backward compatibility**
- **Clear separation of concerns**

## References

- TensorFlow Automatic Differentiation: https://www.tensorflow.org/guide/autodiff
- GradientTape: https://www.tensorflow.org/api_docs/python/tf/GradientTape
- Physics-Informed Neural Networks: Raissi et al. (2019)
