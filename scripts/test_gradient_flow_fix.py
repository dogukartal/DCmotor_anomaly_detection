#!/usr/bin/env python3
"""
Test to verify gradient flow fix for PhysicsInformedLoss.

This script tests that:
1. Gradients flow through physics loss with TensorFlow-native denormalization
2. Physics weight affects gradients
3. The fix resolves the broken gradient path
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.physics_loss import PhysicsInformedLoss
from src.data.normalizer import Normalizer


def test_gradient_flow_with_normalizer():
    """Test gradient flow when using normalizer (the critical case)."""
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST WITH NORMALIZER")
    print("="*70)

    # Create synthetic normalized data
    np.random.seed(42)
    batch_size, seq_len, n_features = 4, 10, 3

    # Create normalized data (range -1 to 1, as would come from MinMaxScaler)
    y_data_normalized = np.random.uniform(-1, 1, (batch_size, seq_len, n_features)).astype(np.float32)
    y_true = tf.constant(y_data_normalized)

    # Create and fit normalizer with synthetic original data
    # Original data: current [1-3A], omega [50-150 rad/s], voltage [10-24V]
    original_data = np.random.uniform(0, 1, (100, seq_len, n_features))
    original_data[:, :, 0] = np.random.uniform(1.0, 3.0, (100, seq_len))  # current
    original_data[:, :, 1] = np.random.uniform(50, 150, (100, seq_len))  # omega
    original_data[:, :, 2] = np.random.uniform(10, 24, (100, seq_len))   # voltage

    normalizer = Normalizer(method='minmax', feature_range=(-1, 1))
    normalizer.fit(original_data)

    print(f"\n✓ Created normalizer with stats:")
    stats = normalizer.get_statistics()
    print(f"  Method: {stats['method']}")
    print(f"  Feature range: {stats['feature_range']}")
    print(f"  Data min: {stats['min']}")
    print(f"  Data max: {stats['max']}")

    # Motor parameters
    motor_params = {
        'R': 1.0, 'L': 0.5, 'Kt': 0.01,
        'Ke': 0.01, 'J': 0.01, 'B': 0.1
    }

    # Create simple model
    input_shape = (seq_len, n_features)
    model = LSTMAutoencoder(
        input_shape=input_shape,
        encoder_units=[8],
        decoder_units=[8],
        dropout=0.0
    )
    model.build()

    print("\n" + "-"*70)
    print("Test 1: Physics weight = 0.0 (reconstruction only)")
    print("-"*70)

    physics_loss_zero = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=0.0,
        enabled=True,
        normalizer=normalizer,  # ← WITH normalizer
        dt=0.002
    )
    physics_loss_zero.set_epoch(100)

    with tf.GradientTape() as tape:
        y_pred = model.model(y_true, training=True)
        loss_zero = physics_loss_zero(y_true, y_pred)

    grads_zero = tape.gradient(loss_zero, model.model.trainable_variables)
    grad_norm_zero = tf.norm(grads_zero[0]).numpy()

    print(f"  Loss: {loss_zero.numpy():.6f}")
    print(f"  First layer gradient norm: {grad_norm_zero:.6f}")

    print("\n" + "-"*70)
    print("Test 2: Physics weight = 1.0 (reconstruction + physics)")
    print("-"*70)

    physics_loss_one = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=1.0,
        enabled=True,
        normalizer=normalizer,  # ← WITH normalizer
        dt=0.002
    )
    physics_loss_one.set_epoch(100)

    with tf.GradientTape() as tape:
        y_pred = model.model(y_true, training=True)
        loss_one = physics_loss_one(y_true, y_pred)

    grads_one = tape.gradient(loss_one, model.model.trainable_variables)

    # Check if gradients exist
    if grads_one[0] is None:
        print("  ✗ FAIL: Gradients are None!")
        print("  → Physics loss is NOT connected to the model!")
        return False

    grad_norm_one = tf.norm(grads_one[0]).numpy()

    print(f"  Loss: {loss_one.numpy():.6f}")
    print(f"  First layer gradient norm: {grad_norm_one:.6f}")

    print("\n" + "-"*70)
    print("Test 3: Gradient comparison")
    print("-"*70)

    gradient_diff = abs(grad_norm_one - grad_norm_zero)
    relative_diff = gradient_diff / (grad_norm_zero + 1e-10)

    print(f"  Gradient norm difference: {gradient_diff:.6f}")
    print(f"  Relative difference: {relative_diff:.2%}")

    print("\n" + "="*70)
    if relative_diff > 0.01:  # More than 1% difference
        print("✓ SUCCESS: Gradients flow correctly through physics loss!")
        print(f"  Physics weight change → {relative_diff:.1%} gradient change")
        print("\n  → TensorFlow-native denormalization maintains gradient flow ✓")
        return True
    else:
        print("✗ FAIL: Gradients don't change with physics weight")
        print("  → Physics loss is still not affecting gradients")
        return False


def test_physics_loss_computation():
    """Test that physics loss is still computed correctly."""
    print("\n" + "="*70)
    print("PHYSICS LOSS COMPUTATION TEST")
    print("="*70)

    np.random.seed(42)
    batch_size, seq_len, n_features = 2, 20, 3

    # Create normalized data
    y_data_normalized = np.random.uniform(-1, 1, (batch_size, seq_len, n_features)).astype(np.float32)
    y_true = tf.constant(y_data_normalized)
    y_pred = tf.constant(y_data_normalized)

    # Create normalizer
    original_data = np.random.uniform(0, 1, (100, seq_len, n_features))
    original_data[:, :, 0] = np.random.uniform(1.0, 3.0, (100, seq_len))
    original_data[:, :, 1] = np.random.uniform(50, 150, (100, seq_len))
    original_data[:, :, 2] = np.random.uniform(10, 24, (100, seq_len))

    normalizer = Normalizer(method='minmax', feature_range=(-1, 1))
    normalizer.fit(original_data)

    motor_params = {
        'R': 1.0, 'L': 0.5, 'Kt': 0.01,
        'Ke': 0.01, 'J': 0.01, 'B': 0.1
    }

    physics_loss = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=0.1,
        enabled=True,
        normalizer=normalizer,
        dt=0.002
    )
    physics_loss.set_epoch(100)

    print("\nComputing loss components...")
    components = physics_loss.compute_loss_components(y_true, y_pred)

    print(f"  Reconstruction: {components['reconstruction'].numpy():.6f}")
    print(f"  Electrical:     {components['electrical'].numpy():.6f}")
    print(f"  Mechanical:     {components['mechanical'].numpy():.6f}")
    print(f"  Physics total:  {components['physics'].numpy():.6f}")

    has_physics = components['physics'].numpy() > 1e-6
    has_electrical = components['electrical'].numpy() > 1e-6
    has_mechanical = components['mechanical'].numpy() > 1e-6

    print("\n" + "="*70)
    if has_physics and has_electrical and has_mechanical:
        print("✓ SUCCESS: Physics loss components computed correctly")
        return True
    else:
        print("✗ FAIL: Some physics components are zero")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  GRADIENT FLOW FIX VERIFICATION" + " "*36 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    results = {}

    try:
        results['gradient_flow'] = test_gradient_flow_with_normalizer()
    except Exception as e:
        print(f"\n✗ GRADIENT FLOW TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        results['gradient_flow'] = False

    try:
        results['physics_computation'] = test_physics_loss_computation()
    except Exception as e:
        print(f"\n✗ PHYSICS COMPUTATION TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        results['physics_computation'] = False

    # Summary
    print("\n" + "#"*70)
    print("#  SUMMARY" + " "*60 + "#")
    print("#"*70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTests passed: {passed_tests}/{total_tests}\n")

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")

    print("\n" + "="*70)
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe gradient flow fix is working correctly:")
        print("  ✓ TensorFlow-native denormalization maintains gradient flow")
        print("  ✓ Physics constraints now affect model training")
        print("  ✓ Gradients flow from physics loss to model weights")
    else:
        print("\n⚠ SOME TESTS FAILED")
        print("Please review the output above for details.")

    print("="*70 + "\n")

    return passed_tests == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
