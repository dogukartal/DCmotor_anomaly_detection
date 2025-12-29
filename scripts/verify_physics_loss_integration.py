#!/usr/bin/env python3
"""
Verification script to test if physics loss is properly integrated into gradient computation.

This script performs three critical tests:
1. Gradient flow test: Verify if physics loss affects model gradients
2. Loss component test: Check if physics loss is being computed correctly
3. Training impact test: Verify if physics loss weight changes affect training
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.physics_loss import PhysicsInformedLoss
from src.data.normalizer import Normalizer


def test_gradient_flow():
    """
    Test 1: Verify if physics loss affects gradients.

    This test creates a simple model and checks if changing the physics_weight
    parameter changes the gradients. If physics loss is properly integrated,
    gradients should be different with different weights.
    """
    print("\n" + "="*70)
    print("TEST 1: GRADIENT FLOW TEST")
    print("="*70)
    print("Testing if physics loss affects model gradients...")

    # Create simple synthetic data
    batch_size, seq_len, n_features = 4, 10, 3

    # Create realistic motor data (current, angular_velocity, voltage)
    np.random.seed(42)
    current = np.random.uniform(0.5, 2.0, (batch_size, seq_len, 1))
    omega = np.random.uniform(50, 150, (batch_size, seq_len, 1))
    voltage = np.random.uniform(10, 24, (batch_size, seq_len, 1))

    y_true = np.concatenate([current, omega, voltage], axis=-1).astype(np.float32)

    # Create model
    input_shape = (seq_len, n_features)
    model = LSTMAutoencoder(
        input_shape=input_shape,
        encoder_units=[8],
        decoder_units=[8],
        bottleneck_units=None,
        dropout=0.0
    )
    model.build()

    # Motor parameters
    motor_params = {
        'R': 1.0,
        'L': 0.5,
        'Kt': 0.01,
        'Ke': 0.01,
        'J': 0.01,
        'B': 0.1
    }

    # Test with physics weight = 0.0 (no physics loss)
    print("\n1. Computing gradients with physics_weight=0.0...")
    physics_loss_off = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=0.0,
        enabled=True,
        dt=0.002
    )
    physics_loss_off.set_epoch(100)  # Past start epoch

    with tf.GradientTape() as tape:
        y_pred = model.model(y_true, training=True)
        loss_off = physics_loss_off(y_true, y_pred)

    gradients_off = tape.gradient(loss_off, model.model.trainable_variables)
    grad_norms_off = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients_off]

    print(f"   Loss value: {loss_off.numpy():.6f}")
    print(f"   First layer gradient norm: {grad_norms_off[0]:.6f}")

    # Test with physics weight = 1.0 (full physics loss)
    print("\n2. Computing gradients with physics_weight=1.0...")
    physics_loss_on = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=1.0,
        enabled=True,
        dt=0.002
    )
    physics_loss_on.set_epoch(100)

    with tf.GradientTape() as tape:
        y_pred = model.model(y_true, training=True)
        loss_on = physics_loss_on(y_true, y_pred)

    gradients_on = tape.gradient(loss_on, model.model.trainable_variables)
    grad_norms_on = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients_on]

    print(f"   Loss value: {loss_on.numpy():.6f}")
    print(f"   First layer gradient norm: {grad_norms_on[0]:.6f}")

    # Compare gradients
    print("\n3. Comparing gradients...")
    gradient_diff = abs(grad_norms_on[0] - grad_norms_off[0])
    relative_diff = gradient_diff / (grad_norms_off[0] + 1e-10)

    print(f"   Gradient difference: {gradient_diff:.6f}")
    print(f"   Relative difference: {relative_diff:.2%}")

    # Verdict
    print("\n" + "-"*70)
    if relative_diff > 0.01:  # More than 1% difference
        print("✓ PASS: Physics loss DOES affect gradients")
        print(f"  Gradients changed by {relative_diff:.1%} when physics weight changed")
        return True
    else:
        print("✗ FAIL: Physics loss DOES NOT affect gradients")
        print("  Gradients are identical regardless of physics weight!")
        print("  → Physics loss is NOT integrated into gradient computation")
        return False


def test_loss_components():
    """
    Test 2: Verify physics loss components are computed correctly.

    This test checks if the physics loss equations produce sensible values
    and if the electrical and mechanical residuals are non-zero when physics
    constraints are violated.
    """
    print("\n" + "="*70)
    print("TEST 2: LOSS COMPONENT TEST")
    print("="*70)
    print("Testing if physics loss components are computed correctly...")

    # Create data that violates physics
    batch_size, seq_len, n_features = 2, 20, 3
    np.random.seed(42)

    # Create data that severely violates motor equations
    current = np.random.uniform(1.0, 3.0, (batch_size, seq_len, 1))
    omega = np.random.uniform(100, 200, (batch_size, seq_len, 1))
    voltage = np.random.uniform(5, 10, (batch_size, seq_len, 1))  # Too low voltage!

    y_data = np.concatenate([current, omega, voltage], axis=-1).astype(np.float32)

    motor_params = {
        'R': 1.0,
        'L': 0.5,
        'Kt': 0.01,
        'Ke': 0.01,
        'J': 0.01,
        'B': 0.1
    }

    physics_loss = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=0.1,
        enabled=True,
        dt=0.002
    )
    physics_loss.set_epoch(100)

    # Compute loss components
    y_true = tf.constant(y_data)
    y_pred = tf.constant(y_data)  # Using same data for simplicity

    components = physics_loss.compute_loss_components(y_true, y_pred)

    print("\n1. Loss component values:")
    print(f"   Reconstruction loss: {components['reconstruction'].numpy():.6f}")
    print(f"   Electrical loss:     {components['electrical'].numpy():.6f}")
    print(f"   Mechanical loss:     {components['mechanical'].numpy():.6f}")
    print(f"   Total physics loss:  {components['physics'].numpy():.6f}")

    # Verify physics loss is non-zero
    physics_total = components['physics'].numpy()
    electrical = components['electrical'].numpy()
    mechanical = components['mechanical'].numpy()

    print("\n2. Checking physics constraints...")
    all_pass = True

    if physics_total > 1e-6:
        print(f"   ✓ Physics loss is non-zero: {physics_total:.6f}")
    else:
        print(f"   ✗ Physics loss is suspiciously zero: {physics_total:.6f}")
        all_pass = False

    if electrical > 1e-6:
        print(f"   ✓ Electrical loss is non-zero: {electrical:.6f}")
    else:
        print(f"   ✗ Electrical loss is zero: {electrical:.6f}")
        all_pass = False

    if mechanical > 1e-6:
        print(f"   ✓ Mechanical loss is non-zero: {mechanical:.6f}")
    else:
        print(f"   ✗ Mechanical loss is zero: {mechanical:.6f}")
        all_pass = False

    # Verdict
    print("\n" + "-"*70)
    if all_pass:
        print("✓ PASS: Physics loss components are computed correctly")
        return True
    else:
        print("✗ FAIL: Physics loss components are not computed correctly")
        return False


def test_training_integration():
    """
    Test 3: Verify how model is currently compiled in actual training.

    This test loads the actual training configuration and checks what loss
    function is being used when the model is compiled.
    """
    print("\n" + "="*70)
    print("TEST 3: TRAINING INTEGRATION TEST")
    print("="*70)
    print("Testing how physics loss is integrated in actual training code...")

    # Load default config
    config = ConfigManager.load_model_config('default')
    simulation_config = ConfigManager.load_simulation_config('default')
    config = ConfigManager.merge_configs(simulation_config, config)

    print("\n1. Configuration settings:")
    physics_config = config.get('physics_loss', {})
    print(f"   Physics loss enabled: {physics_config.get('enabled', False)}")
    print(f"   Physics loss weight:  {physics_config.get('weight', 0.0)}")
    print(f"   Physics start epoch:  {physics_config.get('start_epoch', 0)}")

    training_config = config.get('training', {})
    print(f"   Training loss:        '{training_config.get('loss', 'unknown')}'")

    # Create a dummy model
    input_shape = (10, 3)
    model = LSTMAutoencoder.from_config(config, input_shape=input_shape)
    model.build()

    # Compile it the same way train.py does
    model.compile(
        optimizer=training_config['optimizer']['type'],
        learning_rate=training_config['optimizer']['learning_rate'],
        loss=training_config['loss']  # This is the problem!
    )

    print("\n2. Model compilation analysis:")
    print(f"   Optimizer: {type(model.model.optimizer).__name__}")
    print(f"   Loss function: {model.model.loss}")
    print(f"   Loss type: {type(model.model.loss)}")

    # Check if loss is PhysicsInformedLoss
    is_physics_loss = isinstance(model.model.loss, PhysicsInformedLoss)
    is_custom = not isinstance(model.model.loss, str)

    print("\n3. Loss function verification:")
    if is_physics_loss:
        print("   ✓ Model is using PhysicsInformedLoss")
    elif is_custom:
        print(f"   ~ Model is using custom loss: {type(model.model.loss).__name__}")
    else:
        print(f"   ✗ Model is using simple string loss: '{model.model.loss}'")
        print("   → Physics loss is NOT integrated into training!")

    # Verdict
    print("\n" + "-"*70)
    if is_physics_loss:
        print("✓ PASS: Physics loss is properly integrated in model.compile()")
        return True
    else:
        print("✗ FAIL: Physics loss is NOT integrated in model.compile()")
        print("\nThe model is compiled with simple MSE loss, not PhysicsInformedLoss!")
        print("This means gradients are computed ONLY from reconstruction error,")
        print("and physics constraints are completely ignored during optimization.")
        return False


def test_current_vs_corrected():
    """
    Test 4: Compare current implementation vs corrected implementation.

    This test demonstrates the difference between the current broken implementation
    and the corrected version where physics loss is properly used.
    """
    print("\n" + "="*70)
    print("TEST 4: CURRENT vs CORRECTED IMPLEMENTATION")
    print("="*70)

    # Setup
    batch_size, seq_len, n_features = 2, 10, 3
    np.random.seed(42)

    current = np.random.uniform(1.0, 2.0, (batch_size, seq_len, 1))
    omega = np.random.uniform(100, 150, (batch_size, seq_len, 1))
    voltage = np.random.uniform(15, 20, (batch_size, seq_len, 1))

    y_data = np.concatenate([current, omega, voltage], axis=-1).astype(np.float32)
    y_true = tf.constant(y_data)

    motor_params = {
        'R': 1.0, 'L': 0.5, 'Kt': 0.01,
        'Ke': 0.01, 'J': 0.01, 'B': 0.1
    }

    # Create two models
    input_shape = (seq_len, n_features)

    # Model 1: Current (wrong) implementation - compiled with 'mse'
    print("\n1. Current implementation (model compiled with 'mse'):")
    model_current = LSTMAutoencoder(
        input_shape=input_shape,
        encoder_units=[8],
        decoder_units=[8],
        dropout=0.0
    )
    model_current.build()
    model_current.compile(loss='mse')  # ← Current implementation

    with tf.GradientTape() as tape:
        y_pred = model_current.model(y_true, training=True)
        loss_current = model_current.model.compiled_loss(y_true, y_pred)

    grads_current = tape.gradient(loss_current, model_current.model.trainable_variables)
    grad_norm_current = tf.norm(grads_current[0]).numpy()

    print(f"   Loss value: {loss_current.numpy():.6f}")
    print(f"   Gradient norm: {grad_norm_current:.6f}")
    print(f"   Loss type: {type(model_current.model.loss)}")

    # Model 2: Corrected implementation - compiled with PhysicsInformedLoss
    print("\n2. Corrected implementation (model compiled with PhysicsInformedLoss):")
    model_corrected = LSTMAutoencoder(
        input_shape=input_shape,
        encoder_units=[8],
        decoder_units=[8],
        dropout=0.0
    )
    model_corrected.build()

    physics_loss_fn = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=0.1,
        enabled=True,
        dt=0.002
    )
    physics_loss_fn.set_epoch(100)

    # Compile with physics loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_corrected.model.compile(optimizer=optimizer, loss=physics_loss_fn)  # ← Corrected

    with tf.GradientTape() as tape:
        y_pred = model_corrected.model(y_true, training=True)
        loss_corrected = model_corrected.model.compiled_loss(y_true, y_pred)

    grads_corrected = tape.gradient(loss_corrected, model_corrected.model.trainable_variables)
    grad_norm_corrected = tf.norm(grads_corrected[0]).numpy()

    print(f"   Loss value: {loss_corrected.numpy():.6f}")
    print(f"   Gradient norm: {grad_norm_corrected:.6f}")
    print(f"   Loss type: {type(model_corrected.model.loss)}")

    # Compare
    print("\n3. Comparison:")
    loss_diff = abs(loss_corrected.numpy() - loss_current.numpy())
    grad_diff = abs(grad_norm_corrected - grad_norm_current)

    print(f"   Loss difference: {loss_diff:.6f}")
    print(f"   Gradient difference: {grad_diff:.6f}")

    # Verdict
    print("\n" + "-"*70)
    if loss_diff > 1e-4 or grad_diff > 1e-4:
        print("✓ VERIFICATION: Implementations produce DIFFERENT results")
        print(f"  Using PhysicsInformedLoss changes the loss by {loss_diff:.6f}")
        print(f"  and gradients by {grad_diff:.6f}")
        print("\n  → This proves physics loss CAN affect training when properly integrated!")
        return True
    else:
        print("? UNEXPECTED: Implementations produce same results")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  PHYSICS LOSS INTEGRATION VERIFICATION SUITE" + " "*22 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    results = {}

    try:
        results['gradient_flow'] = test_gradient_flow()
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED WITH ERROR: {e}")
        results['gradient_flow'] = False

    try:
        results['loss_components'] = test_loss_components()
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED WITH ERROR: {e}")
        results['loss_components'] = False

    try:
        results['training_integration'] = test_training_integration()
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED WITH ERROR: {e}")
        results['training_integration'] = False

    try:
        results['comparison'] = test_current_vs_corrected()
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED WITH ERROR: {e}")
        results['comparison'] = False

    # Final summary
    print("\n" + "#"*70)
    print("#  FINAL SUMMARY" + " "*53 + "#")
    print("#"*70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTests passed: {passed_tests}/{total_tests}\n")

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")

    print("\n" + "="*70)
    if results.get('training_integration') == False:
        print("\n🚨 CRITICAL ISSUE CONFIRMED:")
        print("\nPhysics loss is NOT integrated into gradient computation!")
        print("The model is compiled with simple 'mse' loss in train.py:109")
        print("\nTO FIX: Compile the model with PhysicsInformedLoss object instead")
        print("        of the string 'mse'")
        print("\nSee the corrected implementation in Test 4 above.")
    elif passed_tests == total_tests:
        print("\n✓ All tests passed! Physics loss is properly integrated.")
    else:
        print("\n⚠ Some tests failed. Please review the output above.")

    print("="*70 + "\n")

    return passed_tests == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
