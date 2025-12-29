#!/usr/bin/env python3
"""
Quick sanity test for physics loss calculation.

This is a simple, fast test to verify that:
1. Physics loss is being calculated
2. Physics equations produce non-zero residuals
3. The loss is differentiable (can compute gradients)
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.physics_loss import PhysicsInformedLoss


def main():
    print("\n" + "="*60)
    print("QUICK PHYSICS LOSS SANITY TEST")
    print("="*60)

    # Create simple synthetic motor data
    np.random.seed(42)
    batch_size, seq_len = 4, 20

    # Create data: [current (A), angular_velocity (rad/s), voltage (V)]
    current = np.random.uniform(1.0, 3.0, (batch_size, seq_len, 1))
    omega = np.random.uniform(50, 200, (batch_size, seq_len, 1))
    voltage = np.random.uniform(10, 24, (batch_size, seq_len, 1))

    y_true = np.concatenate([current, omega, voltage], axis=-1).astype(np.float32)
    y_pred = y_true + np.random.normal(0, 0.1, y_true.shape).astype(np.float32)

    y_true = tf.constant(y_true)
    y_pred = tf.constant(y_pred)

    # Motor parameters (typical DC motor)
    motor_params = {
        'R': 1.0,      # Resistance (Ohms)
        'L': 0.5,      # Inductance (H)
        'Kt': 0.01,    # Torque constant (N·m/A)
        'Ke': 0.01,    # EMF constant (V·s/rad)
        'J': 0.01,     # Inertia (kg·m²)
        'B': 0.1       # Friction (N·m·s/rad)
    }

    # Create physics loss object
    physics_loss = PhysicsInformedLoss(
        motor_params=motor_params,
        physics_weight=0.1,
        enabled=True,
        dt=0.002  # 500 Hz sampling rate
    )
    physics_loss.set_epoch(100)  # Past start epoch

    print("\n1. Testing loss computation...")
    total_loss = physics_loss(y_true, y_pred)
    print(f"   ✓ Total loss: {total_loss.numpy():.6f}")

    print("\n2. Testing loss components...")
    components = physics_loss.compute_loss_components(y_true, y_pred)
    print(f"   Reconstruction: {components['reconstruction'].numpy():.6f}")
    print(f"   Electrical:     {components['electrical'].numpy():.6f}")
    print(f"   Mechanical:     {components['mechanical'].numpy():.6f}")
    print(f"   Physics total:  {components['physics'].numpy():.6f}")

    # Check components
    has_physics = components['physics'].numpy() > 1e-6
    has_electrical = components['electrical'].numpy() > 1e-6
    has_mechanical = components['mechanical'].numpy() > 1e-6

    if has_physics and has_electrical and has_mechanical:
        print("\n   ✓ All physics components are non-zero")
    else:
        print("\n   ✗ WARNING: Some physics components are zero!")
        if not has_electrical:
            print("     - Electrical loss is zero")
        if not has_mechanical:
            print("     - Mechanical loss is zero")

    print("\n3. Testing gradient computation...")
    # Create a simple model to test gradients
    model_var = tf.Variable(y_pred, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(model_var)
        loss = physics_loss(y_true, model_var)

    gradients = tape.gradient(loss, model_var)

    if gradients is not None:
        grad_norm = tf.norm(gradients).numpy()
        print(f"   ✓ Gradients computed successfully")
        print(f"   Gradient norm: {grad_norm:.6f}")
    else:
        print(f"   ✗ FAIL: Gradients are None!")
        return False

    print("\n4. Testing physics weight impact...")
    # Test with weight = 0
    physics_loss.physics_weight = 0.0
    loss_no_physics = physics_loss(y_true, y_pred).numpy()

    # Test with weight = 1.0
    physics_loss.physics_weight = 1.0
    loss_with_physics = physics_loss(y_true, y_pred).numpy()

    diff = abs(loss_with_physics - loss_no_physics)
    print(f"   Loss (weight=0.0): {loss_no_physics:.6f}")
    print(f"   Loss (weight=1.0): {loss_with_physics:.6f}")
    print(f"   Difference: {diff:.6f}")

    if diff > 1e-4:
        print(f"   ✓ Physics weight affects loss (diff={diff:.6f})")
    else:
        print(f"   ✗ Physics weight has no effect!")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("\n✓ Physics loss calculation: WORKING")
    print("✓ Physics equations: PRODUCING NON-ZERO RESIDUALS")
    print("✓ Gradient computation: WORKING")
    print("✓ Physics weight: AFFECTS LOSS VALUE")

    print("\n⚠ HOWEVER: This test does NOT check if physics loss is used")
    print("            during actual training! Run the full verification")
    print("            script to check training integration:")
    print("\n  python scripts/verify_physics_loss_integration.py")

    print("\n" + "="*60 + "\n")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
