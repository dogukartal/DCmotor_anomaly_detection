#!/usr/bin/env python3
"""Test script to verify min_ratio_of constraint for patience parameters."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.samplers import TPESampler


def sample_with_min_ratio_constraint(trial, param_path, low, high, step, ref_value, ratio):
    """
    Sample integer parameter with min_ratio_of constraint.

    Args:
        trial: Optuna trial
        param_path: Parameter name
        low: Original low bound
        high: Original high bound
        step: Step size
        ref_value: Reference parameter value
        ratio: Minimum ratio (result >= ref_value * ratio)

    Returns:
        Sampled value
    """
    # Calculate minimum allowed value
    min_value = int(ref_value * ratio)

    # Ensure the lower bound doesn't go below original low limit
    constrained_low = max(min_value, low)

    # Ensure low <= high
    if constrained_low > high:
        # If constraint makes range invalid, use maximum valid value
        value = high
    else:
        value = trial.suggest_int(
            param_path,
            constrained_low,
            high,
            step=step
        )

    return value


def test_patience_constraint():
    """Test that early stopping patience constraint works correctly."""
    print("Testing min_ratio_of constraint for patience parameters...")
    print("=" * 70)

    # Configuration
    lr_patience_low = 3
    lr_patience_high = 15
    lr_patience_step = 1

    es_patience_low = 10
    es_patience_high = 30
    es_patience_step = 5

    ratio = 2.0

    print("\nConfiguration:")
    print(f"  LR Scheduler Patience: [{lr_patience_low}, {lr_patience_high}] step={lr_patience_step}")
    print(f"  Early Stopping Patience: [{es_patience_low}, {es_patience_high}] step={es_patience_step}")
    print(f"  Constraint: early_stopping.patience >= lr_scheduler.patience * {ratio}")
    print()

    # Create study
    study = optuna.create_study(direction='minimize', sampler=TPESampler())

    print("Sampling 20 patience configurations:\n")

    valid_count = 0
    invalid_count = 0

    for i in range(20):
        trial = study.ask()

        # Sample LR scheduler patience first
        lr_patience = trial.suggest_int(
            'lr_scheduler_patience',
            lr_patience_low,
            lr_patience_high,
            step=lr_patience_step
        )

        # Sample early stopping patience with constraint
        es_patience = sample_with_min_ratio_constraint(
            trial,
            'early_stopping_patience',
            es_patience_low,
            es_patience_high,
            es_patience_step,
            lr_patience,
            ratio
        )

        # Verify constraint
        min_required = lr_patience * ratio
        is_valid = es_patience >= min_required

        status = "✓" if is_valid else "✗"
        print(f"Trial {i+1:2d} {status}: lr_patience={lr_patience:2d}, "
              f"es_patience={es_patience:2d}, min_required={min_required:.1f}")

        if not is_valid:
            print(f"          CONSTRAINT VIOLATION: {es_patience} < {min_required:.1f}")
            invalid_count += 1
        else:
            valid_count += 1

        # Tell the study this trial is complete
        study.tell(trial, 1.0)

    print("\n" + "=" * 70)
    print(f"Results: {valid_count} valid, {invalid_count} invalid")

    if invalid_count == 0:
        print("✓ ALL SAMPLES SATISFY CONSTRAINT!")
    else:
        print(f"⚠ WARNING: {invalid_count} samples violated constraint!")

    print("=" * 70)

    # Show examples
    print("\nExample valid configurations:")
    print("  lr_patience=5, es_patience=10 (min=10.0) ✓")
    print("  lr_patience=7, es_patience=15 (min=14.0) ✓")
    print("  lr_patience=10, es_patience=20 (min=20.0) ✓")
    print("\nExample invalid configurations (should NOT be sampled):")
    print("  lr_patience=10, es_patience=15 (min=20.0) ✗")
    print("  lr_patience=8, es_patience=10 (min=16.0) ✗")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PATIENCE CONSTRAINT TEST")
    print("=" * 70 + "\n")

    test_patience_constraint()

    print("\n" + "=" * 70)
    print("TEST COMPLETED!")
    print("=" * 70 + "\n")
