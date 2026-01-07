#!/usr/bin/env python3
"""Test script to verify layer sequence sampling with gain constraints."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.samplers import TPESampler
from src.optimization.hpo import HyperparameterOptimizer


def test_layer_sequence_sampling():
    """Test that layer sequence sampling works correctly."""
    print("Testing layer sequence sampling with gain constraints...")
    print("=" * 70)

    # Create a simple test configuration
    base_config = {
        'model': {},
        'training': {}
    }

    # Define parameter space with layer_sequence
    param_space = {
        'model.encoder_units': {
            'type': 'layer_sequence',
            'depth_choices': [2, 3],
            'low': 16,
            'high': 128,
            'step': 16,
            'gain': 0.5
        },
        'model.decoder_units': {
            'type': 'layer_sequence',
            'mirror_from': 'model.encoder_units'
        }
    }

    # Create optimizer instance
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        param_space=param_space,
        n_trials=10,
        metric='val_loss',
        direction='minimize'
    )

    # Create a study to test sampling
    study = optuna.create_study(direction='minimize', sampler=TPESampler())

    print("\nTesting 10 different layer configurations:\n")

    # Test multiple trials
    for i in range(10):
        trial = study.ask()

        # Sample configuration
        trial_config = optimizer._create_trial_config(trial)

        encoder_units = trial_config['model'].get('encoder_units', [])
        decoder_units = trial_config['model'].get('decoder_units', [])

        print(f"Trial {i+1}:")
        print(f"  Depth: {len(encoder_units)}")
        print(f"  Encoder: {encoder_units}")
        print(f"  Decoder: {decoder_units}")

        # Verify constraints
        valid = True

        # Check encoder gain constraints
        for j in range(1, len(encoder_units)):
            ratio = encoder_units[j] / encoder_units[j-1]
            if ratio > 0.5:
                print(f"  ⚠ WARNING: Layer {j} violates gain constraint! "
                      f"{encoder_units[j]}/{encoder_units[j-1]} = {ratio:.2f} > 0.5")
                valid = False

        # Check decoder is reverse of encoder
        if decoder_units != list(reversed(encoder_units)):
            print(f"  ⚠ WARNING: Decoder is not symmetric to encoder!")
            valid = False

        if valid:
            print(f"  ✓ All constraints satisfied")

        print()

        # Tell the study this trial is complete (with dummy value)
        study.tell(trial, 1.0)

    print("=" * 70)
    print("Layer sequence sampling test completed!")


def test_parameter_space_examples():
    """Test specific examples from user requirements."""
    print("\nTesting specific examples from requirements...")
    print("=" * 70)

    print("\nValid examples (should work):")
    print("  [64, 32] - ratio: 32/64 = 0.50 ✓")
    print("  [96, 32] - ratio: 32/96 = 0.33 ✓")
    print("  [128, 64] - ratio: 64/128 = 0.50 ✓")

    print("\nInvalid examples (should NOT be sampled):")
    print("  [64, 48] - ratio: 48/64 = 0.75 ✗ (> 0.5)")
    print("  [96, 128] - ratio: 128/96 = 1.33 ✗ (> 1.0, wrong direction)")
    print("  [128, 96] - ratio: 96/128 = 0.75 ✗ (> 0.5)")

    print("\n" + "=" * 70)


def test_early_stopping_parameter():
    """Test that early stopping patience can be sampled."""
    print("\nTesting early stopping patience parameter...")
    print("=" * 70)

    base_config = {
        'training': {
            'early_stopping': {
                'enabled': True
            }
        }
    }

    param_space = {
        'training.early_stopping.patience': {
            'type': 'int',
            'low': 10,
            'high': 30,
            'step': 5
        }
    }

    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        param_space=param_space,
        n_trials=5,
        metric='val_loss',
        direction='minimize'
    )

    study = optuna.create_study(direction='minimize', sampler=TPESampler())

    print("\nSampling early stopping patience values:\n")

    for i in range(5):
        trial = study.ask()
        trial_config = optimizer._create_trial_config(trial)

        patience = trial_config['training']['early_stopping'].get('patience')
        print(f"Trial {i+1}: early_stopping.patience = {patience}")

        study.tell(trial, 1.0)

    print("\n" + "=" * 70)
    print("Early stopping parameter test completed!")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("HPO LAYER OPTIMIZATION TEST SUITE")
    print("=" * 70 + "\n")

    test_layer_sequence_sampling()
    test_parameter_space_examples()
    test_early_stopping_parameter()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70 + "\n")
