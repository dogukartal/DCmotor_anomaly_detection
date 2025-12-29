#!/usr/bin/env python3
"""Test script to verify HPO trials export functionality."""

import sys
from pathlib import Path
import json
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from src.optimization.hpo import HyperparameterOptimizer


def dummy_objective(trial):
    """Simple objective function for testing."""
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_int('y', 0, 10)
    return (x - 2) ** 2 + (y - 3) ** 2


def test_all_trials_export():
    """Test that all trials are exported correctly."""
    print("Testing HPO all_trials.json export...")

    # Create a simple optimizer
    base_config = {
        'training': {
            'optimizer': {'type': 'adam', 'learning_rate': 0.001},
            'loss': 'mse'
        }
    }

    param_space = {
        'x': {'type': 'uniform', 'low': -10, 'high': 10},
        'y': {'type': 'int', 'low': 0, 'high': 10}
    }

    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        param_space=param_space,
        n_trials=10,
        metric='objective',
        direction='minimize'
    )

    # Create and run a study
    study = optuna.create_study(direction='minimize')
    study.optimize(dummy_objective, n_trials=10)
    optimizer.study = study

    # Create best config (simplified for test)
    optimizer.best_config = base_config

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer.save_study(tmpdir)

        # Verify files exist
        tmpdir_path = Path(tmpdir)
        assert (tmpdir_path / 'best_config.json').exists(), "best_config.json not created"
        assert (tmpdir_path / 'study_summary.json').exists(), "study_summary.json not created"
        assert (tmpdir_path / 'all_trials.json').exists(), "all_trials.json not created"

        # Load and verify all_trials.json
        with open(tmpdir_path / 'all_trials.json', 'r') as f:
            all_trials = json.load(f)

        print(f"\n✓ All files created successfully")
        print(f"✓ Number of trials exported: {len(all_trials)}")

        # Verify structure
        assert len(all_trials) == 10, f"Expected 10 trials, got {len(all_trials)}"

        for i, trial in enumerate(all_trials):
            assert 'trial_number' in trial, f"Trial {i} missing trial_number"
            assert 'value' in trial, f"Trial {i} missing value"
            assert 'params' in trial, f"Trial {i} missing params"
            assert 'state' in trial, f"Trial {i} missing state"
            assert 'x' in trial['params'], f"Trial {i} missing param 'x'"
            assert 'y' in trial['params'], f"Trial {i} missing param 'y'"

        print(f"✓ All trials have correct structure")

        # Verify sorting (should be ascending for minimize)
        values = [t['value'] for t in all_trials]
        assert values == sorted(values), "Trials not sorted correctly"

        print(f"✓ Trials correctly sorted by value (best first)")

        # Display sample output
        print(f"\nSample of all_trials.json (first 3 trials):")
        print(json.dumps(all_trials[:3], indent=2))

        print(f"\n✅ All tests passed!")


if __name__ == '__main__':
    test_all_trials_export()
