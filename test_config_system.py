#!/usr/bin/env python3
"""Test the new config system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import ConfigManager

def test_simulation_config():
    """Test loading simulation config."""
    print("=" * 60)
    print("Testing Simulation Config Loading")
    print("=" * 60)

    try:
        config = ConfigManager.load_simulation_config('default')
        print("✓ Successfully loaded simulation config")
        print(f"  - Simulation ID: {config['simulation_id']}")
        print(f"  - Motor params: R={config['simulation']['motor_params']['R']}, L={config['simulation']['motor_params']['L']}")
        print(f"  - Sampling rate: {config['simulation']['sampling_rate_hz']} Hz")
        return True
    except Exception as e:
        print(f"✗ Failed to load simulation config: {e}")
        return False

def test_model_config():
    """Test loading model config."""
    print("\n" + "=" * 60)
    print("Testing Model Config Loading")
    print("=" * 60)

    try:
        config = ConfigManager.load_model_config('default')
        print("✓ Successfully loaded model config")
        print(f"  - Model ID: {config.get('model_id')}")
        print(f"  - Simulation ID reference: {config['simulation_id']}")
        print(f"  - Encoder units: {config['model']['encoder_units']}")
        print(f"  - Window size: {config['data_processing']['window_size']}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model config: {e}")
        return False

def test_merge_configs():
    """Test merging simulation and model configs."""
    print("\n" + "=" * 60)
    print("Testing Config Merging")
    print("=" * 60)

    try:
        sim_config = ConfigManager.load_simulation_config('default')
        model_config = ConfigManager.load_model_config('default')
        merged = ConfigManager.merge_configs(sim_config, model_config)

        print("✓ Successfully merged configs")
        print(f"  - Has simulation section: {'simulation' in merged}")
        print(f"  - Has model section: {'model' in merged}")
        print(f"  - Has training section: {'training' in merged}")
        print(f"  - Motor params available: {merged['simulation']['motor_params']['R']}")
        print(f"  - Model encoder units: {merged['model']['encoder_units']}")
        return True
    except Exception as e:
        print(f"✗ Failed to merge configs: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_paths():
    """Test data path generation."""
    print("\n" + "=" * 60)
    print("Testing Data Path Generation")
    print("=" * 60)

    try:
        inputs_path = ConfigManager.get_simulation_data_path('default', 'inputs')
        raw_path = ConfigManager.get_simulation_data_path('default', 'raw')
        processed_path = ConfigManager.get_simulation_data_path('default', 'processed')

        print("✓ Successfully generated data paths")
        print(f"  - Inputs: {inputs_path}")
        print(f"  - Raw: {raw_path}")
        print(f"  - Processed: {processed_path}")

        # Check if directories were created
        if inputs_path.exists() and raw_path.exists() and processed_path.exists():
            print("✓ Directories were created successfully")

        return True
    except Exception as e:
        print(f"✗ Failed to generate data paths: {e}")
        return False

def test_validation():
    """Test config validation."""
    print("\n" + "=" * 60)
    print("Testing Config Validation")
    print("=" * 60)

    try:
        # Test simulation config validation
        sim_config = ConfigManager.load('configs/simulation/default.json')
        ConfigManager.validate(sim_config, config_type='simulation')
        print("✓ Simulation config validation passed")

        # Test model config validation
        model_config = ConfigManager.load('configs/model/default.json')
        ConfigManager.validate(model_config, config_type='model')
        print("✓ Model config validation passed")

        return True
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CONFIG SYSTEM TEST SUITE")
    print("=" * 60 + "\n")

    results = []
    results.append(("Simulation Config Loading", test_simulation_config()))
    results.append(("Model Config Loading", test_model_config()))
    results.append(("Config Merging", test_merge_configs()))
    results.append(("Data Path Generation", test_data_paths()))
    results.append(("Config Validation", test_validation()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
