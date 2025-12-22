"""Configuration management utilities."""

import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading, validation, merging, and saving."""

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            config = json.load(f)

        return config

    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and required fields.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = [
            'experiment',
            'simulation',
            'data_processing',
            'normalization',
            'model',
            'training',
            'paths'
        ]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate experiment section
        if 'name' not in config['experiment']:
            raise ValueError("Missing 'name' in experiment configuration")

        # Validate simulation section
        required_motor_params = ['R', 'L', 'Kt', 'Ke', 'J', 'B']
        if 'motor_params' not in config['simulation']:
            raise ValueError("Missing 'motor_params' in simulation configuration")

        for param in required_motor_params:
            if param not in config['simulation']['motor_params']:
                raise ValueError(f"Missing motor parameter: {param}")

        # Validate data processing
        if 'window_size' not in config['data_processing']:
            raise ValueError("Missing 'window_size' in data_processing configuration")

        # Validate model architecture
        if 'encoder_units' not in config['model']:
            raise ValueError("Missing 'encoder_units' in model configuration")
        if 'decoder_units' not in config['model']:
            raise ValueError("Missing 'decoder_units' in model configuration")

        # Validate training
        if 'epochs' not in config['training']:
            raise ValueError("Missing 'epochs' in training configuration")
        if 'batch_size' not in config['training']:
            raise ValueError("Missing 'batch_size' in training configuration")

        return True

    @staticmethod
    def merge_with_defaults(config: Dict[str, Any], default_config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Merge user configuration with defaults.

        Args:
            config: User configuration
            default_config_path: Path to default config file (optional)

        Returns:
            Merged configuration
        """
        if default_config_path:
            default_config = ConfigManager.load(default_config_path)
        else:
            # Use built-in defaults if no path provided
            default_config = {}

        merged = ConfigManager._deep_merge(default_config, config)
        return merged

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            overlay: Dictionary to overlay on base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def save(config: Dict[str, Any], filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
            filepath: Path where to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated path (e.g., 'model.encoder_units')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    @staticmethod
    def set_nested(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
        """
        Set nested configuration value using dot notation.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated path (e.g., 'model.encoder_units')
            value: Value to set

        Returns:
            Modified configuration dictionary
        """
        keys = key_path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        return config
