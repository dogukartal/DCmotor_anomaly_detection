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
    def validate(config: Dict[str, Any], config_type: str = 'full') -> bool:
        """
        Validate configuration structure and required fields.

        Args:
            config: Configuration dictionary to validate
            config_type: Type of config ('full', 'simulation', 'model')

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if config_type == 'simulation':
            # Validate simulation config
            if 'simulation_id' not in config:
                raise ValueError("Missing 'simulation_id' in simulation configuration")

            if 'simulation' not in config:
                raise ValueError("Missing 'simulation' section in simulation configuration")

            # Validate motor params
            required_motor_params = ['R', 'L', 'Kt', 'Ke', 'J', 'B']
            if 'motor_params' not in config['simulation']:
                raise ValueError("Missing 'motor_params' in simulation configuration")

            for param in required_motor_params:
                if param not in config['simulation']['motor_params']:
                    raise ValueError(f"Missing motor parameter: {param}")

            return True

        elif config_type == 'model':
            # Validate model config
            if 'simulation_id' not in config:
                raise ValueError("Missing 'simulation_id' in model configuration")

            if 'experiment' not in config or 'name' not in config['experiment']:
                raise ValueError("Missing 'experiment.name' in model configuration")

            if 'data_processing' not in config or 'window_size' not in config['data_processing']:
                raise ValueError("Missing 'data_processing.window_size' in model configuration")

            if 'model' not in config:
                raise ValueError("Missing 'model' section in model configuration")

            if 'encoder_units' not in config['model']:
                raise ValueError("Missing 'encoder_units' in model configuration")
            if 'decoder_units' not in config['model']:
                raise ValueError("Missing 'decoder_units' in model configuration")

            if 'training' not in config:
                raise ValueError("Missing 'training' section in model configuration")
            if 'epochs' not in config['training']:
                raise ValueError("Missing 'epochs' in training configuration")
            if 'batch_size' not in config['training']:
                raise ValueError("Missing 'batch_size' in training configuration")

            return True

        else:  # config_type == 'full' (backward compatibility)
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

    @staticmethod
    def load_simulation_config(simulation_id: str = 'default') -> Dict[str, Any]:
        """
        Load simulation configuration by ID.

        Args:
            simulation_id: Simulation configuration ID

        Returns:
            Simulation configuration dictionary

        Raises:
            FileNotFoundError: If simulation config file doesn't exist
        """
        config_path = Path('configs/simulation') / f'{simulation_id}.json'
        config = ConfigManager.load(config_path)
        ConfigManager.validate(config, config_type='simulation')
        return config

    @staticmethod
    def load_model_config(model_id: str = 'default') -> Dict[str, Any]:
        """
        Load model configuration by ID.

        Args:
            model_id: Model configuration ID

        Returns:
            Model configuration dictionary

        Raises:
            FileNotFoundError: If model config file doesn't exist
        """
        config_path = Path('configs/model') / f'{model_id}.json'
        config = ConfigManager.load(config_path)
        ConfigManager.validate(config, config_type='model')
        return config

    @staticmethod
    def merge_configs(simulation_config: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge simulation and model configurations into a single config.

        This is useful for scripts that need both simulation and model parameters.
        The model config's simulation_id is verified against the simulation config.

        Args:
            simulation_config: Simulation configuration
            model_config: Model configuration

        Returns:
            Merged configuration dictionary

        Raises:
            ValueError: If simulation_id mismatch
        """
        # Verify simulation_id match
        sim_id = simulation_config.get('simulation_id')
        model_sim_id = model_config.get('simulation_id')

        if sim_id != model_sim_id:
            raise ValueError(
                f"Simulation ID mismatch: simulation config has '{sim_id}' "
                f"but model config references '{model_sim_id}'"
            )

        # Create merged config
        merged = {}

        # Add all simulation config sections
        for key in simulation_config:
            if key != 'paths':  # Handle paths separately
                merged[key] = simulation_config[key]

        # Add all model config sections
        for key in model_config:
            if key not in ['simulation_id', 'paths']:  # Skip simulation_id and handle paths separately
                merged[key] = model_config[key]

        # Merge paths from both configs
        merged['paths'] = {}
        if 'paths' in simulation_config:
            merged['paths'].update(simulation_config['paths'])
        if 'paths' in model_config:
            merged['paths'].update(model_config['paths'])

        # Merge plotting settings (model config takes precedence)
        if 'plotting' in model_config:
            merged['plotting'] = model_config['plotting']
        elif 'plotting' in simulation_config:
            merged['plotting'] = simulation_config['plotting']

        return merged

    @staticmethod
    def get_simulation_data_path(simulation_id: str, data_type: str) -> Path:
        """
        Get the data path for a specific simulation configuration.

        Args:
            simulation_id: Simulation configuration ID
            data_type: Type of data ('inputs', 'raw', 'processed')

        Returns:
            Path to data directory for this simulation
        """
        base_path = Path('data') / data_type / simulation_id
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path
