"""Experiment management utilities."""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ExperimentPaths:
    """Container for experiment directory paths."""
    root: str
    config: str
    checkpoints: str
    plots: str
    logs: str


class ExperimentManager:
    """Manages experiment folder creation, naming, and tracking."""

    def __init__(self, base_path: str, config: Dict[str, Any]):
        """
        Initialize ExperimentManager.

        Args:
            base_path: Base path for experiments
            config: Experiment configuration
        """
        self.base_path = Path(base_path)
        self.config = config
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_experiment(self) -> ExperimentPaths:
        """
        Create experiment directory structure.

        Returns:
            ExperimentPaths object containing all paths

        Format: exp_{id}_{date}_{encoder_units}_{lr}_{batch}_{epochs}_{name}
        Example: exp_001_20241215_64-32_lr0.001_b32_e100_friction_test
        """
        exp_name = self.generate_experiment_name(self.config)
        exp_root = self.base_path / exp_name

        # Create directory structure
        exp_root.mkdir(parents=True, exist_ok=True)
        (exp_root / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (exp_root / 'plots').mkdir(parents=True, exist_ok=True)
        (exp_root / 'logs').mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = exp_root / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        paths = ExperimentPaths(
            root=str(exp_root),
            config=str(config_path),
            checkpoints=str(exp_root / 'checkpoints'),
            plots=str(exp_root / 'plots'),
            logs=str(exp_root / 'logs')
        )

        return paths

    def generate_experiment_name(self, config: Dict[str, Any]) -> str:
        """
        Generate experiment name from configuration.

        Format: exp_{id}_{date}_{encoder_units}_{lr}_{batch}_{epochs}_{name}
        Example: exp_001_20241215_64-32_lr0.001_b32_e100_friction_test

        Args:
            config: Experiment configuration

        Returns:
            Generated experiment name
        """
        # Get experiment ID
        exp_id = self.get_next_id() if config['experiment'].get('auto_increment_id', True) else 0
        exp_id_str = f"{exp_id:03d}"

        # Get date
        date_str = datetime.now().strftime("%Y%m%d")

        # Get encoder architecture
        encoder_units = config['model']['encoder_units']
        encoder_str = "-".join(map(str, encoder_units))

        # Get learning rate
        lr = config['training']['optimizer']['learning_rate']
        lr_str = f"lr{lr}"

        # Get batch size
        batch_size = config['training']['batch_size']
        batch_str = f"b{batch_size}"

        # Get epochs
        epochs = config['training']['epochs']
        epochs_str = f"e{epochs}"

        # Get custom name
        custom_name = config['experiment']['name']
        if custom_name and custom_name != 'default':
            name_str = custom_name.replace(' ', '_')
        else:
            name_str = ""

        # Build experiment name
        parts = [
            f"exp_{exp_id_str}",
            date_str,
            encoder_str,
            lr_str,
            batch_str,
            epochs_str
        ]

        if name_str:
            parts.append(name_str)

        exp_name = "_".join(parts)
        return exp_name

    def get_next_id(self) -> int:
        """
        Get next available experiment ID.

        Returns:
            Next experiment ID
        """
        if not self.base_path.exists():
            return 1

        existing_experiments = [d for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith('exp_')]

        if not existing_experiments:
            return 1

        # Extract IDs from experiment names
        ids = []
        for exp_dir in existing_experiments:
            try:
                # Extract ID from exp_XXX_...
                id_str = exp_dir.name.split('_')[1]
                ids.append(int(id_str))
            except (IndexError, ValueError):
                continue

        return max(ids) + 1 if ids else 1

    def list_experiments(self) -> List[str]:
        """
        List all experiments in base path.

        Returns:
            List of experiment directory names
        """
        if not self.base_path.exists():
            return []

        experiments = [
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith('exp_')
        ]

        return sorted(experiments)

    def load_experiment(self, name_or_id: str) -> ExperimentPaths:
        """
        Load experiment by name or ID.

        Args:
            name_or_id: Experiment name or ID (e.g., 'exp_001_...' or '001' or '1')

        Returns:
            ExperimentPaths object

        Raises:
            ValueError: If experiment not found
        """
        # Try to find by exact name first
        exp_root = self.base_path / name_or_id
        if exp_root.exists():
            return self._create_paths_from_root(exp_root)

        # Try to find by ID
        try:
            exp_id = int(name_or_id)
            exp_id_str = f"{exp_id:03d}"

            # Find experiment starting with this ID
            for exp_dir in self.base_path.iterdir():
                if exp_dir.is_dir() and exp_dir.name.startswith(f"exp_{exp_id_str}_"):
                    return self._create_paths_from_root(exp_dir)

        except ValueError:
            pass

        raise ValueError(f"Experiment not found: {name_or_id}")

    def _create_paths_from_root(self, root: Path) -> ExperimentPaths:
        """
        Create ExperimentPaths from root directory.

        Args:
            root: Root experiment directory

        Returns:
            ExperimentPaths object
        """
        return ExperimentPaths(
            root=str(root),
            config=str(root / 'config.json'),
            checkpoints=str(root / 'checkpoints'),
            plots=str(root / 'plots'),
            logs=str(root / 'logs')
        )

    def save_config(self, config: Dict[str, Any], experiment_root: str) -> None:
        """
        Save configuration to experiment directory.

        Args:
            config: Configuration to save
            experiment_root: Root directory of experiment
        """
        config_path = Path(experiment_root) / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
