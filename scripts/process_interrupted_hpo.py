#!/usr/bin/env python3
"""Process interrupted HPO runs and generate summary from saved trial data.

When HPO is stopped early, trial data remains in experiments/hpo_temp/.
This script processes that data to generate the same outputs as a completed run:
- Sorted trial summary
- Best trial identification
- Copy best trial artifacts to output directory

Usage:
    python scripts/process_interrupted_hpo.py --output experiments/hpo_recovered
    python scripts/process_interrupted_hpo.py --output experiments/hpo_recovered --metric val_loss --direction minimize
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_trial_data(trial_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load trial data from a trial directory.

    Args:
        trial_dir: Path to trial directory (e.g., experiments/hpo_temp/trial_0)

    Returns:
        Dictionary with trial data, or None if invalid
    """
    trial_data = {
        'trial_number': None,
        'params': {},
        'layer_sequences': {},
        'history': {},
        'best_val_loss': None,
        'final_val_loss': None,
        'best_epoch': None,
        'total_epochs': 0,
        'state': 'UNKNOWN'
    }

    # Extract trial number from directory name
    try:
        trial_data['trial_number'] = int(trial_dir.name.split('_')[1])
    except (ValueError, IndexError):
        print(f"  Warning: Could not parse trial number from {trial_dir.name}")
        return None

    # Load trial parameters
    params_path = trial_dir / 'trial_params.json'
    if params_path.exists():
        with open(params_path, 'r') as f:
            trial_data['params'] = json.load(f)

    # Load training history
    history_path = trial_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            trial_data['history'] = json.load(f)

        # Extract metrics from history
        if 'val_loss' in trial_data['history']:
            val_losses = trial_data['history']['val_loss']
            trial_data['best_val_loss'] = min(val_losses)
            trial_data['final_val_loss'] = val_losses[-1]
            trial_data['best_epoch'] = val_losses.index(trial_data['best_val_loss']) + 1
            trial_data['total_epochs'] = len(val_losses)
            trial_data['state'] = 'COMPLETE'
    else:
        trial_data['state'] = 'INCOMPLETE'

    # Load trial config if available (contains layer sequences)
    config_path = trial_dir / 'trial_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'model' in config:
                if 'encoder_units' in config['model']:
                    trial_data['layer_sequences']['model.encoder_units'] = config['model']['encoder_units']
                if 'decoder_units' in config['model']:
                    trial_data['layer_sequences']['model.decoder_units'] = config['model']['decoder_units']

    # Infer layer sequences from params if not in config
    if not trial_data['layer_sequences']:
        encoder_layers = []
        decoder_layers = []

        # Look for encoder layer params
        for key, value in trial_data['params'].items():
            if 'encoder' in key and 'layer_' in key and 'depth' not in key:
                try:
                    idx = int(key.split('layer_')[1])
                    while len(encoder_layers) <= idx:
                        encoder_layers.append(None)
                    encoder_layers[idx] = value
                except (ValueError, IndexError):
                    pass

        # Filter out None values and set
        encoder_layers = [x for x in encoder_layers if x is not None]
        if encoder_layers:
            trial_data['layer_sequences']['model.encoder_units'] = encoder_layers
            trial_data['layer_sequences']['model.decoder_units'] = list(reversed(encoder_layers))

    return trial_data


def process_hpo_temp(
    temp_dir: Path = Path('experiments/hpo_temp'),
    output_dir: Path = Path('experiments/hpo_recovered'),
    metric: str = 'val_loss',
    direction: str = 'minimize'
) -> None:
    """
    Process interrupted HPO trials and generate summary.

    Args:
        temp_dir: Directory containing trial subdirectories
        output_dir: Directory to save processed results
        metric: Metric to use for ranking (default: val_loss)
        direction: 'minimize' or 'maximize'
    """
    print("=" * 70)
    print("PROCESSING INTERRUPTED HPO TRIALS")
    print("=" * 70)
    print(f"\nSource: {temp_dir}")
    print(f"Output: {output_dir}")
    print(f"Metric: {metric} ({direction})")
    print()

    if not temp_dir.exists():
        print(f"ERROR: Directory not found: {temp_dir}")
        return

    # Find all trial directories
    trial_dirs = sorted([
        d for d in temp_dir.iterdir()
        if d.is_dir() and d.name.startswith('trial_')
    ], key=lambda x: int(x.name.split('_')[1]))

    if not trial_dirs:
        print(f"ERROR: No trial directories found in {temp_dir}")
        return

    print(f"Found {len(trial_dirs)} trial directories")
    print()

    # Load data from each trial
    trials = []
    for trial_dir in trial_dirs:
        print(f"Loading {trial_dir.name}...", end=" ")
        trial_data = load_trial_data(trial_dir)
        if trial_data:
            trials.append(trial_data)
            if trial_data['best_val_loss'] is not None:
                print(f"best_{metric}={trial_data['best_val_loss']:.6f}, epochs={trial_data['total_epochs']}")
            else:
                print(f"state={trial_data['state']}")
        else:
            print("SKIPPED (invalid)")

    # Filter to only complete trials with valid metrics
    complete_trials = [t for t in trials if t['best_val_loss'] is not None]

    if not complete_trials:
        print("\nERROR: No trials with valid metrics found")
        return

    print(f"\n{len(complete_trials)} trials have valid metrics")

    # Sort trials by metric
    reverse = (direction == 'maximize')
    sorted_trials = sorted(
        complete_trials,
        key=lambda x: x['best_val_loss'],
        reverse=reverse
    )

    # Identify best trial
    best_trial = sorted_trials[0]

    print(f"\nBest trial: #{best_trial['trial_number']}")
    print(f"  Best {metric}: {best_trial['best_val_loss']:.6f}")
    print(f"  Best epoch: {best_trial['best_epoch']}/{best_trial['total_epochs']}")
    if best_trial['layer_sequences']:
        print(f"  Layer sequences: {best_trial['layer_sequences']}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save study summary
    summary = {
        'best_trial': best_trial['trial_number'],
        'best_value': best_trial['best_val_loss'],
        'best_params': best_trial['params'],
        'best_layer_sequences': best_trial['layer_sequences'],
        'best_epoch': best_trial['best_epoch'],
        'total_epochs': best_trial['total_epochs'],
        'n_trials': len(trials),
        'n_complete': len(complete_trials),
        'metric': metric,
        'direction': direction,
        'source': str(temp_dir),
        'recovered': True
    }

    summary_path = output_dir / 'study_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Save all trials sorted
    all_trials = []
    for trial in sorted_trials:
        trial_entry = {
            'trial_number': trial['trial_number'],
            'value': trial['best_val_loss'],
            'params': trial['params'],
            'layer_sequences': trial['layer_sequences'],
            'best_epoch': trial['best_epoch'],
            'total_epochs': trial['total_epochs'],
            'state': trial['state']
        }
        all_trials.append(trial_entry)

    trials_path = output_dir / 'all_trials.json'
    with open(trials_path, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"Saved: {trials_path}")

    # Copy best trial artifacts
    best_trial_dir = temp_dir / f"trial_{best_trial['trial_number']}"

    # Copy training history
    history_src = best_trial_dir / 'training_history.json'
    if history_src.exists():
        history_dst = output_dir / 'training_history.json'
        shutil.copy(history_src, history_dst)
        print(f"Saved: {history_dst}")

    # Copy plots
    plots_src = best_trial_dir / 'plots'
    if plots_src.exists() and plots_src.is_dir():
        plots_dst = output_dir / 'plots'
        if plots_dst.exists():
            shutil.rmtree(plots_dst)
        shutil.copytree(plots_src, plots_dst)
        print(f"Saved: {plots_dst}/")

    # Copy trial config if available
    config_src = best_trial_dir / 'trial_config.json'
    if config_src.exists():
        config_dst = output_dir / 'best_config.json'
        shutil.copy(config_src, config_dst)
        print(f"Saved: {config_dst}")

    print("\n" + "=" * 70)
    print("TRIAL RANKING (sorted by best val_loss)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Trial':<8} {'Best Val Loss':<15} {'Epochs':<10} {'Encoder Layers'}")
    print("-" * 70)
    for rank, trial in enumerate(sorted_trials[:20], 1):  # Show top 20
        encoder = trial['layer_sequences'].get('model.encoder_units', '?')
        print(f"{rank:<6} #{trial['trial_number']:<6} {trial['best_val_loss']:<15.6f} {trial['total_epochs']:<10} {encoder}")

    if len(sorted_trials) > 20:
        print(f"... and {len(sorted_trials) - 20} more trials")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Process interrupted HPO runs and generate summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process with default settings
    python scripts/process_interrupted_hpo.py --output experiments/hpo_recovered

    # Specify custom temp directory
    python scripts/process_interrupted_hpo.py --temp experiments/hpo_temp --output experiments/hpo_recovered

    # Use different metric (if your HPO tracked something else)
    python scripts/process_interrupted_hpo.py --output experiments/hpo_recovered --metric val_loss --direction minimize
        """
    )
    parser.add_argument(
        '--temp',
        type=str,
        default='experiments/hpo_temp',
        help='Directory containing trial subdirectories (default: experiments/hpo_temp)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed results'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='val_loss',
        help='Metric to use for ranking trials (default: val_loss)'
    )
    parser.add_argument(
        '--direction',
        type=str,
        choices=['minimize', 'maximize'],
        default='minimize',
        help='Optimization direction (default: minimize)'
    )

    args = parser.parse_args()

    process_hpo_temp(
        temp_dir=Path(args.temp),
        output_dir=Path(args.output),
        metric=args.metric,
        direction=args.direction
    )


if __name__ == '__main__':
    main()
