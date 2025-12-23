#!/usr/bin/env python3
"""Simulate DC motor response to voltage input."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.simulation.input_generator import VoltageInputGenerator
from src.simulation.dc_motor import DCMotorSimulator
from src.visualization.plotter import Plotter


def main():
    parser = argparse.ArgumentParser(description='Simulate DC motor response')
    parser.add_argument('--sim-config', type=str, help='Simulation config ID or path (default: default)')
    parser.add_argument('--input', type=str, help='Path to voltage input file (optional, will generate if not provided)')
    parser.add_argument('--output', type=str, help='Output directory (optional, default: data/raw/{simulation_id})')
    args = parser.parse_args()

    # Load simulation configuration
    if args.sim_config:
        if args.sim_config.endswith('.json'):
            # Full path provided
            print(f"Loading simulation configuration from {args.sim_config}")
            config = ConfigManager.load(args.sim_config)
            ConfigManager.validate(config, config_type='simulation')
        else:
            # ID provided
            print(f"Loading simulation configuration: {args.sim_config}")
            config = ConfigManager.load_simulation_config(args.sim_config)
    else:
        # Use default
        print("Loading default simulation configuration")
        config = ConfigManager.load_simulation_config('default')

    simulation_id = config['simulation_id']
    print(f"Simulation ID: {simulation_id}")

    # Setup directories for this simulation
    input_dir = ConfigManager.get_simulation_data_path(simulation_id, 'inputs')
    output_dir = Path(args.output) if args.output else ConfigManager.get_simulation_data_path(simulation_id, 'raw')

    # Generate or load voltage input
    if args.input:
        print(f"Loading voltage input from {args.input}")
        voltage_data = np.load(args.input, allow_pickle=True).item()
        voltage_input = voltage_data['voltage']
    else:
        print("Generating voltage input signal...")
        generator = VoltageInputGenerator.from_config(config)
        voltage_input = generator.generate()

        # Save generated input
        input_path = input_dir / 'voltage_input.npy'
        generator.save(str(input_path))
        print(f"Voltage input saved to {input_path}")

        # Plot voltage input
        plotter = Plotter.from_config(config)
        fig = generator.plot(title=f"Generated Voltage Input ({simulation_id})")
        plotter.save_figure(fig, input_dir / 'voltage_input')

    # Run simulation
    print("Running DC motor simulation...")
    simulator = DCMotorSimulator.from_config(config)
    external_torque = config['simulation'].get('external_torque', 0.0)
    result = simulator.simulate(voltage_input, external_torque=external_torque)

    # Save simulation results
    output_file = output_dir / 'simulation_result.npy'
    simulator.save_results(result, str(output_file), format='npy')
    print(f"Simulation results saved to {output_file}")

    # Plot simulation results
    print("Generating plots...")
    plotter = Plotter.from_config(config)
    fig = plotter.plot_simulation(
        time=result.time,
        voltage=result.voltage,
        current=result.current,
        angular_velocity=result.angular_velocity,
        title=f"DC Motor Simulation Results ({simulation_id})"
    )
    plotter.save_figure(fig, output_dir / 'simulation_results')

    print("\nSimulation completed successfully!")
    print(f"Simulation ID: {simulation_id}")
    print(f"Duration: {result.time[-1]:.3f} seconds")
    print(f"Samples: {len(result.time)}")
    print(f"Sampling rate: {result.metadata['sampling_rate']} Hz")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
