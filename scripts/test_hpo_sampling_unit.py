#!/usr/bin/env python3
"""Unit test for layer sequence sampling logic (no TensorFlow required)."""

import optuna
from optuna.samplers import TPESampler


def sample_layer_sequence(trial, param_path, depth_choices, layer_min, layer_max, layer_step, gain):
    """
    Sample layer sequence with gain constraints (encoder logic).

    Args:
        trial: Optuna trial
        param_path: Parameter path name
        depth_choices: List of possible depths
        layer_min: Minimum layer size
        layer_max: Maximum layer size
        layer_step: Step size for layer sampling
        gain: Maximum ratio for next layer (next <= prev * gain)

    Returns:
        List of layer sizes
    """
    # Sample depth
    depth = trial.suggest_categorical(f"{param_path}_depth", depth_choices)

    # Sample encoder layers with decreasing constraint
    layers = []
    for i in range(depth):
        if i == 0:
            # First layer: full range
            layer_size = trial.suggest_int(
                f"{param_path}_layer_{i}",
                layer_min,
                layer_max,
                step=layer_step
            )
        else:
            # Subsequent layers: constrained by previous layer * gain
            prev_layer = layers[i - 1]
            constrained_max_raw = int(prev_layer * gain)

            # Round down to nearest valid step
            constrained_max = (constrained_max_raw // layer_step) * layer_step

            # Ensure constrained_max is within valid range
            constrained_max = min(constrained_max, layer_max)

            # Check if we have a valid range that satisfies the constraint
            if constrained_max < layer_min or constrained_max == 0:
                # Constraint cannot be satisfied with current minimum step
                # Use the raw constrained value (before rounding)
                if constrained_max_raw > 0:
                    layer_size = constrained_max_raw
                else:
                    # Shouldn't happen with valid gain, but use minimum step as fallback
                    layer_size = layer_step
            else:
                layer_size = trial.suggest_int(
                    f"{param_path}_layer_{i}",
                    layer_min,
                    constrained_max,
                    step=layer_step
                )

        layers.append(layer_size)

    return layers


def verify_constraints(encoder_layers, gain):
    """Verify that encoder layers satisfy gain constraints."""
    for i in range(1, len(encoder_layers)):
        ratio = encoder_layers[i] / encoder_layers[i-1]
        if ratio > gain + 0.001:  # Small epsilon for floating point
            return False, f"Layer {i}: {encoder_layers[i]}/{encoder_layers[i-1]} = {ratio:.3f} > {gain}"
    return True, "OK"


def main():
    print("\n" + "=" * 70)
    print("HPO LAYER SEQUENCE SAMPLING TEST")
    print("=" * 70 + "\n")

    # Test parameters matching user requirements
    depth_choices = [2, 3]
    layer_min = 16
    layer_max = 128
    layer_step = 16
    gain = 0.5

    print("Configuration:")
    print(f"  Depth choices: {depth_choices}")
    print(f"  Layer size range: {layer_min} to {layer_max} (step: {layer_step})")
    print(f"  Gain constraint: {gain} (each layer <= previous * {gain})")
    print()

    # Create study
    study = optuna.create_study(direction='minimize', sampler=TPESampler())

    print("Sampling 20 layer configurations:\n")

    valid_count = 0
    invalid_count = 0

    for i in range(20):
        trial = study.ask()

        # Sample encoder layers
        encoder_layers = sample_layer_sequence(
            trial,
            'encoder',
            depth_choices,
            layer_min,
            layer_max,
            layer_step,
            gain
        )

        # Decoder mirrors encoder
        decoder_layers = list(reversed(encoder_layers))

        # Verify constraints
        is_valid, message = verify_constraints(encoder_layers, gain)

        status = "✓" if is_valid else "✗"
        print(f"Trial {i+1:2d} {status}: Depth={len(encoder_layers)}, "
              f"Encoder={encoder_layers}, Decoder={decoder_layers}")

        if not is_valid:
            print(f"          CONSTRAINT VIOLATION: {message}")
            invalid_count += 1
        else:
            valid_count += 1

        # Tell the study this trial is complete
        study.tell(trial, 1.0)

    print("\n" + "=" * 70)
    print(f"Results: {valid_count} valid, {invalid_count} invalid")

    if invalid_count == 0:
        print("✓ ALL SAMPLES SATISFY CONSTRAINTS!")
    else:
        print(f"⚠ WARNING: {invalid_count} samples violated constraints!")

    print("=" * 70)

    # Show specific examples
    print("\nExpected behavior (from requirements):")
    print("  Valid:   [64, 32] (0.50), [96, 32] (0.33), [128, 64] (0.50)")
    print("  Invalid: [64, 48] (0.75), [96, 128] (>1.0), [128, 96] (0.75)")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
