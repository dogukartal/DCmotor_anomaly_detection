"""LSTM Autoencoder model for anomaly detection."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class LSTMAutoencoder:
    """
    LSTM Autoencoder for sequence reconstruction.

    Architecture:
        Input -> [Encoder LSTM layers] -> [Optional Bottleneck Dense] -> [Decoder LSTM layers] -> Output
    """

    def __init__(self,
                 input_shape: Tuple[int, int],
                 encoder_units: List[int] = [64, 32],
                 decoder_units: List[int] = [32, 64],
                 bottleneck_units: Optional[int] = None,
                 bottleneck_activation: str = 'relu',
                 dropout: float = 0.2,
                 recurrent_dropout: float = 0.0,
                 activation: str = 'tanh',
                 recurrent_activation: str = 'sigmoid'):
        """
        Initialize LSTM Autoencoder.

        Args:
            input_shape: (sequence_length, n_features)
            encoder_units: List of units for encoder LSTM layers
            decoder_units: List of units for decoder LSTM layers
            bottleneck_units: Units for optional bottleneck dense layer (None to disable)
            bottleneck_activation: Activation for bottleneck layer
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout rate for LSTM
            activation: LSTM activation function
            recurrent_activation: LSTM recurrent activation function
        """
        self.input_shape = input_shape
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.bottleneck_units = bottleneck_units
        self.bottleneck_activation = bottleneck_activation
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.activation = activation
        self.recurrent_activation = recurrent_activation

        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def build(self) -> keras.Model:
        """
        Build the autoencoder model.

        Returns:
            Keras Model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')

        # Encoder
        x = inputs
        for i, units in enumerate(self.encoder_units):
            return_sequences = (i < len(self.encoder_units) - 1) or (self.bottleneck_units is not None)
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                recurrent_dropout=self.recurrent_dropout,
                name=f'encoder_lstm_{i}'
            )(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout, name=f'encoder_dropout_{i}')(x)

        # Optional bottleneck
        if self.bottleneck_units is not None:
            # If encoder returns sequences, take last timestep
            if len(self.encoder_units) > 0:
                x = layers.Lambda(lambda t: t[:, -1, :], name='encoder_last_step')(x)

            x = layers.Dense(
                self.bottleneck_units,
                activation=self.bottleneck_activation,
                name='bottleneck'
            )(x)

            if self.dropout > 0:
                x = layers.Dropout(self.dropout, name='bottleneck_dropout')(x)

        # Repeat for decoder
        x = layers.RepeatVector(self.input_shape[0], name='repeat_vector')(x)

        # Decoder
        for i, units in enumerate(self.decoder_units):
            return_sequences = True  # Always return sequences for decoder
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                recurrent_dropout=self.recurrent_dropout,
                name=f'decoder_lstm_{i}'
            )(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout, name=f'decoder_dropout_{i}')(x)

        # Output layer (TimeDistributed Dense to match input features)
        outputs = layers.TimeDistributed(
            layers.Dense(self.input_shape[1]),
            name='output'
        )(x)

        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_autoencoder')

        return self.model

    def compile(self,
                optimizer: str = 'adam',
                learning_rate: float = 0.001,
                loss: str = 'mse',
                metrics: Optional[List[str]] = None) -> None:
        """
        Compile the model.

        Args:
            optimizer: Optimizer name or optimizer instance
            learning_rate: Learning rate (only used if optimizer is a string)
            loss: Loss function (string name or callable loss function object)
            metrics: List of metric names
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        # Create optimizer (if string is provided)
        if isinstance(optimizer, str):
            if optimizer == 'adam':
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                opt = keras.optimizers.SGD(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            # Assume it's already an optimizer instance
            opt = optimizer

        # Compile (loss can be string or callable)
        self.model.compile(
            optimizer=opt,
            loss=loss,  # Can now accept both string ('mse') or callable (PhysicsInformedLoss)
            metrics=metrics or []
        )

    def get_encoder(self) -> keras.Model:
        """
        Extract encoder model.

        Returns:
            Encoder model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        if self.encoder_model is None:
            # Find bottleneck or last encoder layer
            if self.bottleneck_units is not None:
                encoder_output_layer = self.model.get_layer('bottleneck')
            else:
                encoder_output_layer = self.model.get_layer(f'encoder_lstm_{len(self.encoder_units)-1}')

            self.encoder_model = keras.Model(
                inputs=self.model.input,
                outputs=encoder_output_layer.output,
                name='encoder'
            )

        return self.encoder_model

    def get_decoder(self) -> keras.Model:
        """
        Extract decoder model.

        Returns:
            Decoder model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        if self.decoder_model is None:
            # Find first decoder layer
            if self.bottleneck_units is not None:
                decoder_input_layer = self.model.get_layer('repeat_vector')
            else:
                decoder_input_layer = self.model.get_layer('decoder_lstm_0')

            # Create decoder input
            if self.bottleneck_units is not None:
                decoder_input = layers.Input(shape=(self.bottleneck_units,))
                x = layers.RepeatVector(self.input_shape[0])(decoder_input)
            else:
                decoder_input = layers.Input(shape=(self.input_shape[0], self.encoder_units[-1]))
                x = decoder_input

            # Rebuild decoder layers
            for i, units in enumerate(self.decoder_units):
                x = layers.LSTM(
                    units,
                    return_sequences=True,
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    recurrent_dropout=self.recurrent_dropout,
                    name=f'decoder_lstm_{i}'
                )(x)
                if self.dropout > 0:
                    x = layers.Dropout(self.dropout)(x)

            x = layers.TimeDistributed(layers.Dense(self.input_shape[1]))(x)

            self.decoder_model = keras.Model(
                inputs=decoder_input,
                outputs=x,
                name='decoder'
            )

        return self.decoder_model

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save directory
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        self.model.save(filepath)

    def load(self, filepath: str, custom_objects: Optional[Dict[str, Any]] = None) -> 'LSTMAutoencoder':
        """
        Load model from file.

        Args:
            filepath: Path to saved model directory
            custom_objects: Dictionary of custom objects (e.g., custom loss functions)

        Returns:
            Self for method chaining
        """
        # Import here to avoid circular dependency
        from ..models.physics_loss import PhysicsInformedLoss

        # Default custom objects
        default_custom_objects = {
            'PhysicsInformedLoss': PhysicsInformedLoss
        }

        # Merge with provided custom objects
        if custom_objects:
            default_custom_objects.update(custom_objects)

        self.model = keras.models.load_model(filepath, custom_objects=default_custom_objects)
        return self

    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        self.model.summary()

    @classmethod
    def from_config(cls, config: Dict[str, Any], input_shape: Tuple[int, int]) -> 'LSTMAutoencoder':
        """
        Create LSTMAutoencoder from configuration.

        Args:
            config: Configuration dictionary
            input_shape: (sequence_length, n_features)

        Returns:
            LSTMAutoencoder instance
        """
        model_config = config.get('model', {})

        encoder_units = model_config.get('encoder_units', [64, 32])
        decoder_units = model_config.get('decoder_units', [32, 64])

        # Bottleneck configuration
        bottleneck_config = model_config.get('bottleneck', {})
        bottleneck_enabled = bottleneck_config.get('enabled', True)
        bottleneck_units = bottleneck_config.get('units', 16) if bottleneck_enabled else None
        bottleneck_activation = bottleneck_config.get('activation', 'relu')

        dropout = model_config.get('dropout', 0.2)
        recurrent_dropout = model_config.get('recurrent_dropout', 0.0)
        activation = model_config.get('activation', 'tanh')
        recurrent_activation = model_config.get('recurrent_activation', 'sigmoid')

        autoencoder = cls(
            input_shape=input_shape,
            encoder_units=encoder_units,
            decoder_units=decoder_units,
            bottleneck_units=bottleneck_units,
            bottleneck_activation=bottleneck_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            activation=activation,
            recurrent_activation=recurrent_activation
        )

        return autoencoder
