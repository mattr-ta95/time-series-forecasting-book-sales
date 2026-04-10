"""Tests for model functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import tensorflow as tf

from models import (
    build_encoder_decoder_model,
    compute_metrics,
    create_hybrid_lstm_model,
    create_input_output_sequences_xgb,
    create_input_sequences_lstm,
    create_lag_features,
    create_lstm_model,
    create_simple_parallel_lstm,
)


class TestSequenceCreation:
    def test_lstm_sequences_shape(self):
        """LSTM sequence creation produces correct shapes."""
        data = np.random.randn(100, 1)
        result = create_input_sequences_lstm(lookback=10, forecast=5, sequence_data=data)
        X = np.array(result["input_sequences"])
        y = np.array(result["output_sequences"])
        assert X.shape[1] == 10  # lookback
        assert y.shape[1] == 5   # forecast
        assert len(X) == len(y)
        assert len(X) == 100 - 10 - 5 + 1  # expected count

    def test_xgb_sequences_shape(self):
        """XGBoost sequence creation produces correct shapes."""
        data = np.random.randn(50).tolist()
        X, y = create_input_output_sequences_xgb(lookback=12, forecast=1, sequence_data=data)
        assert len(X) == len(y)
        assert len(X[0]) == 12
        assert len(y[0]) == 1

    def test_lag_features(self):
        """Lag feature creation adds correct number of columns."""
        data = np.arange(20).astype(float)
        df = create_lag_features(data, lag=3)
        assert 'lag_1' in df.columns
        assert 'lag_2' in df.columns
        assert 'lag_3' in df.columns
        assert df.isna().sum().sum() == 0  # no NaN after dropna


class TestModelBuilders:
    def test_lstm_model_architecture(self):
        """LSTM model has expected layer count and output shape."""
        model = create_lstm_model(nodes=50, lookback=52, forecast=32)
        assert model.output_shape == (None, 32)
        # 2 LSTM + 2 Dropout + 1 Dense = 5 layers
        assert len(model.layers) == 5

    def test_hybrid_lstm_model(self):
        """Hybrid LSTM outputs correct forecast steps."""
        model = create_hybrid_lstm_model(forecast_steps=32)
        assert model.output_shape == (None, 32)

    def test_simple_parallel_lstm(self):
        """Parallel hybrid LSTM outputs single value."""
        model = create_simple_parallel_lstm()
        assert model.output_shape == (None, 1)

    def test_lstm_model_reduced_params(self):
        """Phase 2 model has fewer parameters than Phase 1 would."""
        model = create_lstm_model(nodes=50, lookback=52, forecast=32)
        param_count = model.count_params()
        # Phase 1 had ~200K params (5 layers x 80 units)
        # Phase 2 should be ~50K or less
        assert param_count < 100_000, f"Model has {param_count} params, expected <100K"


class TestMetrics:
    def test_compute_metrics_keys(self):
        """compute_metrics returns expected keys."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        metrics = compute_metrics(actual, predicted)
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'rmse' in metrics

    def test_compute_metrics_perfect(self):
        """Perfect predictions yield zero error."""
        actual = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(actual, actual)
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0

    def test_compute_metrics_mismatched_lengths(self):
        """compute_metrics handles different-length arrays."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1, 2, 3])  # shorter
        metrics = compute_metrics(actual, predicted)
        assert metrics['mae'] == 0.0  # first 3 match exactly


class TestEncoderDecoderLSTM:
    """Tests for the Functional-API encoder-decoder LSTM with attention."""

    def test_encoder_decoder_architecture(self):
        """Model has sane output shape, param budget, and encoder+decoder LSTMs."""
        model = build_encoder_decoder_model(
            lookback=52, horizon=32, encoder_units=64, decoder_units=64,
        )
        assert model.output_shape == (None, 32)
        assert model.count_params() < 150_000, (
            f"Model has {model.count_params()} params, expected <150K"
        )

        layer_names = [layer.name for layer in model.layers]
        assert 'encoder_lstm' in layer_names, "expected an encoder LSTM layer"
        assert 'decoder_lstm' in layer_names, "expected a decoder LSTM layer"

        lstm_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)
        ]
        assert len(lstm_layers) >= 2, (
            f"expected at least 2 LSTM layers (encoder+decoder), found {len(lstm_layers)}"
        )

    def test_encoder_decoder_uses_functional_api(self):
        """Model must be built with the Functional API, not Sequential."""
        model = build_encoder_decoder_model(
            lookback=52, horizon=32, encoder_units=64, decoder_units=64,
        )
        assert isinstance(model, tf.keras.Model)
        assert not isinstance(model, tf.keras.Sequential)

    def test_encoder_decoder_has_attention(self):
        """Model must contain a Keras Attention layer over encoder outputs."""
        model = build_encoder_decoder_model(
            lookback=52, horizon=32, encoder_units=64, decoder_units=64,
        )
        attention_layers = [
            layer for layer in model.layers
            if isinstance(
                layer,
                (tf.keras.layers.Attention, tf.keras.layers.AdditiveAttention),
            )
        ]
        assert len(attention_layers) >= 1, (
            "expected at least one Attention / AdditiveAttention layer"
        )

    def test_encoder_decoder_trains_one_step(self):
        """Model fits one epoch on tiny synthetic data and produces a finite loss."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal(size=(64, 52, 1)).astype(np.float32)
        y = rng.standard_normal(size=(64, 32)).astype(np.float32)

        model = build_encoder_decoder_model(
            lookback=52, horizon=32, encoder_units=32, decoder_units=32,
        )
        history = model.fit(X, y, epochs=1, batch_size=16, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1
        assert np.isfinite(history.history['loss'][0])
