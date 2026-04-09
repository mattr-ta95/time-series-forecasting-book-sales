"""Tests for model functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from models import (
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
