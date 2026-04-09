"""Shared test fixtures."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SALES_FILE = os.path.join(DATA_DIR, 'UK_Weekly_Trended_Timeline_from_200101_202429.xlsx')

requires_data = pytest.mark.skipif(
    not os.path.exists(SALES_FILE),
    reason="Excel data files not available",
)


@pytest.fixture
def synthetic_weekly_series():
    """Small deterministic weekly time series for unit testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2015-01-04', periods=n, freq='W')
    trend = np.linspace(100, 200, n)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise = np.random.normal(0, 5, n)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name='Volume')


@pytest.fixture
def synthetic_train_test(synthetic_weekly_series):
    """Split synthetic series into train/test (last 32 held out)."""
    horizon = 32
    train = synthetic_weekly_series.iloc[:-horizon]
    test = synthetic_weekly_series.iloc[-horizon:]
    return train, test
