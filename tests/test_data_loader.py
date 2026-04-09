"""Tests for data loading and preprocessing."""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.conftest import requires_data
from data_loader import aggregate_monthly, extract_book_series, split_train_test


def test_split_train_test_sizes(synthetic_weekly_series):
    """Train/test split produces correct sizes."""
    horizon = 32
    train, test = split_train_test(synthetic_weekly_series, horizon=horizon)
    assert len(test) == horizon
    assert len(train) == len(synthetic_weekly_series) - horizon
    assert len(train) + len(test) == len(synthetic_weekly_series)


def test_split_train_test_no_overlap(synthetic_weekly_series):
    """Train and test indices don't overlap."""
    train, test = split_train_test(synthetic_weekly_series, horizon=32)
    assert train.index.max() < test.index.min()


def test_aggregate_monthly(synthetic_weekly_series):
    """Monthly aggregation produces fewer periods than weekly."""
    monthly = aggregate_monthly(synthetic_weekly_series)
    assert len(monthly) < len(synthetic_weekly_series)
    assert isinstance(monthly.index, pd.DatetimeIndex)
    assert monthly.index.freqstr == 'MS'


def test_aggregate_monthly_preserves_total(synthetic_weekly_series):
    """Monthly aggregation preserves total volume (sum)."""
    monthly = aggregate_monthly(synthetic_weekly_series)
    np.testing.assert_almost_equal(monthly.sum(), synthetic_weekly_series.sum(), decimal=0)


def test_aggregate_monthly_handles_period_index(synthetic_weekly_series):
    """Monthly aggregation works when input has PeriodIndex."""
    period_series = synthetic_weekly_series.copy()
    period_series.index = period_series.index.to_period('W')
    monthly = aggregate_monthly(period_series)
    assert isinstance(monthly.index, pd.DatetimeIndex)


@requires_data
def test_load_and_extract_books():
    """Integration test: load real data and extract both books."""
    from data_loader import load_raw_data, preprocess_weekly

    df_weekly, _ = load_raw_data()
    df_full = preprocess_weekly(df_weekly)

    for title in ['Alchemist, The', 'Very Hungry Caterpillar, The']:
        book = extract_book_series(df_full, title)
        assert len(book) > 100, f"{title} should have >100 weekly observations"
        assert 'Volume' in book.columns
        assert book.index.min() > pd.Timestamp('2012-01-01')
