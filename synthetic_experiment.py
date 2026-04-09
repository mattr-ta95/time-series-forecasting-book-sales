#!/usr/bin/env python3
"""
Synthetic Data Experiment: Does more data make LSTM competitive with ARIMA?

Generates synthetic book sales data at multiple scales (2K, 5K, 10K points)
preserving the statistical properties of real Caterpillar data, then trains
the same LSTM architecture on each scale to measure the effect of data volume.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from statsmodels.tsa.seasonal import STL

from config import (
    EARLY_STOP_PATIENCE, FORECAST_HORIZON_WEEKS, LOOKBACK_WEEKS,
    LSTM_BATCH_SIZE, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_UNITS, PLOT_DIR,
)
from data_loader import prepare_all_books
from models import (
    compute_metrics, create_input_sequences_lstm, create_lstm_model,
    run_auto_arima,
)


def decompose_real_data(series):
    """Extract trend, seasonal, and residual components from real data via STL."""
    stl = STL(series, period=52, robust=True)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid


def generate_synthetic_data(real_series, n_points, seed=42):
    """Generate synthetic weekly sales data preserving real data's statistical properties.

    Uses STL decomposition of real data to extract components, then synthesizes
    a longer series by tiling the seasonal pattern, extending the trend, and
    sampling noise from the residual distribution.

    Args:
        real_series: pd.Series of real weekly sales (Volume).
        n_points: Number of synthetic data points to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.Series with DatetimeIndex at weekly frequency.
    """
    rng = np.random.RandomState(seed)

    trend, seasonal, residual = decompose_real_data(real_series)

    # Extract one full seasonal cycle (52 weeks)
    # Use the median seasonal pattern to avoid edge effects
    n_full_cycles = len(seasonal) // 52
    seasonal_matrix = seasonal.values[:n_full_cycles * 52].reshape(n_full_cycles, 52)
    base_seasonal = np.median(seasonal_matrix, axis=0)

    # Tile seasonal pattern to desired length
    n_tiles = (n_points // 52) + 1
    seasonal_long = np.tile(base_seasonal, n_tiles)[:n_points]

    # Add slow amplitude drift (±20% over the full series)
    amplitude_drift = 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(n_points) / (52 * 20))
    seasonal_long = seasonal_long * amplitude_drift

    # Extend trend linearly using the real data's trend slope
    trend_clean = trend.dropna()
    slope = (trend_clean.iloc[-1] - trend_clean.iloc[0]) / len(trend_clean)
    intercept = trend_clean.iloc[len(trend_clean) // 2]
    mid = n_points // 2
    trend_long = intercept + slope * (np.arange(n_points) - mid)

    # Sample residuals from fitted distribution
    resid_std = residual.dropna().std()
    resid_mean = residual.dropna().mean()
    noise = rng.normal(resid_mean, resid_std, n_points)

    # Combine components and clip to non-negative
    synthetic = trend_long + seasonal_long + noise
    synthetic = np.clip(synthetic, 0, None)

    # Add occasional spikes (Christmas-like, ~week 51-52 each year)
    for year_start in range(0, n_points, 52):
        spike_week = year_start + 51
        if spike_week < n_points:
            spike_magnitude = rng.uniform(1.5, 3.0) * np.abs(base_seasonal).max()
            synthetic[spike_week] += spike_magnitude
            if spike_week - 1 >= 0:
                synthetic[spike_week - 1] += spike_magnitude * 0.5

    # Create DatetimeIndex
    start_date = pd.Timestamp('1900-01-07')  # Arbitrary start for synthetic
    dates = pd.date_range(start=start_date, periods=n_points, freq='W')

    return pd.Series(synthetic, index=dates, name='Volume')


def train_and_evaluate_lstm(train, test, label=""):
    """Train LSTM on train data and evaluate on test, returning metrics dict.

    Uses the same architecture, scaling, and training config as the main pipeline.
    """
    train_arr = train.values.reshape(-1, 1)
    test_arr = test.values.reshape(-1, 1)

    # Create sequences
    lookback = LOOKBACK_WEEKS
    horizon = FORECAST_HORIZON_WEEKS
    seq_data = create_input_sequences_lstm(lookback, horizon, sequence_data=train_arr)
    X_train = np.array(seq_data["input_sequences"])
    y_train = np.array(seq_data["output_sequences"])

    n_sequences = len(X_train)
    print(f"  [{label}] {len(train)} data points → {n_sequences} training sequences")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

    # Build and train
    model = create_lstm_model(LSTM_UNITS, lookback, horizon)
    early_stop = EarlyStopping(
        monitor='val_loss', patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True, verbose=0,
    )
    model.fit(
        X_train_scaled, y_train_scaled, epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE, validation_split=0.2,
        callbacks=[early_stop], verbose=0,
    )

    # Predict on test data
    full_data = np.concatenate([train_arr, test_arr])
    seq_data_test = create_input_sequences_lstm(lookback, horizon, sequence_data=full_data)
    X_test = np.array(seq_data_test["input_sequences"])[-1:]
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    predictions_scaled = model.predict(X_test_scaled, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    metrics = compute_metrics(test.values, predictions[:len(test)])
    return metrics, predictions[:len(test)]


def run_experiment():
    """Run the full synthetic data scaling experiment."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load real Caterpillar data
    print("Loading real data...")
    books_data = prepare_all_books()
    cat_data = books_data['caterpillar']
    real_train = cat_data['train']
    real_test = cat_data['test']
    real_volume = cat_data['volume']

    print(f"\nReal data: {len(real_volume)} total, {len(real_train)} train, {len(real_test)} test")

    # Baseline 1: LSTM on real data
    print("\n" + "=" * 60)
    print("BASELINE: LSTM on real Caterpillar data")
    print("=" * 60)
    real_lstm_metrics, real_lstm_preds = train_and_evaluate_lstm(
        real_train, real_test, label="Real data",
    )
    print(f"  LSTM (real): MAE={real_lstm_metrics['mae']:.2f}, "
          f"MAPE={real_lstm_metrics['mape']:.4f}, RMSE={real_lstm_metrics['rmse']:.2f}")

    # Baseline 2: Auto ARIMA on real data (target to beat)
    print("\nBaseline: Auto ARIMA on real data...")
    arima_result = run_auto_arima(real_train)
    arima_metrics = compute_metrics(real_test, arima_result['predictions'])
    print(f"  ARIMA (real): MAE={arima_metrics['mae']:.2f}, "
          f"MAPE={arima_metrics['mape']:.4f}, RMSE={arima_metrics['rmse']:.2f}")

    # Synthetic data experiments at multiple scales
    scales = [
        ("2K", 2_084),
        ("5K", 5_084),
        ("10K", 10_084),
    ]

    results = {
        'LSTM (real data, ~600pts)': real_lstm_metrics,
        'Auto ARIMA (real data)': arima_metrics,
    }
    all_predictions = {}
    synthetic_tests = {}

    for label, n_points in scales:
        print(f"\n{'=' * 60}")
        print(f"SYNTHETIC: {label} data points ({n_points} weeks ≈ {n_points // 52} years)")
        print("=" * 60)

        # Use only training data for STL decomposition — avoid test period leakage
        synthetic = generate_synthetic_data(real_train, n_points, seed=42)
        syn_train = synthetic.iloc[:-FORECAST_HORIZON_WEEKS]
        syn_test = synthetic.iloc[-FORECAST_HORIZON_WEEKS:]

        metrics, preds = train_and_evaluate_lstm(syn_train, syn_test, label=label)
        results[f'LSTM (synthetic {label})'] = metrics
        all_predictions[label] = preds
        synthetic_tests[label] = syn_test

        print(f"  LSTM ({label}): MAE={metrics['mae']:.2f}, "
              f"MAPE={metrics['mape']:.4f}, RMSE={metrics['rmse']:.2f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS: Effect of Data Volume on LSTM Performance")
    print("=" * 70)
    print(f"  {'Model':<30} {'MAE':>10} {'MAPE':>10} {'RMSE':>10}")
    print(f"  {'-' * 60}")
    for model_name, metrics in results.items():
        print(f"  {model_name:<30} {metrics['mae']:>10.2f} "
              f"{metrics['mape']:>10.4f} {metrics['rmse']:>10.2f}")

    # Summary interpretation
    real_mape = real_lstm_metrics['mape']
    print(f"\n  LSTM MAPE on real data: {real_mape:.4f}")
    print(f"  ARIMA MAPE (target):    {arima_metrics['mape']:.4f}")
    for label, n_points in scales:
        syn_mape = results[f'LSTM (synthetic {label})']['mape']
        improvement = (real_mape - syn_mape) / real_mape * 100
        print(f"  LSTM on {label}: MAPE={syn_mape:.4f} "
              f"({'↓' if improvement > 0 else '↑'}{abs(improvement):.1f}% vs real)")

    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Effect of Training Data Volume on LSTM Forecast Quality\n'
                 '(The Very Hungry Caterpillar — Weekly Sales)', fontsize=14)

    # Plot 1: Real data LSTM vs ARIMA
    ax = axes[0, 0]
    ax.plot(real_test.index, real_test.values, 'k-', label='Actual', linewidth=2)
    ax.plot(real_test.index, real_lstm_preds, 'r--', label='LSTM (real, ~600pts)')
    ax.plot(real_test.index, arima_result['predictions'][:len(real_test)],
            'b:', label='Auto ARIMA')
    ax.set_title(f'Real Data Baselines\nLSTM MAPE={real_lstm_metrics["mape"]:.3f}, '
                 f'ARIMA MAPE={arima_metrics["mape"]:.3f}')
    ax.set_ylabel('Sales Volume')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plots 2-4: Synthetic data at each scale
    for idx, (label, n_points) in enumerate(scales):
        row, col = divmod(idx + 1, 2)
        ax = axes[row, col]
        syn_test = synthetic_tests[label]
        syn_preds = all_predictions[label]
        syn_metrics = results[f'LSTM (synthetic {label})']

        ax.plot(range(len(syn_test)), syn_test.values, 'k-', label='Actual', linewidth=2)
        ax.plot(range(len(syn_preds)), syn_preds, 'r--',
                label=f'LSTM ({label})')
        ax.set_title(f'Synthetic {label} ({n_points} weeks ≈ {n_points // 52} yrs)\n'
                     f'MAPE={syn_metrics["mape"]:.3f}')
        ax.set_ylabel('Sales Volume')
        ax.set_xlabel('Weeks ahead')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 'synthetic_experiment.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {plot_path}")

    # MAPE bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    mapes = [results[k]['mape'] for k in labels]
    colors = ['#d32f2f', '#1565c0', '#2e7d32', '#f9a825', '#7b1fa2']
    bars = ax.bar(range(len(labels)), mapes, color=colors[:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('MAPE (lower is better)')
    ax.set_title('LSTM MAPE vs Training Data Volume\n(The Very Hungry Caterpillar)')
    for bar, mape in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{mape:.3f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    bar_path = os.path.join(PLOT_DIR, 'synthetic_mape_comparison.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {bar_path}")


if __name__ == '__main__':
    run_experiment()
