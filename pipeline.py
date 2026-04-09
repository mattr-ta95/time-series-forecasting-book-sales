"""Orchestration: run all models for each book, generate plots, print summary."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.graphics.api as smgraphics
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller as adf

from config import (
    FORECAST_HORIZON_WEEKS,
    MONTHLY_FORECAST_MONTHS,
    MONTHLY_TRAIN_RATIO,
    PLOT_DIR,
    PLOT_NAMES,
)
from data_loader import aggregate_monthly, prepare_all_books
from models import (
    compute_metrics,
    run_auto_arima,
    run_lstm,
    run_monthly_arima,
    run_monthly_xgboost,
    run_optuna_lstm,
    run_parallel_hybrid,
    run_sequential_hybrid,
    run_xgboost,
)


# ─── Visualization helpers ────────────────────────────────────────────


def decompose_and_plot(data, book_title, model='additive'):
    """Decomposition plot for a book's volume data."""
    volume = data['Volume'].copy()
    if model == 'multiplicative':
        volume = volume.clip(lower=1e-6)
    decomposition = sm.tsa.seasonal_decompose(volume, model=model, period=52)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1); ax1.set_ylabel('Observed')
    decomposition.trend.plot(ax=ax2); ax2.set_ylabel('Trend')
    decomposition.seasonal.plot(ax=ax3); ax3.set_ylabel('Seasonal')
    decomposition.resid.plot(ax=ax4); ax4.set_ylabel('Residual')
    plt.suptitle(f'Decomposition of Sales for {book_title}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def save_forecast_plot(test, forecast, title, filename):
    """Save a standardized forecast-vs-actual plot."""
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label='Actual')
    forecast_vals = forecast.values if hasattr(forecast, 'values') else forecast
    plt.plot(test.index, forecast_vals[:len(test)], label='Forecast')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def data_for_visualisations_lstm(original_data, predictions, forecast_horizon, train_index):
    """Prepare composed data for LSTM forecast visualization."""
    existing_data = original_data.to_frame(name='Actual')
    existing_data['Forecast'] = np.nan
    existing_data['Forecast'].iloc[-1:] = existing_data['Actual'].iloc[-1:]

    if isinstance(train_index, pd.DatetimeIndex):
        forecast_index = pd.date_range(
            start=train_index[-1] + pd.DateOffset(weeks=1),
            periods=forecast_horizon, freq='W',
        )
    elif isinstance(train_index, pd.PeriodIndex):
        forecast_index = pd.period_range(
            start=train_index[-1] + 1, periods=forecast_horizon, freq='W',
        )
    else:
        raise ValueError("train_index must be a DatetimeIndex or PeriodIndex")

    predicted_data = pd.DataFrame({'Forecast': predictions.flatten()}, index=forecast_index)
    predicted_data['Actual'] = np.nan
    return pd.concat([existing_data, predicted_data], ignore_index=False)


# ─── EDA ──────────────────────────────────────────────────────────────


def run_eda(book_df, display_name):
    """Run exploratory analysis: decomposition, ACF/PACF, ADF stationarity test."""
    decompose_and_plot(book_df, display_name)
    decompose_and_plot(book_df, display_name, model='multiplicative')

    stl = STL(book_df['Volume'], period=52)
    result = stl.fit()
    result.plot()
    plt.show()

    # ACF
    plt.figure(figsize=(20, 12))
    smgraphics.tsa.plot_acf(book_df['Volume'], lags=100)
    plt.title(f'ACF Plot for {display_name}')
    plt.show()

    acf_vals = acf(book_df['Volume'], nlags=60)
    print(acf_vals)

    # PACF
    smgraphics.tsa.plot_pacf(book_df['Volume'], lags=100)
    plt.title(f'PACF Plot for {display_name}')
    plt.show()

    # ADF stationarity test
    adf_result = adf(book_df['Volume'])
    print(f'p-value. {display_name}:', adf_result[1])


# ─── Per-book model orchestration ─────────────────────────────────────


def run_all_models(book_name, train, test, book_df, display_name):
    """Run all model types for one book and return results dict."""
    results = {}

    # 1. Auto ARIMA
    print(f"\n--- Auto ARIMA ({display_name}) ---")
    arima_result = run_auto_arima(train)
    arima_metrics = compute_metrics(test, arima_result['predictions'])
    results['auto_arima'] = arima_metrics

    # ARIMA visualization
    N_plot = 200
    time_idx = pd.date_range(start=train.index[-1], periods=FORECAST_HORIZON_WEEKS, freq='W')
    plt.figure(figsize=(14, 4))
    plt.plot(train.index[-N_plot:], train[-N_plot:])
    plt.plot(train.index[-N_plot:], arima_result['fitted'][-N_plot:], ':', c='red')
    plt.plot(time_idx, arima_result['predictions'])
    plt.plot(time_idx, arima_result['conf_int'][:, 0], ':', c='grey')
    plt.plot(time_idx, arima_result['conf_int'][:, 1], ':', c='grey')
    plt.title(f'Auto ARIMA Forecast - {display_name}')
    plt.xlabel('Time'); plt.ylabel('Sales Volume')
    plt.legend(['Observed', 'Fitted', 'Forecast', '95% CI'])
    plt.show()

    plt.plot(arima_result['residuals'])
    plt.title(f'Residuals of ARIMA Model - {display_name}')
    plt.show()

    print(f"  RMSE: {arima_metrics['rmse']:.2f}, MSE: {arima_metrics['rmse']**2:.2f}")

    # 2. XGBoost
    print(f"\n--- XGBoost ({display_name}) ---")
    xgb_result = run_xgboost(train, test, book_name)
    if xgb_result['predictions'] is not None:
        xgb_preds = xgb_result['predictions']
        xgb_preds.index = test.index
        xgb_metrics = compute_metrics(test, xgb_preds)
        results['xgboost'] = xgb_metrics
        plt.figure(figsize=(12, 6))
        plt.title(f"MAE: {xgb_metrics['mae']:.2f}, MAPE: {xgb_metrics['mape']:.3f}", size=18)
        train.plot(label="Train", color="b")
        test.plot(label="Test", color="g")
        xgb_preds.plot(label="Forecast", color="r")
        plt.legend(prop={"size": 16})
        plt.show()
        print(f"  MAE: {xgb_metrics['mae']:.2f}, MAPE: {xgb_metrics['mape']:.4f}")

    # 3. LSTM
    print(f"\n--- LSTM ({display_name}) ---")
    lstm_result = run_lstm(train, test)
    results['lstm'] = lstm_result['metrics']

    plot_key = ('lstm', book_name)
    if plot_key in PLOT_NAMES:
        save_forecast_plot(
            test, lstm_result['predictions'],
            f'LSTM Forecast vs Actual - {display_name} (FIXED)',
            PLOT_NAMES[plot_key],
        )
    print(f"  MAE: {lstm_result['metrics']['mae']:.2f}, MAPE: {lstm_result['metrics']['mape']:.4f}")

    # 4. Optuna LSTM
    print(f"\n--- Optuna LSTM ({display_name}) ---")
    optuna_result = run_optuna_lstm(train, test)
    results['optuna_lstm'] = optuna_result['metrics']

    composed = data_for_visualisations_lstm(
        train, optuna_result['predictions'], FORECAST_HORIZON_WEEKS, train.index,
    )
    composed.plot(figsize=(12, 6))
    plt.title(f'{display_name} LSTM Forecast (Optuna-optimized)')
    plot_key = ('optuna', book_name)
    if plot_key in PLOT_NAMES:
        filepath = os.path.join(PLOT_DIR, PLOT_NAMES[plot_key])
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")

    # 5. Sequential Hybrid
    print(f"\n--- Sequential Hybrid ({display_name}) ---")
    seq_hybrid = run_sequential_hybrid(test, arima_result['model'])
    results['sequential_hybrid'] = seq_hybrid['metrics']

    plot_key = ('seq_hybrid', book_name)
    if plot_key in PLOT_NAMES:
        save_forecast_plot(
            test, seq_hybrid['predictions'],
            f'Hybrid Forecast for {display_name} - Sequential (SARIMA + LSTM)',
            PLOT_NAMES[plot_key],
        )

    # 6. Parallel Hybrid
    print(f"\n--- Parallel Hybrid ({display_name}) ---")
    par_hybrid = run_parallel_hybrid(train, test, arima_result['model'])
    results['parallel_hybrid'] = par_hybrid['metrics']

    plot_key = ('par_hybrid', book_name)
    if plot_key in PLOT_NAMES:
        save_forecast_plot(
            test, par_hybrid['predictions'],
            f'Parallel Hybrid Model Forecast - {display_name} (Weighted SARIMA+LSTM)',
            PLOT_NAMES[plot_key],
        )

    # 7. Monthly XGBoost
    print(f"\n--- Monthly XGBoost ({display_name}) ---")
    monthly = aggregate_monthly(train)
    monthly_split = int(len(monthly) * MONTHLY_TRAIN_RATIO)
    monthly_train = monthly[:monthly_split]
    monthly_test_data = monthly[monthly_split:]

    monthly_xgb = run_monthly_xgboost(monthly_train, monthly_test_data)
    results['monthly_xgboost'] = monthly_xgb['metrics']

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_train.index, monthly_train, label='Training Data', color='blue')
    plt.plot(monthly_test_data.index, monthly_test_data, label='Test Data (Actual)', color='green')
    plt.plot(monthly_test_data.index, monthly_xgb['predictions'],
             label='Test Forecast', color='red', linestyle='--')
    plt.title(f'Monthly XGBoost Forecast for {display_name} (Test Set)')
    plt.xlabel('Date'); plt.ylabel('Sales Volume')
    plt.legend()
    plot_key = ('monthly_xgb', book_name)
    if plot_key in PLOT_NAMES:
        filepath = os.path.join(PLOT_DIR, PLOT_NAMES[plot_key])
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filepath}")

    print(f"  Monthly XGBoost MAE: {monthly_xgb['metrics']['mae']:.2f}, "
          f"MAPE: {monthly_xgb['metrics']['mape']:.4f}")

    # 8. Monthly ARIMA
    print(f"\n--- Monthly ARIMA ({display_name}) ---")
    full_volume = book_df['Volume']
    full_monthly = full_volume.resample('MS').sum()
    monthly_arima_split = len(full_monthly) - MONTHLY_FORECAST_MONTHS
    monthly_arima_train = full_monthly[:monthly_arima_split]
    monthly_arima_test = full_monthly[monthly_arima_split:]

    monthly_arima = run_monthly_arima(monthly_arima_train, monthly_arima_test)
    results['monthly_arima'] = monthly_arima['metrics']
    print(f"  Monthly ARIMA MAE: {monthly_arima['metrics']['mae']:.2f}, "
          f"MAPE: {monthly_arima['metrics']['mape']:.4f}")

    return results


# ─── Main pipeline ────────────────────────────────────────────────────


def print_summary(all_results):
    """Print final comparison table of all models across books."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for book_name, results in all_results.items():
        print(f"\n{book_name.upper()}:")
        print(f"  {'Model':<25} {'MAE':>10} {'MAPE':>10} {'RMSE':>10}")
        print(f"  {'-'*55}")
        for model_name, metrics in results.items():
            print(f"  {model_name:<25} {metrics['mae']:>10.2f} "
                  f"{metrics['mape']:>10.4f} {metrics['rmse']:>10.2f}")

    print("\n" + "=" * 60)
    print("ALL PLOTS SAVED TO: results/plots/")
    print("=" * 60)


def run_pipeline():
    """Main entry: load data, run all models per book, save plots, print summary."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Plots will be saved to: {PLOT_DIR}/")

    books_data = prepare_all_books()
    all_results = {}

    for book_name, data in books_data.items():
        cfg = data['config']
        print(f"\n{'='*60}")
        print(f"Processing: {cfg.display_name}")
        print(f"{'='*60}")

        run_eda(data['full_df'], cfg.display_name)

        results = run_all_models(
            book_name=book_name,
            train=data['train'],
            test=data['test'],
            book_df=data['full_df'],
            display_name=cfg.display_name,
        )
        all_results[book_name] = results

    print_summary(all_results)
