"""Model definitions, training functions, and evaluation utilities."""

import joblib
import numpy as np
import optuna
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from pmdarima.arima import auto_arima
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
)
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from config import (
    EARLY_STOP_PATIENCE,
    FORECAST_HORIZON_WEEKS,
    HYBRID_EPOCHS,
    HYBRID_PATIENCE,
    HYPEROPT_MAX_EVALS,
    LOOKBACK_WEEKS,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_UNITS,
    MONTHLY_XGB_LEARNING_RATE,
    MONTHLY_XGB_MAX_DEPTH,
    MONTHLY_XGB_N_ESTIMATORS,
    N_JOBS,
    OPTUNA_EPOCHS,
    OPTUNA_RETRAIN_EPOCHS,
    OPTUNA_TRIALS,
    PARALLEL_SARIMA_WEIGHT,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
)


# ─── Sequence creation utilities ───────────────────────────────────────


def create_input_sequences_lstm(lookback, forecast, sequence_data):
    """Create input/output sequence pairs for LSTM models."""
    input_sequences = []
    output_sequences = []
    for i in range(lookback, len(sequence_data) - forecast + 1):
        input_sequences.append(sequence_data[i - lookback:i])
        output_sequences.append(sequence_data[i:i + forecast])
    return {"input_sequences": input_sequences, "output_sequences": output_sequences}


def create_input_output_sequences_xgb(lookback, forecast, sequence_data):
    """Create input/output pairs for XGBoost."""
    input_sequences = []
    output_sequences = []
    for i in range(lookback, len(sequence_data) - forecast + 1):
        input_sequences.append(sequence_data[i - lookback:i])
        output_sequences.append(sequence_data[i:i + forecast])
    return input_sequences, output_sequences


def create_lag_features(y, lag=1):
    """Create lag features from a time series."""
    df = pd.DataFrame(y)
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df[0].shift(i)
    df.dropna(inplace=True)
    return df


def create_lstm_dataset(dataset, look_back=1):
    """Create dataset for single-step LSTM (parallel hybrid)."""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# ─── Model builders ───────────────────────────────────────────────────


def create_lstm_model(nodes, lookback, forecast):
    """Simplified LSTM architecture (Phase 2: 2 layers with dropout)."""
    model = Sequential()
    model.add(LSTM(units=nodes, input_shape=(lookback, 1), return_sequences=True))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(LSTM(units=nodes))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(Dense(forecast))
    model.compile(loss='mse', optimizer='adam')
    return model


def build_lstm_model_with_params(units, dropout_rate, forecast_steps=FORECAST_HORIZON_WEEKS):
    """Build LSTM with given hyperparameters for multi-step forecasting."""
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(LOOKBACK_WEEKS, 1), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(forecast_steps))
    model.compile(loss='mse', optimizer='adam')
    return model


def create_hybrid_lstm_model(forecast_steps=FORECAST_HORIZON_WEEKS):
    """LSTM for forecasting residuals (multi-step output)."""
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS, input_shape=(LOOKBACK_WEEKS, 1), return_sequences=True))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(LSTM(units=LSTM_UNITS))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(Dense(forecast_steps))
    model.compile(loss='mse', optimizer='adam')
    return model


def create_simple_parallel_lstm():
    """Simple single-step LSTM for parallel hybrid model."""
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS, activation='relu', input_shape=(LOOKBACK_WEEKS, 1)))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# ─── XGBoost helpers ──────────────────────────────────────────────────


def xgboost_train(train_inputs, train_outputs):
    """Train XGBoost model with default parameters."""
    train_inputs = np.asarray(train_inputs)
    train_outputs = np.asarray(train_outputs).flatten()
    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS, min_child_weight=1, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, booster='gbtree', tree_method='exact',
        reg_alpha=0, subsample=0.5, validate_parameters=1,
        colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0,
    )
    model.fit(train_inputs, train_outputs)
    return model


def xgboost_predictions(model, test_input):
    """Predict with fitted XGBoost model."""
    return model.predict(np.asarray([test_input]))[0]


def walk_forward_validation(train_set, test_set):
    """Walk-forward validation for XGBoost."""
    predictions = []
    train_input, train_output = create_input_output_sequences_xgb(12, 1, train_set)
    test_input, test_output = create_input_output_sequences_xgb(12, 1, test_set)
    model = xgboost_train(train_input, train_output)
    for i in range(len(test_input)):
        prediction = xgboost_predictions(model, test_input[i])
        predictions.append(prediction)
        print('>expected=%.1f, predicted=%.1f' % (test_output[i][0], prediction))
    error = mean_absolute_error(np.asarray(test_output).flatten(), predictions)
    return error, np.asarray(test_output).flatten(), predictions


def create_predictor_with_deseasonaliser_xgboost(
    sp=12,
    degree=1,
    max_depth=XGB_MAX_DEPTH,
    gamma=0,
    reg_alpha=0,
    min_child_weight=1,
    colsample_bytree=1,
    n_estimators=600,
):
    """Create sktime pipeline with detrending and deseasonalization.

    Accepts tuned XGB hyperparameters so Hyperopt's best params can be
    threaded through to the internal XGBRegressor. Defaults preserve the
    original un-tuned behaviour for backward compatibility.
    """
    regressor = XGBRegressor(
        base_score=0.5,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        max_depth=max_depth,
        learning_rate=XGB_LEARNING_RATE,
        booster='gbtree',
        tree_method='exact',
        reg_alpha=reg_alpha,
        subsample=0.5,
        validate_parameters=1,
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
    )
    return TransformedTargetForecaster([
        ("deseasonalize", Deseasonalizer(model="additive", sp=sp)),
        ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=degree))),
        ("forecast", make_reduction(regressor, window_length=12, strategy="recursive")),
    ])


def grid_search_predictor(train, test, predictor, param_grid):
    """Grid search with cross-validation for sktime forecaster."""
    cv = ExpandingWindowSplitter(initial_window=int(len(train) * 0.7))
    gscv = ForecastingGridSearchCV(
        predictor, strategy="refit", cv=cv, param_grid=param_grid,
        scoring=MeanAbsolutePercentageError(symmetric=True),
        error_score="raise", refit=True, verbose=1,
    )
    gscv.fit(train)
    print(f"Best parameters: {gscv.best_params_}")
    future_horizon = np.arange(len(test)) + 1
    return gscv.predict(fh=future_horizon)


# ─── Evaluation ───────────────────────────────────────────────────────


def compute_metrics(actual, predicted):
    """Compute MAE, MAPE, RMSE between actual and predicted."""
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    min_len = min(len(actual), len(predicted))
    actual, predicted = actual[:min_len], predicted[:min_len]
    return {
        'mae': mean_absolute_error(actual, predicted),
        'mape': mean_absolute_percentage_error(actual, predicted),
        'rmse': np.sqrt(mean_squared_error(actual, predicted)),
    }


def time_series_cv_score(model_func, X, y, n_splits=3):
    """Time series cross-validation with expanding window."""
    n_samples = len(X)
    test_size = n_samples // (n_splits + 1)
    scores = []
    print(f"  Running {n_splits}-fold time series CV...")
    for i in range(n_splits):
        train_end = test_size * (i + 2)
        test_start = test_size * (i + 1)
        test_end = train_end
        X_train_cv, y_train_cv = X[:test_start], y[:test_start]
        X_val_cv, y_val_cv = X[test_start:test_end], y[test_start:test_end]
        if len(X_val_cv) == 0:
            continue
        model = model_func()
        early_stop = EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True, verbose=0)
        model.fit(X_train_cv, y_train_cv, epochs=50, batch_size=32,
                  validation_split=0.2, callbacks=[early_stop], verbose=0)
        pred = model.predict(X_val_cv, verbose=0)
        mae = mean_absolute_error(y_val_cv.flatten(), pred.flatten())
        scores.append(mae)
        print(f"    Fold {i+1}: MAE = {mae:.2f}")
    return scores


# ─── Model runners ────────────────────────────────────────────────────


def run_auto_arima(train, horizon=FORECAST_HORIZON_WEEKS, m=52, n_jobs=N_JOBS):
    """Fit auto_arima and return forecast, fitted values, and residuals."""
    model = auto_arima(
        y=train, X=None,
        start_p=0, max_p=5, d=None, D=None,
        start_q=0, max_q=5, start_P=0, max_P=2, start_Q=0, max_Q=2,
        m=m, seasonal=True, information_criterion='aic', alpha=0.05,
        stepwise=False, suppress_warnings=True, error_action='ignore',
        trace=True, random=True, scoring='mse', enforce_stationarity=True,
        n_jobs=n_jobs,
    )
    model.summary()
    predictions = model.predict(horizon, return_conf_int=True, alpha=0.05)
    fitted = model.fittedvalues()
    residuals = train - fitted
    return {
        'model': model,
        'predictions': predictions[0],
        'conf_int': predictions[1],
        'fitted': fitted,
        'residuals': residuals,
    }


def run_xgboost(train, test, book_name):
    """Run XGBoost pipeline with Hyperopt tuning and sktime grid search."""
    train_values = train.values

    # Hyperopt tuning
    param_space = {
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform('gamma', 0, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0, 10),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': XGB_N_ESTIMATORS,
        'seed': 0,
    }

    def auto_tune(params):
        train_set = train_values
        test_set = train_values[-FORECAST_HORIZON_WEEKS:]
        train_input, train_output = create_input_output_sequences_xgb(12, 1, train_set)
        test_input, test_output = create_input_output_sequences_xgb(12, 1, test_set)
        model = XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=int(params['max_depth']),
            gamma=params['gamma'],
            reg_alpha=float(params['reg_alpha']),
            min_child_weight=int(params['min_child_weight']),
            colsample_bytree=float(params['colsample_bytree']),
            eval_metric="rmse", early_stopping_rounds=10,
        )
        evaluation = [(train_input, train_output), (test_input, test_output)]
        model.fit(train_input, train_output, eval_set=evaluation, verbose=False)
        pred = model.predict(test_input)
        accuracy = mean_absolute_error(test_output, pred)
        print("MAE:", accuracy)
        return {'loss': accuracy, 'status': STATUS_OK}

    trials = Trials()
    best_raw = fmin(fn=auto_tune, space=param_space, algo=tpe.suggest,
                    max_evals=HYPEROPT_MAX_EVALS, trials=trials)

    # Translate Hyperopt's best raw params into the kwargs expected by
    # create_predictor_with_deseasonaliser_xgboost. fmin returns only the
    # tuned values, so pull n_estimators from the fixed slot in param_space.
    best_params = {
        'max_depth': int(best_raw['max_depth']),
        'gamma': float(best_raw['gamma']),
        'reg_alpha': float(best_raw['reg_alpha']),
        'min_child_weight': int(best_raw['min_child_weight']),
        'colsample_bytree': float(best_raw['colsample_bytree']),
        'n_estimators': int(param_space['n_estimators']),
    }

    # Persist best params so subsequent runs can skip tuning.
    params_file = f'best_xgb_params_{book_name}.pkl'
    joblib.dump(best_params, params_file)

    # Copy to avoid mutating the original index
    train_period = train.copy()
    train_period.index = train_period.index.to_period('W')

    predictor = create_predictor_with_deseasonaliser_xgboost(**best_params)
    predictor.fit(train_period)
    future_horizon = np.arange(len(test)) + 1
    predictions = predictor.predict(fh=future_horizon)

    return {'predictions': predictions}


def run_lstm(train, test, lookback=LOOKBACK_WEEKS, horizon=FORECAST_HORIZON_WEEKS):
    """Train LSTM model and predict on test data with cross-validation."""
    train_lstm = train.values.reshape(-1, 1)
    test_lstm = test.values.reshape(-1, 1)

    # Create sequences
    seq_data = create_input_sequences_lstm(lookback, horizon, sequence_data=train_lstm)
    X_train = np.array(seq_data["input_sequences"])
    y_train = np.array(seq_data["output_sequences"])

    # Scale with consistent scaler for input and output
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

    # Build and train
    model = create_lstm_model(LSTM_UNITS, lookback, horizon)
    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE,
                               restore_best_weights=True, verbose=1)
    model.fit(X_train_scaled, y_train_scaled, epochs=LSTM_EPOCHS,
              batch_size=LSTM_BATCH_SIZE, validation_split=0.2,
              callbacks=[early_stop], verbose=0)

    # Create test sequences and predict on TEST data
    full_data = np.concatenate([train_lstm, test_lstm])
    seq_data_test = create_input_sequences_lstm(lookback, horizon, sequence_data=full_data)
    X_test = np.array(seq_data_test["input_sequences"])[-1:]
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    predictions_scaled = model.predict(X_test_scaled)
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    predicted_series = pd.Series(predictions[:len(test)], index=test.index)
    metrics = compute_metrics(test, predicted_series)

    print(f"LSTM Test Predictions (first 5): {predictions[:5]}")
    print(f"LSTM Test Actual (first 5): {test.values[:5]}")

    # Cross-validation for stability assessment
    print("\nCross-validation for model stability assessment")

    def create_cv_model():
        return create_lstm_model(nodes=LSTM_UNITS, lookback=lookback, forecast=horizon)

    cv_scores = time_series_cv_score(create_cv_model, X_train_scaled, y_train_scaled, n_splits=3)
    print(f"  CV MAE scores: {cv_scores}")
    print(f"  Mean CV MAE: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")

    return {
        'model': model,
        'scaler': scaler,
        'predictions': predicted_series,
        'metrics': metrics,
        'cv_scores': cv_scores,
    }


def run_optuna_lstm(train, test, lookback=LOOKBACK_WEEKS, horizon=FORECAST_HORIZON_WEEKS,
                    n_trials=OPTUNA_TRIALS):
    """Run Optuna hyperparameter search, retrain best model, predict on test."""
    train_lstm = train.values.reshape(-1, 1)

    def optuna_objective(trial):
        units = trial.suggest_int('units', 32, 128, step=16)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        model = build_lstm_model_with_params(units, dropout, forecast_steps=horizon)
        seq_data = create_input_sequences_lstm(
            lookback=lookback, forecast=horizon, sequence_data=train_lstm,
        )
        X = np.array(seq_data["input_sequences"])
        y = np.array(seq_data["output_sequences"])
        history = model.fit(X, y, epochs=OPTUNA_EPOCHS, batch_size=LSTM_BATCH_SIZE,
                            validation_split=0.2, verbose=0)
        return min(history.history['val_loss'])

    print(f"Running Optuna hyperparameter tuning ({n_trials} trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=n_trials)
    best_units = study.best_params['units']
    best_dropout = study.best_params['dropout']
    print(f"Best params: units={best_units}, dropout={best_dropout}")

    # Retrain with best params on scaled data
    print("Training Optuna-optimized model...")
    best_model = build_lstm_model_with_params(best_units, best_dropout, forecast_steps=horizon)
    seq_data = create_input_sequences_lstm(
        lookback=lookback, forecast=horizon, sequence_data=train_lstm,
    )
    X_train = np.array(seq_data["input_sequences"])
    y_train = np.array(seq_data["output_sequences"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

    best_model.fit(X_train_scaled, y_train_scaled, epochs=OPTUNA_RETRAIN_EPOCHS,
                   batch_size=32, verbose=0)

    # Create test sequence and predict on TEST data
    full_data = np.concatenate([train_lstm, test.values.reshape(-1, 1)])
    seq_test = create_input_sequences_lstm(
        lookback=lookback, forecast=horizon, sequence_data=full_data,
    )
    X_test = np.array(seq_test["input_sequences"])[-1:]
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    forecast_scaled = best_model.predict(X_test_scaled, verbose=0)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    metrics = compute_metrics(test.values, forecast)
    print(f"Optuna LSTM - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, "
          f"MAPE: {metrics['mape']:.4f}")

    return {
        'model': best_model,
        'scaler': scaler,
        'predictions': forecast,
        'metrics': metrics,
        'best_params': study.best_params,
    }


def run_sequential_hybrid(test, arima_model,
                          lookback=LOOKBACK_WEEKS, horizon=FORECAST_HORIZON_WEEKS):
    """Sequential hybrid: SARIMA base forecast + LSTM on residuals."""
    residuals = arima_model.resid()

    # Build and train hybrid model on residuals
    hybrid_model = create_hybrid_lstm_model(forecast_steps=horizon)
    residuals_reshaped = residuals.values.reshape(-1, 1)
    seq_residuals = create_input_sequences_lstm(
        lookback=lookback, forecast=horizon, sequence_data=residuals_reshaped,
    )
    X_residuals = np.array(seq_residuals["input_sequences"])
    y_residuals = np.array(seq_residuals["output_sequences"])

    early_stop = EarlyStopping(monitor='val_loss', patience=HYBRID_PATIENCE,
                               restore_best_weights=True, verbose=1)
    hybrid_model.fit(X_residuals, y_residuals, epochs=HYBRID_EPOCHS, batch_size=32,
                     validation_split=0.2, callbacks=[early_stop], verbose=0)

    # SARIMA forecast for test period
    sarima_predictions = arima_model.predict(n_periods=horizon)

    # Use last `lookback` residuals to forecast next `horizon` residuals
    last_residuals = residuals.values[-lookback:].reshape(1, lookback, 1)
    lstm_residual_forecast = hybrid_model.predict(last_residuals, verbose=0).flatten()

    # Combine SARIMA + LSTM residual forecast
    final_predictions = sarima_predictions + lstm_residual_forecast
    metrics = compute_metrics(test, final_predictions)
    print(f"Sequential Hybrid - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.4f}")

    return {
        'model': hybrid_model,
        'predictions': final_predictions,
        'metrics': metrics,
        'sarima_forecast': sarima_predictions,
        'residual_forecast': lstm_residual_forecast,
    }


def run_parallel_hybrid(train, test, arima_model,
                        horizon=FORECAST_HORIZON_WEEKS,
                        sarima_weight=PARALLEL_SARIMA_WEIGHT):
    """Parallel hybrid: weighted combination of SARIMA and LSTM forecasts."""
    # Fit SARIMAX using auto_arima's identified orders
    sarima_fit = SARIMAX(
        train, order=arima_model.order, seasonal_order=arima_model.seasonal_order,
    ).fit()
    sarima_forecast = sarima_fit.predict(start=len(train), end=len(train) + horizon - 1)

    # Scale data and prepare LSTM sequences
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    X_train, y_train = create_lstm_dataset(train_scaled, look_back=LOOKBACK_WEEKS)

    # Train single-step LSTM
    lstm_model = create_simple_parallel_lstm()
    early_stop = EarlyStopping(monitor='val_loss', patience=HYBRID_PATIENCE,
                               restore_best_weights=True, verbose=0)
    lstm_model.fit(
        X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
        epochs=HYBRID_EPOCHS, validation_split=0.2, callbacks=[early_stop], verbose=0,
    )

    # Recursive multi-step forecast
    inputs = train_scaled[-LOOKBACK_WEEKS:]
    lstm_forecast = []
    for _ in range(horizon):
        prediction = lstm_model.predict(inputs.reshape(1, LOOKBACK_WEEKS, 1))
        lstm_forecast.append(prediction[0, 0])
        inputs = np.append(inputs[1:], prediction)
    lstm_forecast = scaler.inverse_transform(
        np.array(lstm_forecast).reshape(-1, 1)
    ).flatten()

    # Weighted combination
    lstm_weight = 1 - sarima_weight
    hybrid_forecast = sarima_weight * sarima_forecast + lstm_weight * lstm_forecast
    metrics = compute_metrics(test, hybrid_forecast)

    print(f"Parallel Hybrid - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.4f}")

    # Weight search
    print("\nWeight search:")
    for w1 in np.arange(0, 1.1, 0.1):
        w2 = 1 - w1
        hf = w1 * sarima_forecast + w2 * lstm_forecast
        mae = mean_absolute_error(test, hf)
        mape = mean_absolute_percentage_error(test, hf)
        print(f"  Weights: {w1:.1f}, {w2:.1f} - MAE: {mae:.2f}, MAPE: {mape:.2f}")

    return {
        'predictions': hybrid_forecast,
        'metrics': metrics,
        'sarima_forecast': sarima_forecast,
        'lstm_forecast': lstm_forecast,
    }


def run_monthly_xgboost(monthly_train, monthly_test):
    """Train XGBoost on monthly aggregated data."""
    import xgboost as xgb

    X_train = monthly_train.index.to_series().astype(int).values.reshape(-1, 1)
    X_test = monthly_test.index.to_series().astype(int).values.reshape(-1, 1)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=MONTHLY_XGB_N_ESTIMATORS,
        max_depth=MONTHLY_XGB_MAX_DEPTH,
        learning_rate=MONTHLY_XGB_LEARNING_RATE,
        booster='gbtree', tree_method='exact',
    )
    model.fit(X_train, monthly_train.values)
    predictions = model.predict(X_test)
    metrics = compute_metrics(monthly_test.values, predictions)

    return {
        'model': model,
        'predictions': predictions,
        'metrics': metrics,
        'monthly_train': monthly_train,
        'monthly_test': monthly_test,
    }


def run_monthly_arima(monthly_train, monthly_test, n_jobs=N_JOBS):
    """Train monthly SARIMA and predict."""
    horizon = len(monthly_test)
    model = auto_arima(
        y=monthly_train, X=None,
        start_p=0, max_p=5, d=None, D=None,
        start_q=0, max_q=5, start_P=0, max_P=2, start_Q=0, max_Q=2,
        m=12, seasonal=True, information_criterion='aic', alpha=0.05,
        stepwise=False, suppress_warnings=True, error_action='ignore',
        trace=True, random=True, scoring='mse', enforce_stationarity=True,
        n_jobs=n_jobs,
    )
    predictions = model.predict(n_periods=horizon)
    metrics = compute_metrics(monthly_test, predictions)

    return {
        'model': model,
        'predictions': predictions,
        'metrics': metrics,
    }
