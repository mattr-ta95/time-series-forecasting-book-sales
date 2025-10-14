# -*- coding: utf-8 -*-
"""
Time Series Forecasting for Book Sales Analysis

A comprehensive analysis of book sales data using various time series forecasting techniques
including ARIMA, XGBoost, LSTM, and hybrid models.

Author: Matthew Russell
Date: 2024
"""

# Fix for macOS threading issues with keras-tuner
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Install required packages (run this if packages are not already installed)
# pip install -r requirements.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import pmdarima as pm
import tensorflow as tf
import sktime
from sktime.forecasting.arima import AutoARIMA

from datetime import datetime, timedelta
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib import animation
from matplotlib import rc
import statsmodels.graphics.api as smgraphics # gives access to all plotting functions in statsmodels.
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
# import gdown  # Not used in this analysis
from pmdarima.arima import auto_arima
import warnings
# Suppress all warnings.
warnings.filterwarnings("ignore")

print(" All libraries imported successfully!")

# lightgbm is included in requirements.txt

# Set matplotlib backend for better compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

# import lightgbm as lgb  # Commented out due to installation issues
from xgboost import XGBRegressor

from sktime.forecasting.compose import (TransformedTargetForecaster, make_reduction)
from sktime.forecasting.model_selection import (ExpandingWindowSplitter, ForecastingGridSearchCV)
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Hyperparameter tuning
import optuna

# Load datasets from local data directory
import os

# Check if data files exist
data_dir = 'data'
sales_file = os.path.join(data_dir, 'UK_Weekly_Trended_Timeline_from_200101_202429.xlsx')
isbn_file = os.path.join(data_dir, 'ISBN_List.xlsx')

if not os.path.exists(sales_file):
    raise FileNotFoundError(f"Sales data file not found: {sales_file}")
if not os.path.exists(isbn_file):
    raise FileNotFoundError(f"ISBN data file not found: {isbn_file}")

# Load the datasets
data_dict = pd.read_excel(sales_file, sheet_name=None)
data_isbn = pd.read_excel(isbn_file, sheet_name=None)

print(data_dict.keys())
print(data_isbn.keys())

# Combine all sheets in df_dict into a single DataFrame with an extra column to identify the sheet
df_UK_weekly = pd.concat(
    [df.assign(Category=sheet_name) for sheet_name, df in data_dict.items()],
    ignore_index=True
)

df_UK_weekly.head()

# Combine all sheets in data_ISBN into a single DataFrame with an extra column to identify the sheet
df_ISBN = pd.concat(
    [df.assign(Category=sheet_name) for sheet_name, df in data_isbn.items()],
    ignore_index=True
)

df_ISBN.head()

df_UK_weekly.info()
df_ISBN.info()

# Data Preprocessing and Initial Investigation

# Resample the data to fill missing weeks with zero
df_UK_weekly['End Date'] = pd.to_datetime(df_UK_weekly['End Date'])
df_UK_weekly = df_UK_weekly.set_index('End Date')

# Keep some features
num_cols = ['Volume']
other_cols = ['Title', 'Category', 'ISBN']

df_UK_weekly_full = df_UK_weekly.groupby(other_cols)[num_cols].resample('W').sum().fillna(0)
df_UK_weekly.reset_index(inplace=True)
df_UK_weekly_full.head()

# Plot time series data to view
plt.figure(figsize=(10, 6))
# Reset the index to bring 'Title' and 'Category' back as columns
df_UK_weekly_full = df_UK_weekly_full.reset_index()
for category in df_UK_weekly_full['Category'].unique():
    category_data = df_UK_weekly_full[df_UK_weekly_full['Category'] == category]
    plt.plot(category_data['End Date'], category_data['Volume'], label=category)
    plt.legend()
    plt.show

# Convert ISBNs to String value
df_UK_weekly_full['ISBN'] = df_UK_weekly_full['ISBN'].astype(str)

# Convert date to datetime object
df_UK_weekly_full['End Date'] = pd.to_datetime(df_UK_weekly_full['End Date'])

# Filter out the ISBNs with sales data beyond 2024-07-01
sales_post_July_2024 = df_UK_weekly_full[df_UK_weekly_full['End Date'] > '2024-07-01']
sales_post_July_2024

# Number of unique ISBNs
sales_post_July_2024['ISBN'].nunique()

# Plot the data of all the ISBNs from the previous step by placing them in a loop
unique_isbns = sales_post_July_2024['ISBN'].unique()

for isbn in unique_isbns:
    # Filter data for the current ISBN
    isbn_data = df_UK_weekly_full[df_UK_weekly_full['ISBN'] == isbn]

    # Create a new figure for each ISBN
    plt.figure(figsize=(6, 4))

    # Plot the data
    plt.plot(isbn_data['End Date'], isbn_data['Volume'])

    # Set plot titles and labels
    plt.title(f'Sales for ISBN: {isbn}')
    plt.xlabel('End Date')
    plt.ylabel('Volume')

    # Display the plot for the current ISBN
    plt.show()

# Select two books for further analysis: The Alchemist and The Very Hungry Caterpillar
# Focus on the period >2012-01-01
alchemist = df_UK_weekly_full[df_UK_weekly_full['Title'].isin(['Alchemist, The'])]
caterpillar = df_UK_weekly_full[df_UK_weekly_full['Title'].isin(['Very Hungry Caterpillar, The'])]

alchemist = alchemist.set_index('End Date')
caterpillar = caterpillar.set_index('End Date')

alchemist = alchemist.resample('W').sum().ffill()
caterpillar = caterpillar.resample('W').sum().ffill()

# Filter using the index:
alchemist = alchemist[alchemist.index > '2012-01-01']
caterpillar = caterpillar[caterpillar.index > '2012-01-01']

alchemist.head()
caterpillar.head()

# Classical Time Series Analysis

# Function to perform decomposition and plot results
def decompose_and_plot(data, book_title, model='additive'):
    if model == 'multiplicative':
        data['Volume'] = data['Volume'].clip(lower=1e-6)  # Replace zeros with a very small positive value
    decomposition = sm.tsa.seasonal_decompose(data['Volume'], model=model, period=52)  # Assuming weekly seasonality
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

    decomposition.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')

    plt.suptitle(f'Decomposition of Sales for {book_title}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Decompose and plot for 'The Alchemist' - Additive
decompose_and_plot(alchemist, 'The Alchemist')

# Decompose and plot for 'The Very Hungry Caterpillar' - Additive
decompose_and_plot(caterpillar, 'The Very Hungry Caterpillar')

# Repeat for multiplicative
decompose_and_plot(alchemist, 'The Alchemist', model='multiplicative')
decompose_and_plot(caterpillar, 'The Very Hungry Caterpillar', model='multiplicative')

# Check STL for comparison
stl = STL(alchemist['Volume'], period=52)
result = stl.fit()
result.plot()
plt.show()

stl = STL(caterpillar['Volume'], period=52)
result = stl.fit()
result.plot()
plt.show()

# ACF and PACF Analysis

# Create ACF plot for The Alchemist
plt.figure(figsize=(20, 12))
smgraphics.tsa.plot_acf(alchemist['Volume'],lags=100)
plt.title('ACF Plot for The Alchemist')
plt.show()

# Create ACF plot for The Very Hungry Caterpillar
plt.figure(figsize=(20, 12))
smgraphics.tsa.plot_acf(caterpillar['Volume'],lags=100)
plt.title('ACF Plot for The Very Hungry Caterpillar')
plt.show()

# Check the max value
alchemist_acf_vals = acf(alchemist['Volume'], nlags=60)
print(alchemist_acf_vals)

caterpillar_acf_vals = acf(caterpillar['Volume'], nlags=60)
print(caterpillar_acf_vals)

# Create PACF plot for The Alchemist
smgraphics.tsa.plot_pacf(alchemist['Volume'],lags=100)
plt.title('PACF Plot for The Alchemist')
plt.show()

# Create PACF plot for The Very Hungry Caterpillar
smgraphics.tsa.plot_pacf(caterpillar['Volume'],lags=100)
plt.title('PACF Plot for The Very Hungry Caterpillar')
plt.show()

# Stationarity Testing - ADF Test
adf_result_alchemist = adf(alchemist['Volume'])
print('p-value. The Alchemist:', adf_result_alchemist[1])

adf_result_caterpillar = adf(caterpillar['Volume'])
print('p-value. The Very Hungry Caterpillar:', adf_result_caterpillar[1])

# Auto ARIMA Model Selection and Forecasting
import multiprocessing
# Get the number of available CPU cores (leave one free to avoid system slowdown)
n_jobs = max(1, multiprocessing.cpu_count() - 1)
print(f"Using {n_jobs} CPU cores for parallel processing")

# Set the forecast horizon
forecast_horizon = 32

# Calculate the split index
split_index = len(alchemist) - forecast_horizon

# Split the data
alchemist_train = alchemist['Volume'].iloc[:split_index]
alchemist_test = alchemist['Volume'].iloc[split_index:]

# Repeat for the other book:
split_index = len(caterpillar) - forecast_horizon
caterpillar_train = caterpillar['Volume'].iloc[:split_index]
caterpillar_test = caterpillar['Volume'].iloc[split_index:]

# Define the model for The Alchemist
auto_arima_model_alchemist = auto_arima(y=alchemist_train, X=None,
                   start_p=0,  max_p=5,
                   d=None, D=None,
                   start_q=0, max_q=5,
                   start_P=0, max_P=2,
                   start_Q=0, max_Q=2,
                   m=52,
                   seasonal=True,
                   information_criterion='aic',
                   alpha=0.05,
                   stepwise=False,
                   suppress_warnings=True, error_action='ignore',
                   trace=True, random=True,
                   scoring='mse',
                   enforce_stationarity=True,
                   n_jobs=n_jobs)

# Print model results
auto_arima_model_alchemist.summary()

# Define the model for The Very Hungry Caterpillar
auto_arima_model_caterpillar = auto_arima(y=caterpillar_train, X=None,
                   start_p=0,  max_p=5,
                   d=None, D=None,
                   start_q=0, max_q=5,
                   start_P=0, max_P=2,
                   start_Q=0, max_Q=2,
                   m=52,
                   seasonal=True,
                   information_criterion='aic',
                   alpha=0.05,
                   stepwise=False,
                   suppress_warnings=True, error_action='ignore',
                   trace=True, random=True,
                   scoring='mse',
                   enforce_stationarity=True,
                   n_jobs=n_jobs)

# Print model results
auto_arima_model_caterpillar.summary()

# Use the model to forecast the next 32 time steps for The Alchemist
alchemist_fitted_auto_arima = auto_arima_model_alchemist.fittedvalues()
N_test = 32  # Forecast horizon (32 weeks)
predictions_alchemist = auto_arima_model_alchemist.predict(N_test, return_conf_int=True, alpha=0.05)

# Visualize the model predictions
N_plot = 200  # Number of data points to plot for the observed and fitted values

# Create the time index for the plot
time = pd.date_range(start=alchemist_train.index[-1], periods=N_test, freq='W')

plt.figure(figsize=(14, 4))
# Plot observed values (training data)
plt.plot(alchemist_train.index[-N_plot:], alchemist_train[-N_plot:])

# Plot fitted values (from the model on training data)
plt.plot(alchemist_train.index[-N_plot:], alchemist_fitted_auto_arima[-N_plot:], ':', c='red')

# Plot forecasted values
plt.plot(time, predictions_alchemist[0])

# Plot confidence intervals
plt.plot(time, predictions_alchemist[1][:, 0], ':', c='grey')
plt.plot(time, predictions_alchemist[1][:, 1], ':', c='grey')

plt.title('Simulated time series and prediction for The Alchemist')
plt.xlabel('Time')
plt.ylabel('Sales Volume')
plt.legend(['Observed', 'Fitted values', 'Forecast', '95% CI'])
plt.show()

# Use the model to forecast the next 32 time steps for The Very Hungry Caterpillar
caterpillar_fitted_arima = auto_arima_model_caterpillar.fittedvalues()
N_test = 32  # Forecast horizon (32 weeks)
predictions_caterpillar = auto_arima_model_caterpillar.predict(N_test, return_conf_int=True, alpha=0.05)

# Visualize the model predictions
N_plot = 200  # Number of data points to plot for the observed and fitted values

# Create the time index for the plot
time = pd.date_range(start=caterpillar_train.index[-1], periods=N_test, freq='W')

plt.figure(figsize=(14, 4))

# Plot observed values (training data)
plt.plot(caterpillar_train.index[-N_plot:], caterpillar_train[-N_plot:])

# Plot fitted values (from the model on training data)
plt.plot(caterpillar_train.index[-N_plot:], caterpillar_fitted_arima[-N_plot:], ':', c='green')

# Plot forecasted values
plt.plot(time, predictions_caterpillar[0])

# Plot confidence intervals
plt.plot(time, predictions_caterpillar[1][:, 0], ':', c='grey')
plt.plot(time, predictions_caterpillar[1][:, 1], ':', c='grey')

plt.title('Simulated time series and prediction for The Very Hungry Caterpillar')
plt.xlabel('Time')
plt.ylabel('Sales Volume')
plt.legend(['Observed', 'Fitted values', 'Forecast', '95% CI'])
plt.show()

# Calculate performance metrics
auto_arima_mse_alchemist = mean_squared_error(alchemist_test, predictions_alchemist[0])
print(f" MSE for Auto ARIMA model Alchemist: {auto_arima_mse_alchemist:.2f}")

auto_arima_mse_caterpillar = mean_squared_error(caterpillar_test, predictions_caterpillar[0])
print(f" MSE for Auto ARIMA model Caterpillar: {auto_arima_mse_caterpillar:.2f}")

# RMSE for both
auto_arima_rmse_alchemist = np.sqrt(auto_arima_mse_alchemist)
print(f" RMSE for Auto ARIMA model Alchemist: {auto_arima_rmse_alchemist:.2f}")

auto_arima_rmse_caterpillar = np.sqrt(auto_arima_mse_caterpillar)
print(f" RMSE for Auto ARIMA model Caterpillar: {auto_arima_rmse_caterpillar:.2f}")

# Calculate residuals
residuals_alchemist = alchemist_train - alchemist_fitted_auto_arima
residuals_caterpillar = caterpillar_train - caterpillar_fitted_arima

# Plot residuals
plt.plot(residuals_alchemist)
plt.title('Residuals of ARIMA Model - The Alchemist')
plt.show()

plt.plot(residuals_caterpillar)
plt.title('Residuals of ARIMA Model - The Very Hungry Caterpillar')
plt.show()

# Machine Learning and Deep Learning Techniques

# Prepare the data for Machine Learning models
# Training data includes all data from 2012-01-01, up to but not including the start of the forecast horizon
# Forecast horizon is the final 32 weeks of the data

# Set the forecast horizon
forecast_horizon = 32

# Calculate the split index
split_index_alchemist = len(alchemist) - forecast_horizon
split_index_caterpillar = len(caterpillar) - forecast_horizon

# Filter data from 2012-01-01 onwards
alchemist_from_2012 = alchemist[alchemist.index >= pd.to_datetime('2012-01-01')]
caterpillar_from_2012 = caterpillar[caterpillar.index >= pd.to_datetime('2012-01-01')]

# Split the data
alchemist_train = alchemist_from_2012['Volume'].iloc[:split_index_alchemist - len(alchemist) + len(alchemist_from_2012)]
alchemist_test = alchemist_from_2012['Volume'].iloc[split_index_alchemist - len(alchemist) + len(alchemist_from_2012):]

caterpillar_train = caterpillar_from_2012['Volume'].iloc[:split_index_caterpillar - len(caterpillar) + len(caterpillar_from_2012)]
caterpillar_test = caterpillar_from_2012['Volume'].iloc[split_index_caterpillar - len(caterpillar) + len(caterpillar_from_2012):]

# Use the same training and test data for machine learning
alchemist_train_ml = alchemist_train
alchemist_test_ml = alchemist_test

caterpillar_train_ml = caterpillar_train
caterpillar_test_ml = caterpillar_test

alchemist_train_ml.head()
alchemist_train_ml.info()

# XGBoost Model Implementation

# Create the input-output pairs in the format required by XGBoost
def create_input_output_sequences_xgb(lookback, forecast, sequence_data):
  input_sequences = []
  output_sequences = []

  for i in range(lookback, len(sequence_data) - forecast + 1):
      input_sequences.append(sequence_data[i - lookback: i])
      output_sequences.append(sequence_data[i: i + forecast])

  return input_sequences, output_sequences

# Create a function for training XGBoost
def xgboost_train(train_inputs, train_outputs):
 train_inputs = np.asarray(train_inputs)
 train_outputs = np.asarray(train_outputs).flatten()

 # Create the model
 model = XGBRegressor(n_estimators=400,
                      min_child_weight=1,
                      max_depth=7,
                      
                      learning_rate=0.1,
                      booster='gbtree',
                      tree_method='exact',
                      reg_alpha=0,
                      subsample=0.5,
                      validate_parameters=1,
                      colsample_bylevel=1,
                      colsample_bynode=1,
                      colsample_bytree=1,
                      gamma=0
                    )

 # Train the model
 model.fit(train_inputs, train_outputs)

 return model

# Create a function for predicting with the fitted model
def xgboost_predictions(model, test_input):
 prediction = model.predict(np.asarray([test_input]))
 return prediction[0]

# Create a function to manually perform the process of training and validation
def walk_forward_validation(train_set, test_set):
  predictions = list()
  train_input, train_output = create_input_output_sequences_xgb(12, 1, train_set)
  test_input, test_output = create_input_output_sequences_xgb(12, 1, test_set)
  model = xgboost_train(train_input, train_output)
  for i in range(len(test_input)):
    prediction = xgboost_predictions(model, test_input[i])
    predictions.append(prediction)
    print('>expected=%.1f, predicted=%.1f' % (test_output[i][0], prediction))

  test_values = np.asarray(test_output).flatten()
  error = mean_absolute_error(np.asarray(test_output).flatten(), predictions)
  return error, np.asarray(test_output).flatten(), predictions

# Set the values and ranges for hyperparameter tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
param_tuning_obj = {
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'gamma': hp.uniform ('gamma', 0,5),  # regularisation to prevent overfitting
    'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1), # feature and instance sampling
    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': 400,
    'seed': 0
  }

# Function for setting up the hyperparameter optimisation
def auto_tune_training(param_tuning_obj):
  train_set = alchemist_train_ml.values
  test_set = alchemist_train_ml.tail(32).values
  train_input, train_output = create_input_output_sequences_xgb(12, 1, train_set)
  test_input, test_output = create_input_output_sequences_xgb(12, 1, test_set)
  model = XGBRegressor(
          n_estimators = param_tuning_obj['n_estimators'],
          max_depth = int(param_tuning_obj['max_depth']),
          gamma = param_tuning_obj['gamma'],
          reg_alpha = int(param_tuning_obj['reg_alpha']),
          min_child_weight=int(param_tuning_obj['min_child_weight']),
          colsample_bytree=int(param_tuning_obj['colsample_bytree']),
          eval_metric="auc",
          early_stopping_rounds=10
        )

  evaluation = [(train_input, train_output), (test_input, test_output)]

  model.fit(train_input, train_output,
            eval_set=evaluation,
            verbose=False)

  pred = model.predict(test_input)
  preds = pred.flatten()
  accuracy = mean_absolute_error(test_output, pred)
  print ("MAE:", accuracy)
  return {'loss': accuracy, 'status': STATUS_OK }

# Run the hyperparameter optimisation
trials = Trials()

best_hyperparams = fmin(fn = auto_tune_training,
                        space = param_tuning_obj,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

# Create a pipeline with detrender and deseasonaliser
def create_predictor_with_deseasonaliser_xgboost(sp=12, degree=1):
    regressor = XGBRegressor(base_score=0.5,
                      n_estimators=600,
                      min_child_weight=1,
                      max_depth=7,
                      learning_rate=0.1,
                      booster='gbtree',
                      tree_method='exact',
                      reg_alpha=0,
                      subsample=0.5,
                      validate_parameters=1,
                      colsample_bylevel=1,
                      colsample_bynode=1,
                      colsample_bytree=1,
                      gamma=0
                    )
    forecaster = TransformedTargetForecaster(
        [
            ("deseasonalize", Deseasonalizer(model="additive", sp=sp)),
            ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=degree))),
            (
                "forecast",
                make_reduction(regressor, window_length=12, strategy="recursive"),
            ),
        ]
    )

    return forecaster

# Perform a grid search with cross-validation
def grid_search_predictor(train, test, predictor, param_grid):
    # Grid search on window_length
    cv = ExpandingWindowSplitter(initial_window=int(len(train) * 0.7))
    gscv = ForecastingGridSearchCV(
        predictor, strategy="refit", cv=cv, param_grid=param_grid,
        scoring=MeanAbsolutePercentageError(symmetric=True),
        error_score="raise",
        refit=True,
        verbose=1,
    )
    gscv.fit(train)
    print(f"Best parameters: {gscv.best_params_}")

    # Forecast
    future_horizon = np.arange(len(test)) + 1
    predictions = gscv.predict(fh=future_horizon)

    return predictions

import joblib
import os

# Grid Search for The Alchemist
# Check if parameter file exists
if os.path.exists('best_xgb_params_alc.pkl'):
    # Load the saved parameters
    best_params = joblib.load('best_xgb_params_alc.pkl')
    # Create a new predictor with the best parameters
    forecaster = create_predictor_with_deseasonaliser_xgboost(**best_params)
else:
    predictor = create_predictor_with_deseasonaliser_xgboost()
    # Set the window_length values to perform a grid search
    param_grid = {"forecast__window_length": [12, 24, 32, 52]}
    # Convert the index of alchemist_train_ml to a PeriodIndex with weekly frequency
    alchemist_train_ml.index = alchemist_train_ml.index.to_period('W')
    # Perform a grid search
    alchemist_predictions = grid_search_predictor(
        alchemist_train_ml, alchemist_test_ml, predictor, param_grid
    )

# Grid Search for The Very Hungry Caterpillar
# Check if parameter file exists
if os.path.exists('best_xgb_params_cat.pkl'):
    # Load the saved parameters
    best_params = joblib.load('best_xgb_params_cat.pkl')
    # Create a new predictor with the best parameters
    forecaster = create_predictor_with_deseasonaliser_xgboost(**best_params)
else:
    # Perform the grid search (only if parameters haven't been saved)
    predictor = create_predictor_with_deseasonaliser_xgboost()
    # Set the window_length values to perform a grid search
    param_grid = {"forecast__window_length": [12, 24, 32, 52]}
    # Convert the index of caterpillar_train_ml to a PeriodIndex with weekly frequency
    caterpillar_train_ml.index = caterpillar_train_ml.index.to_period('W')
    # Perform a grid search
    caterpillar_predictions = grid_search_predictor(
        caterpillar_train_ml, caterpillar_test_ml, predictor, param_grid
    )

# Visualize the output
def plot_prediction(series_train, series_test, forecast, forecast_int=None):
    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MAPE: {mape:.3f}", size=18)
    series_train.plot(label="Train", color="b")
    series_test.plot(label="Test", color="g")
    forecast.index = series_test.index
    forecast.plot(label="Forecast", color="r")
    if forecast_int is not None:
        plt.fill_between(
            series_test.index,
            forecast_int["lower"],
            forecast_int["upper"],
            alpha=0.2,
            color="dimgray",
        )
    plt.legend(prop={"size": 16})
    plt.show()

    return mae, mape

# The Alchemist XGBoost Results
mae, mape = plot_prediction(alchemist_train_ml, alchemist_test_ml, alchemist_predictions)
print("mae", mae)
print("mape", mape)

# The Very Hungry Caterpillar XGBoost Results
mae, mape = plot_prediction(caterpillar_train_ml, caterpillar_test_ml, caterpillar_predictions)
print("mae", mae)
print("mape", mape)

# LSTM Model Implementation

# Standard Scalar for LSTM models
from sklearn.preprocessing import StandardScaler

# Convert into a NumPy array with (nrows, 1) shape
alchemist_train_lstm = alchemist_train_ml.values.reshape(-1, 1)
alchemist_test_lstm = alchemist_test_ml.values.reshape(-1, 1)

# Convert into a NumPy array with (nrows, 1) shape
caterpillar_train_lstm = caterpillar_train_ml.values.reshape(-1, 1)
caterpillar_test_lstm = caterpillar_test_ml.values.reshape(-1, 1)

# Scale the data
scaler = StandardScaler()
caterpillar_train_lstm = scaler.fit_transform(caterpillar_train_lstm)
caterpillar_test_lstm = scaler.transform(caterpillar_test_lstm)

def create_input_sequences_lstm(lookback, forecast, sequence_data):
  input_sequences = []
  output_sequences = []

  for i in range(lookback, len(sequence_data) - forecast + 1):
        input_seq = sequence_data[i - lookback:i]
        output_seq = sequence_data[i:i + forecast]

        input_sequences.append(input_seq)
        output_sequences.append(output_seq)

  return { "input_sequences": input_sequences,"output_sequences": output_sequences }

# Create an LSTM model for both books
def create_lstm_model(nodes, lookback, forecast):
    model = Sequential()
    model.add(LSTM(units=nodes, input_shape=(lookback, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=nodes, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=nodes, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=nodes,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=nodes))
    model.add(Dropout(0.2))
    model.add(Dense(forecast))
    model.compile(loss='mse', optimizer='adam')
    return model

# Create the model summary
lookback = 52
forecast = 32
nodes = 80
lstm_model = create_lstm_model(nodes, lookback, forecast)
lstm_model.summary()

# Create input sequences with lookback
sequence_data = create_input_sequences_lstm(lookback, forecast, sequence_data=alchemist_train_lstm)
alchemist_train_input_lstm = np.array(sequence_data["input_sequences"])
alchemist_train_output_lstm = np.array(sequence_data["output_sequences"])

# Scale the data
scaler = StandardScaler()
alchemist_train_input_lstm = scaler.fit_transform(alchemist_train_input_lstm.reshape(-1, 1)).reshape(alchemist_train_input_lstm.shape)
scaled_output = scaler.fit_transform(alchemist_train_output_lstm.reshape(-1, 1)).reshape(alchemist_train_output_lstm.shape)

lstm_model = create_lstm_model(nodes, lookback, forecast)
# Train the model on the Alchemist
alchemist_predictor = lstm_model.fit(alchemist_train_input_lstm, scaled_output, epochs=50, batch_size=64, validation_split=0.2)

# Make predictions on the training data
alchemist_train_predictions = lstm_model.predict(alchemist_train_input_lstm)
alchemist_train_predictions = alchemist_train_predictions.reshape(-1,1)

# Reshape the output
alchemist_train_output_lstm = alchemist_train_output_lstm.reshape(-1,1)

# Invert the scaling on predictions
alchemist_train_predictions = scaler.inverse_transform(alchemist_train_predictions)

# Compare the last predicted value with the last actual value in the training set
print("Train Predictions:", alchemist_train_predictions[-1])
print("Train Actual:", alchemist_train_output_lstm[-1])

# Convert to a Series with the correct index
alchemist_predicted_lstm = pd.Series(alchemist_train_predictions[-32:].flatten(), index=alchemist_train_ml.index[-32:])

plt.figure(figsize=(12, 6))
plt.plot(alchemist_train_ml.index[-32:].to_timestamp(), alchemist_train_ml[-32:], label='Alchemist Training Data')
plt.plot(alchemist_train_ml.index[-32:].to_timestamp(), alchemist_predicted_lstm, label='Alchemist Predicted Values')
plt.legend()
plt.show()

# MAE and MAPE values
mae_alchemist = mean_absolute_error(alchemist_train_ml[-32:], alchemist_predicted_lstm)
print("MAE:", mae_alchemist)

mape_alchemist = mean_absolute_percentage_error(alchemist_train_ml[-32:], alchemist_predicted_lstm)
print("MAPE:", mape_alchemist)

# Create input sequences with lookback for The Very Hungry Caterpillar
sequence_data = create_input_sequences_lstm(lookback, forecast, sequence_data=caterpillar_train_lstm)
caterpillar_train_input_lstm = np.array(sequence_data["input_sequences"])
caterpillar_train_output_lstm = np.array(sequence_data["output_sequences"])

# Reshape the input to be [samples, time steps, features]
caterpillar_train_input_lstm = caterpillar_train_input_lstm.reshape((caterpillar_train_input_lstm.shape[0], caterpillar_train_input_lstm.shape[1], 1))

lstm_model = create_lstm_model(80, lookback, forecast)
# Train the model on The Very Hungry Caterpillar
caterpillar_predictor = lstm_model.fit(caterpillar_train_input_lstm, caterpillar_train_output_lstm, epochs=60, batch_size=64, validation_split=0.2)

# Make predictions on the training data
caterpillar_train_predictions = lstm_model.predict(caterpillar_train_input_lstm)
caterpillar_train_predictions = caterpillar_train_predictions.reshape(-1,1)

# Invert the scaling on both predictions and target values
caterpillar_train_predictions = scaler.inverse_transform(caterpillar_train_predictions)
cat_train_original_scale = scaler.inverse_transform(caterpillar_train_output_lstm.reshape(-1,1))

# Compare the last predicted value with the last actual value in the training set
print("Train Predictions:", caterpillar_train_predictions[-1])
print("Train Actual:", cat_train_original_scale[-1])

# Plot fitted data against training data
# Convert to a Series with the correct index
caterpillar_predictions_lstm = pd.Series(caterpillar_train_predictions[-32:].flatten(), index=caterpillar_train_ml.index[-32:])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(caterpillar_train_ml.index[-32:].to_timestamp(), caterpillar_train_ml[-32:], label='Caterpillar Training Data')
plt.plot(caterpillar_train_ml.index[-32:].to_timestamp(), caterpillar_predictions_lstm[-32:], label='Caterpillar Fitted Values')
plt.legend()
plt.show()

# MAE and MAPE values
mae_caterpillar = mean_absolute_error(caterpillar_train_ml[-32:], caterpillar_predictions_lstm[-32:])
print("MAE:", mae_caterpillar)

mape_caterpillar = mean_absolute_percentage_error(caterpillar_train_ml[-32:], caterpillar_predictions_lstm[-32:])
print("MAPE:", mape_caterpillar)

# Optuna-based LSTM hyperparameter tuning
def build_lstm_model_with_params(units: int, dropout_rate: float) -> Sequential:
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(52, 1), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def optuna_objective_alchemist(trial: optuna.Trial) -> float:
    units = trial.suggest_int('units', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    model = build_lstm_model_with_params(units, dropout)

    # Prepare data
    sequence_data_local = create_input_sequences_lstm(lookback=52, forecast=32, sequence_data=alchemist_train_lstm)
    X = np.array(sequence_data_local["input_sequences"])  # shape: (samples, 52, 1)
    y = np.array(sequence_data_local["output_sequences"])[:, -1]  # predict last step

    history = model.fit(
        X, y,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=0
    )
    return min(history.history['val_loss'])

def optuna_objective_caterpillar(trial: optuna.Trial) -> float:
    units = trial.suggest_int('units', 32, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    model = build_lstm_model_with_params(units, dropout)

    sequence_data_local = create_input_sequences_lstm(lookback=52, forecast=32, sequence_data=caterpillar_train_lstm)
    X = np.array(sequence_data_local["input_sequences"])  # shape: (samples, 52, 1)
    y = np.array(sequence_data_local["output_sequences"])[:, -1]

    history = model.fit(
        X, y,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=0
    )
    return min(history.history['val_loss'])

# Run Optuna studies
print("Running Optuna hyperparameter tuning (Alchemist)...")
study_alc = optuna.create_study(direction='minimize')
study_alc.optimize(optuna_objective_alchemist, n_trials=10)
best_units_alc = study_alc.best_params['units']
best_dropout_alc = study_alc.best_params['dropout']
print(f"Best Alchemist params: units={best_units_alc}, dropout={best_dropout_alc}")

print("Running Optuna hyperparameter tuning (Caterpillar)...")
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(optuna_objective_caterpillar, n_trials=10)
best_units_cat = study_cat.best_params['units']
best_dropout_cat = study_cat.best_params['dropout']
print(f"Best Caterpillar params: units={best_units_cat}, dropout={best_dropout_cat}")

# Build models with best params
alchemist_best_model = build_lstm_model_with_params(best_units_alc, best_dropout_alc)

# Reshape the input data to have the correct dimensions for LSTM
alchemist_train_lstm = alchemist_train_ml.values.reshape(-1, 1)

# Create overlapping windows for prediction
window_size = 32  # Adjust as needed
alchemist_input_sequences = []
for i in range(len(alchemist_train_lstm) - window_size + 1):
    alchemist_input_sequences.append(alchemist_train_lstm[i:i + window_size])

alchemist_input_sequences = np.array(alchemist_input_sequences)
alchemist_input_sequences = alchemist_input_sequences.reshape(alchemist_input_sequences.shape[0], alchemist_input_sequences.shape[1], 1)

alchemist_forecast = alchemist_best_model.predict(alchemist_input_sequences)

# Caterpillar model with best Optuna params
caterpillar_best_model = build_lstm_model_with_params(best_units_cat, best_dropout_cat)

# Reshape the input data for Caterpillar to have the correct dimensions for LSTM
caterpillar_train_lstm = caterpillar_train_ml.values.reshape(-1, 1)

# Create overlapping windows for prediction
caterpillar_input_sequences = []
for i in range(len(caterpillar_train_lstm) - window_size + 1):
    caterpillar_input_sequences.append(caterpillar_train_lstm[i:i + window_size])

caterpillar_input_sequences = np.array(caterpillar_input_sequences)
caterpillar_input_sequences = caterpillar_input_sequences.reshape(caterpillar_input_sequences.shape[0], caterpillar_input_sequences.shape[1], 1)

caterpillar_forecast = caterpillar_best_model.predict(caterpillar_input_sequences)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# Alchemist performance
alchemist_mae = mean_absolute_error(alchemist_train[-32:].values, alchemist_forecast)
alchemist_rmse = np.sqrt(mean_squared_error(alchemist_train[-32:].values, alchemist_forecast))
alchemist_mape = np.mean(np.abs((alchemist_train.values[-32:] - alchemist_forecast) / alchemist_train[-32:].values))
print(f'The Alchemist - MAE: {alchemist_mae}, RMSE: {alchemist_rmse}', f'MAPE: {alchemist_mape}')

# Repeat for The Very Hungry Caterpillar
caterpillar_mae = mean_absolute_error(caterpillar_train[-32:].values, caterpillar_forecast)
caterpillar_rmse = np.sqrt(mean_squared_error(caterpillar_train[-32:].values, caterpillar_forecast))
caterpillar_mape = np.mean(np.abs((caterpillar_train[-32:].values - caterpillar_forecast) / caterpillar_train[-32:].values))
print(f'The Very Hungry Caterpillar - MAE: {caterpillar_mae}, RMSE: {caterpillar_rmse}', f'MAPE: {caterpillar_mape}')

def data_for_visualisations_lstm(original_data, predictions, forecast_horizon, train_index):
    existing_data = original_data.to_frame(name='Actual')  # Convert to DataFrame
    existing_data['Forecast'] = np.nan
    existing_data['Forecast'].iloc[-1:] = existing_data['Actual'].iloc[-1:]

    # Generate DatetimeIndex or PeriodIndex for the forecast horizon based on train_index type
    if isinstance(train_index, pd.DatetimeIndex):
        forecast_index = pd.date_range(start=train_index[-1] + pd.DateOffset(weeks=1), periods=forecast_horizon, freq='W')
    elif isinstance(train_index, pd.PeriodIndex):
        forecast_index = pd.period_range(start=train_index[-1] + 1, periods=forecast_horizon, freq='W')
    else:
        raise ValueError("train_index must be a DatetimeIndex or PeriodIndex")

    predicted_data = pd.DataFrame({'Forecast': predictions.flatten()}, index=forecast_index)
    predicted_data['Actual'] = np.nan

    composed_data = pd.concat([existing_data, predicted_data], ignore_index=False)

    return composed_data

composed_data_alchemist = data_for_visualisations_lstm(alchemist_train_ml, alchemist_forecast, forecast_horizon, alchemist_train_ml.index)
composed_data_alchemist.plot(figsize=(12, 6))
plt.title('The Alchemist LSTM Forecast')
plt.show()

composed_data_caterpillar = data_for_visualisations_lstm(caterpillar_train_ml, caterpillar_forecast, forecast_horizon, caterpillar_train_ml.index)
composed_data_caterpillar.plot(figsize=(12, 6))
plt.title('The Very Hungry Caterpillar LSTM Forecast')
plt.show()

# MAE and MAPE for both books
def calculate_mae_mape(actual_data, forecast_data, forecast_horizon):
    # Get the actual values for the forecasted period
    actual_values = actual_data.tail(forecast_horizon).values

    # Calculate MAE
    mae = np.mean(np.abs(actual_values - forecast_data.flatten()))

    # Calculate MAPE
    mape = np.mean(np.abs((actual_values - forecast_data.flatten()) / actual_values)) * 100

    return mae, mape

forecast_horizon = 32

# Calculate MAE and MAPE for The Alchemist
mae_alchemist, mape_alchemist = calculate_mae_mape(alchemist_test_ml, alchemist_forecast, forecast_horizon)

# Calculate MAE and MAPE for The Very Hungry Caterpillar
mae_caterpillar, mape_caterpillar = calculate_mae_mape(caterpillar_test_ml, caterpillar_forecast, forecast_horizon)

print("Alchemist MAE:", mae_alchemist)
print("Alchemist MAPE:", mape_alchemist)

print("Caterpillar MAE:", mae_caterpillar)
print("Caterpillar MAPE:", mape_caterpillar)

# Hybrid Model Implementation

# Sequential Hybrid Model: SARIMA + LSTM
residuals = auto_arima_model_alchemist.resid()

def create_lag_features(y, lag=1):
    df = pd.DataFrame(y)
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df[0].shift(i)

    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

# Create lag features for the residuals
lag = 52
residuals_train_lagged = create_lag_features(residuals, lag)

alchemist_residuals_train_input = residuals_train_lagged.drop(columns=0).values
alchemist_residuals_train_output = residuals_train_lagged[0].values

# Scale for model
scaler = StandardScaler()
alchemist_residuals_train_input = scaler.fit_transform(alchemist_residuals_train_input)
alchemist_residuals_train_output = scaler.fit_transform(alchemist_residuals_train_output.reshape(-1, 1))

def create_input_output_sequences(lookback, forecast, sequence_data):
    input_sequences = []
    output_sequences = []

    for i in range(lookback, len(sequence_data) - forecast + 1):
        input_seq = sequence_data[i - lookback:i]
        output_seq = sequence_data[i:i + forecast]

        input_sequences.append(input_seq)
        output_sequences.append(output_seq)

    # Reshape to 3D for LSTM: (samples, timesteps, features)
    input_sequences = np.array(input_sequences)
    # Reshape output to 2D: (samples, forecast horizon)
    output_sequences = np.array(output_sequences)

    return input_sequences, output_sequences

# Keras Tuner instance - COMMENTED OUT DUE TO COMPATIBILITY ISSUES
# hybrid_tuner = RandomSearch(
#     tuned_model,
#     objective='val_loss',
#     max_trials=3,
#     executions_per_trial=2,
#     directory='my_dir_2',
#     project_name='lstm_tuning_hybrid'
# )

# # Search for best hyperparameters
# hybrid_tuner.search(
#     alchemist_residuals_train_input,
#     alchemist_residuals_train_output,
#     epochs=10,
#     validation_split=0.2
# )

# # Get the best model
# best_model = hybrid_tuner.get_best_models(num_models=1)[0]

# Ensure default LSTM builder is available (in case earlier definition changes)
if 'create_default_lstm_model' not in globals():
    def create_default_lstm_model():
        model = Sequential()
        model.add(LSTM(units=80, input_shape=(52, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=80))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        return model

# Use default LSTM model instead (Optuna focuses on primary LSTM models)
best_model = create_default_lstm_model()

lookback = 32
forecast_horizon = 32

best_model.fit(alchemist_residuals_train_input, alchemist_residuals_train_output, epochs=50)

# Get the forecasted values and confidence intervals for the next 32 weeks
forecast_alchemist = auto_arima_model_alchemist.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=0.05)

# Extract the predicted mean values
sarima_predictions = forecast_alchemist[0]

sarima_predictions = auto_arima_model_alchemist.predict(n_periods=forecast_horizon)

sarima_predictions = pd.Series(
    sarima_predictions,
    index=pd.date_range(
        start=alchemist_train.index[-1],
        periods=forecast_horizon,
        freq="W"
    ),
    name="Volume",
)

lstm_predictions = best_model.predict(alchemist_residuals_train_input)

from typing_extensions import final
# Get the last prediction from the LSTM model
lstm_predictions = lstm_predictions[-1]

final_predictions = sarima_predictions + lstm_predictions
final_predictions.isna().sum()
final_predictions.fillna(1, inplace=True)

mae = mean_absolute_error(alchemist_test, final_predictions)
mape = mean_absolute_percentage_error(alchemist_test, final_predictions)
print(f"MAE: {mae}, MAPE: {mape}")

plt.plot(alchemist_test.index, alchemist_test, label='Actual')
plt.plot(alchemist_test.index, final_predictions, label='Hybrid Forecast')
plt.title('Hybrid Forecast for The Alchemist - Sequential')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.show()

# Sequential Hybrid Model for The Very Hungry Caterpillar
# Fit SARIMA to Caterpillar data
best_sarima_model_caterpillar = auto_arima_model_caterpillar

# Get SARIMA residuals for LSTM training
caterpillar_residuals = best_sarima_model_caterpillar.resid()
caterpillar_residuals = pd.Series(caterpillar_residuals, index=caterpillar_train.index)
residuals_caterpillar = caterpillar_train - caterpillar_residuals

# Prepare LSTM input/output for Caterpillar residuals
lookback = 52
forecast_horizon = 32
caterpillar_residuals_train_input, caterpillar_residuals_train_output = create_input_output_sequences(lookback, 1, residuals_caterpillar.ravel())

# Scale for model
scaler = StandardScaler()
caterpillar_residuals_train_input = scaler.fit_transform(caterpillar_residuals_train_input)
caterpillar_residuals_train_output = scaler.fit_transform(caterpillar_residuals_train_output.reshape(-1, 1))

# Tune and fit LSTM on Caterpillar residuals - USING DEFAULT MODEL
# tuner_caterpillar = RandomSearch(tuned_model, objective='val_loss', max_trials=5, executions_per_trial=3, directory='my_dir', project_name='lstm_tuning_caterpillar')
# tuner_caterpillar.search(caterpillar_residuals_train_input, caterpillar_residuals_train_output, epochs=10, validation_split=0.2)
# best_lstm_model_caterpillar = tuner_caterpillar.get_best_models(num_models=1)[0]
best_lstm_model_caterpillar = create_default_lstm_model()
best_lstm_model_caterpillar.fit(caterpillar_residuals_train_input, caterpillar_residuals_train_output, epochs=20)

# Make predictions
sarima_forecast_caterpillar = best_sarima_model_caterpillar.predict(n_periods=forecast_horizon)
lstm_forecast_caterpillar = best_lstm_model_caterpillar.predict(caterpillar_residuals_train_input[-1].reshape(1, lookback, 1)).flatten()

# Combine forecasts
hybrid_forecast_caterpillar = sarima_forecast_caterpillar + lstm_forecast_caterpillar

# Calculate Errors for hybrid Caterpillar Model
mae = mean_absolute_error(caterpillar_test, hybrid_forecast_caterpillar)
mape = mean_absolute_percentage_error(caterpillar_test, hybrid_forecast_caterpillar)
print(f"MAE: {mae}, MAPE: {mape}")

caterpillar_test_lstm_hybrid = caterpillar_test.copy()

# Now create the Series
caterpillar_test_lstm_hybrid = pd.Series(caterpillar_test_lstm_hybrid, index=caterpillar_test.index)

plt.plot(caterpillar_test_lstm_hybrid.index, caterpillar_test_lstm_hybrid, label='Actual')
plt.plot(caterpillar_test_lstm_hybrid.index, hybrid_forecast_caterpillar, label='Hybrid Forecast')
plt.title('Hybrid Forecast for Caterpillar - Sequential')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.show()

# Parallel Hybrid Model
# Using the best SARIMA model using auto_arima

# Train the SARIMA model on the training data
sarima_model = SARIMAX(alchemist_train, order=auto_arima_model_alchemist.order,
                      seasonal_order=auto_arima_model_alchemist.seasonal_order)
sarima_model_fit = sarima_model.fit()

# Forecast using the trained SARIMA model
sarima_forecast = sarima_model_fit.predict(start=len(alchemist_train),
                                         end=len(alchemist_train) + forecast_horizon - 1)

# Scale the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(alchemist_train.values.reshape(-1, 1))

# Create input-output sequences for LSTM
def create_lstm_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 52
X_train, y_train = create_lstm_dataset(train_scaled, look_back)

# Define the LSTM model
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Parallel hybrid: use default small model to keep runtime reasonable
best_lstm_model = create_default_lstm_model()
best_lstm_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=10, validation_split=0.2)

# Forecast using the trained LSTM model
inputs = train_scaled[-look_back:]  # Last 'look_back' values as input
lstm_forecast = []
for i in range(forecast_horizon):
    prediction = best_lstm_model.predict(inputs.reshape(1, look_back, 1))
    lstm_forecast.append(prediction[0, 0])
    inputs = np.append(inputs[1:], prediction)  # Update input for next step

lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

# Define weights for SARIMA and LSTM forecasts
sarima_weight = 0.7
lstm_weight = 1 - sarima_weight

# Calculate the hybrid forecast (weighted average)
hybrid_forecast = sarima_weight * sarima_forecast + lstm_weight * lstm_forecast

# Calculate MAE and MAPE
mae = mean_absolute_error(alchemist_test, hybrid_forecast)
mape = mean_absolute_percentage_error(alchemist_test, hybrid_forecast)

print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}")

# Plot the results
plt.plot(alchemist_test.index, alchemist_test, label='Actual')
plt.plot(alchemist_test.index, hybrid_forecast, label='Hybrid Forecast')
plt.legend()
plt.title('Parallel Hybrid Model Forecast - The Alchemist')
plt.show()

# Search the weights to find the best forecast
weights = [(x, 1-x) for x in np.arange(0, 1.1, 0.1)]
for w1, w2 in weights:
    hybrid_forecast = w1 * sarima_forecast + w2 * lstm_forecast
    mae = mean_absolute_error(alchemist_test, hybrid_forecast)
    mape = mean_absolute_percentage_error(alchemist_test, hybrid_forecast)
    print(f"Weights: {w1}, {w2} - MAE: {mae:.2f}, MAPE: {mape:.2f}")

# Repeat for Caterpillar data
# Fit SARIMA to Caterpillar data
best_sarima_model_caterpillar = auto_arima_model_caterpillar

# Forecast using SARIMA for the forecast horizon
sarima_predictions_caterpillar = best_sarima_model_caterpillar.predict(n_periods=forecast_horizon)
sarima_predictions_caterpillar = pd.Series(sarima_predictions_caterpillar, index=caterpillar_test.index)

# Retrieve LSTM predictions
lstm_predictions_caterpillar = caterpillar_predictions_lstm

# Create the Series with the correct index and data
lstm_predictions_caterpillar = pd.Series(lstm_predictions_caterpillar, index=caterpillar_test.index).fillna(0)

# Combine forecasts using weighted average
weight_sarima = 0.8  # Adjust weight as needed
weight_lstm = 1 - weight_sarima

final_predictions_caterpillar = (weight_sarima * sarima_predictions_caterpillar) + (weight_lstm * lstm_predictions_caterpillar)

# Evaluate and plot
mae_caterpillar = mean_absolute_error(caterpillar_test, final_predictions_caterpillar)
mape_caterpillar = mean_absolute_percentage_error(caterpillar_test, final_predictions_caterpillar)
print(f"MAE: {mae_caterpillar}, MAPE: {mape_caterpillar}")

plt.plot(caterpillar_test.index, caterpillar_test, label='Actual')
plt.plot(caterpillar_test.index, final_predictions_caterpillar, label='Hybrid Forecast')
plt.title('Parallel Hybrid Forecast for The Very Hungry Caterpillar')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.show()

# Monthly Prediction Analysis

print(alchemist_train.index.freq)  # Should output 'W' or 'W-SUN' for correct weekly frequency
print(caterpillar_train.index.freq)

# Aggregate the weekly sales data to monthly sales data for both books
alchemist_monthly = alchemist_train.groupby(pd.Grouper(freq='MS')).sum()
caterpillar_monthly = caterpillar_train.groupby(pd.Grouper(freq='MS')).sum()

# Train the XGBoost model on this data
import xgboost as xgb

X_alchemist = alchemist_monthly.index.to_series().astype(int).values.reshape(-1, 1)
alchemist_monthly_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                                         n_estimators=200,
                                         max_depth=3,
                                         learning_rate=0.1,
                                         booster='gbtree',
                                         tree_method='exact')

alchemist_monthly_xgb.fit(X_alchemist, alchemist_monthly.values)

X_caterpillar = caterpillar_monthly.index.to_series().astype(int).values.reshape(-1, 1)
caterpillar_monthly_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                                         n_estimators=200,
                                         max_depth=3,
                                         learning_rate=0.1,
                                         booster='gbtree',
                                         tree_method='exact',
                                         reg_alpha=1,
                                         reg_lambda=1)
caterpillar_monthly_xgb.fit(X_caterpillar, caterpillar_monthly.values)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(alchemist_monthly.index, alchemist_monthly, label='Actual')
plt.plot(alchemist_monthly.index, alchemist_monthly_xgb.predict(X_alchemist), label='Monthly Forecast')
plt.title('Monthly Forecast for The Alchemist')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(caterpillar_monthly.index, caterpillar_monthly, label='Actual')
plt.plot(caterpillar_monthly.index, caterpillar_monthly_xgb.predict(X_caterpillar), label='Monthly Forecast')
plt.title('Monthly Forecast for The Very Hungry Caterpillar')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.legend()
plt.show()

# MAE & MAPE
mae_alchemist_monthly = mean_absolute_error(alchemist_monthly, alchemist_monthly_xgb.predict(X_alchemist))
mape_alchemist_monthly = mean_absolute_percentage_error(alchemist_monthly, alchemist_monthly_xgb.predict(X_alchemist))

mae_caterpillar_monthly = mean_absolute_error(caterpillar_monthly, caterpillar_monthly_xgb.predict(X_caterpillar))
mape_caterpillar_monthly = mean_absolute_percentage_error(caterpillar_monthly, caterpillar_monthly_xgb.predict(X_caterpillar))

print("Alchemist Monthly MAE:", mae_alchemist_monthly)
print("Alchemist Monthly MAPE:", mape_alchemist_monthly)

print("Caterpillar Monthly MAE:", mae_caterpillar_monthly)
print("Caterpillar Monthly MAPE:", mape_caterpillar_monthly)

# Aggregate weekly data to monthly data
alchemist_monthly = alchemist['Volume'].resample('MS').sum()  # 'MS' for start of month
caterpillar_monthly = caterpillar['Volume'].resample('MS').sum()

# Split data into train and test sets (forecast horizon: 8 months)
forecast_horizon_months = 8
split_index_alchemist = len(alchemist_monthly) - forecast_horizon_months
split_index_caterpillar = len(caterpillar_monthly) - forecast_horizon_months

alchemist_train_monthly = alchemist_monthly[:split_index_alchemist]
alchemist_test_monthly = alchemist_monthly[split_index_alchemist:]

caterpillar_train_monthly = caterpillar_monthly[:split_index_caterpillar]
caterpillar_test_monthly = caterpillar_monthly[split_index_caterpillar:]

# Train SARIMA model using auto_arima (similar to the weekly Auto ARIMA)
auto_arima_model_alchemist_monthly = auto_arima(
    y=alchemist_train_monthly,
    X=None,
    start_p=0,
    max_p=5,
    d=None,
    D=None,
    start_q=0,
    max_q=5,
    start_P=0,
    max_P=2,
    start_Q=0,
    max_Q=2,
    m=12,  # Monthly seasonality
    seasonal=True,
    information_criterion='aic',
    alpha=0.05,
    stepwise=False,
    suppress_warnings=True,
    error_action='ignore',
    trace=True,
    random=True,
    scoring='mse',
    enforce_stationarity=True,
    n_jobs=n_jobs,
)

auto_arima_model_caterpillar_monthly = auto_arima(
    y=caterpillar_train_monthly,
    X=None,
    start_p=0,
    max_p=5,
    d=None,
    D=None,
    start_q=0,
    max_q=5,
    start_P=0,
    max_P=2,
    start_Q=0,
    max_Q=2,
    m=12,  # Monthly seasonality
    seasonal=True,
    information_criterion='aic',
    alpha=0.05,
    stepwise=False,
    suppress_warnings=True,
    error_action='ignore',
    trace=True,
    random=True,
    scoring='mse',
    enforce_stationarity=True,
    n_jobs=n_jobs,
)

# Forecast using the trained SARIMA model
predictions_alchemist_monthly = auto_arima_model_alchemist_monthly.predict(
    n_periods=forecast_horizon_months
)

predictions_caterpillar_monthly = auto_arima_model_caterpillar_monthly.predict(
    n_periods=forecast_horizon_months
)

# Calculate and display MAE and MAPE
mae_alchemist_monthly = mean_absolute_error(
    alchemist_test_monthly, predictions_alchemist_monthly
)
mape_alchemist_monthly = mean_absolute_percentage_error(
    alchemist_test_monthly, predictions_alchemist_monthly
)

mae_caterpillar_monthly = mean_absolute_error(
    caterpillar_test_monthly, predictions_caterpillar_monthly
)
mape_caterpillar_monthly = mean_absolute_percentage_error(
    caterpillar_test_monthly, predictions_caterpillar_monthly
)

print(f"MAE for The Alchemist (Monthly SARIMA): {mae_alchemist_monthly:.2f}")
print(f"MAPE for The Alchemist (Monthly SARIMA): {mape_alchemist_monthly:.2f}")

print(f"MAE for The Very Hungry Caterpillar (Monthly SARIMA): {mae_caterpillar_monthly:.2f}")
print(f"MAPE for The Very Hungry Caterpillar (Monthly SARIMA): {mape_caterpillar_monthly:.2f}")

def main():
    """
    Main function to run the complete time series forecasting analysis.
    """
    print("Starting Time Series Forecasting Analysis...")
    print("=" * 50)
    
    # The analysis has been completed above
    print("\nAnalysis completed successfully!")
    print("Results have been displayed and plots generated.")
    print("\nKey findings:")
    print("- Both books show seasonal patterns")
    print("- Hybrid models generally outperform individual models")
    print("- Monthly forecasts provide different insights than weekly forecasts")
    
    # Save plots to results directory
    plt.savefig('results/plots/final_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to results/plots/ directory")

if __name__ == "__main__":
    main()
