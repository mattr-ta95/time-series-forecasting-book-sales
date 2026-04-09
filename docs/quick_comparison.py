"""
Quick Before/After Comparison - Phase 1 Fixes
Shows the impact of fixing critical bugs
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os

# Load data
data_dict = pd.read_excel('data/UK_Weekly_Trended_Timeline_from_200101_202429.xlsx', sheet_name=None)
df_UK_weekly = pd.concat([df.assign(Category=sheet_name) for sheet_name, df in data_dict.items()], ignore_index=True)

# Prepare data
df_UK_weekly['End Date'] = pd.to_datetime(df_UK_weekly['End Date'])
df_UK_weekly = df_UK_weekly.set_index('End Date')
num_cols = ['Volume']
other_cols = ['Title', 'Category', 'ISBN']
df_UK_weekly_full = df_UK_weekly.groupby(other_cols)[num_cols].resample('W').sum().fillna(0)
df_UK_weekly_full = df_UK_weekly_full.reset_index()
df_UK_weekly_full['ISBN'] = df_UK_weekly_full['ISBN'].astype(str)
df_UK_weekly_full['End Date'] = pd.to_datetime(df_UK_weekly_full['End Date'])

# Select books
alchemist = df_UK_weekly_full[df_UK_weekly_full['Title'].isin(['Alchemist, The'])]
alchemist = alchemist.set_index('End Date').resample('W').sum().ffill()
alchemist = alchemist[alchemist.index > '2012-01-01']

# Train/test split
forecast_horizon = 32
split_index = len(alchemist) - forecast_horizon
alchemist_train = alchemist['Volume'].iloc[:split_index]
alchemist_test = alchemist['Volume'].iloc[split_index:]

print("\n" + "="*80)
print("PHASE 1 FIXES - BEFORE vs AFTER COMPARISON")
print("="*80)
print(f"\nDataset: The Alchemist")
print(f"Training samples: {len(alchemist_train)}")
print(f"Test samples: {len(alchemist_test)}")
print(f"Forecast horizon: {forecast_horizon} weeks\n")

# Helper function
def create_sequences(lookback, forecast, data):
    X, y = [], []
    for i in range(lookback, len(data) - forecast + 1):
        X.append(data[i - lookback:i])
        y.append(data[i:i + forecast])
    return np.array(X), np.array(y)

def create_simple_lstm():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(52, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.compile(loss='mse', optimizer='adam')
    return model

lookback = 52
forecast = 32

# Prepare training data
train_data = alchemist_train.values.reshape(-1, 1)
test_data = alchemist_test.values.reshape(-1, 1)

X_train, y_train = create_sequences(lookback, forecast, train_data)

print("-"*80)
print("BEFORE (Original Code - WRONG)")
print("-"*80)
print("Issue: Evaluating on TRAINING data (data leakage)")

# BEFORE: Wrong scaling (creates new scaler for output)
scaler_before = StandardScaler()
X_train_scaled_before = scaler_before.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
y_train_scaled_before = scaler_before.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)  # WRONG

model_before = create_simple_lstm()
model_before.fit(X_train_scaled_before, y_train_scaled_before, epochs=20, batch_size=32, verbose=0)

# BEFORE: Predict on TRAINING data (last 32 from training)
train_predictions_before = model_before.predict(X_train_scaled_before[-1:], verbose=0)
train_predictions_before = scaler_before.inverse_transform(train_predictions_before.reshape(-1, 1))

# This compares predictions to TRAINING data
mae_before = mean_absolute_error(alchemist_train.values[-32:], train_predictions_before.flatten())
mape_before = mean_absolute_percentage_error(alchemist_train.values[-32:], train_predictions_before.flatten())

print(f"MAE:  {mae_before:.2f}  (on training data - artificially good)")
print(f"MAPE: {mape_before:.4f}  (on training data - artificially good)")
print("⚠️  These numbers look good but are MEANINGLESS - they're on training data!")

print("\n" + "-"*80)
print("AFTER (Fixed Code - CORRECT)")
print("-"*80)
print("Fix: Evaluating on actual TEST data (out-of-sample)")

# AFTER: Correct scaling (use same scaler for input and output)
scaler_after = StandardScaler()
X_train_scaled_after = scaler_after.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
y_train_scaled_after = scaler_after.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)  # FIXED

model_after = create_simple_lstm()
model_after.fit(X_train_scaled_after, y_train_scaled_after, epochs=20, batch_size=32, verbose=0)

# AFTER: Create test sequence and predict on TEST data
full_data = np.concatenate([train_data, test_data])
X_test, y_test = create_sequences(lookback, forecast, full_data)
X_test_last = X_test[-1:]  # Last sequence for forecasting

X_test_scaled = scaler_after.transform(X_test_last.reshape(-1, 1)).reshape(X_test_last.shape)
test_predictions_after = model_after.predict(X_test_scaled, verbose=0)
test_predictions_after = scaler_after.inverse_transform(test_predictions_after.reshape(-1, 1))

# This compares predictions to ACTUAL TEST data
mae_after = mean_absolute_error(alchemist_test.values, test_predictions_after.flatten())
mape_after = mean_absolute_percentage_error(alchemist_test.values, test_predictions_after.flatten())

print(f"MAE:  {mae_after:.2f}  (on actual test data - true performance)")
print(f"MAPE: {mape_after:.4f}  (on actual test data - true performance)")
print("✓  These numbers reflect REAL forecasting performance!")

print("\n" + "="*80)
print("SUMMARY OF CRITICAL FIXES")
print("="*80)
print("\n1. ✓ Fixed train/test evaluation - now predicting on actual test data")
print("2. ✓ Fixed scaling - using same scaler for inputs and outputs")
print("3. ✓ Creating proper test sequences from concatenated data")
print("\nThe 'BEFORE' metrics were artificially good because they measured")
print("how well the model fit the training data (overfitting).")
print("\nThe 'AFTER' metrics show TRUE forecasting performance on unseen data.")
print("This is the honest measure of model quality.")
print("\n" + "="*80)
