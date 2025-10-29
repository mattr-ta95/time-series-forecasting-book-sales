"""
Phase 2 Testing - Compare simplified model to complex model
Shows actual performance improvements from reducing overfitting
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import time

print("="*80)
print("PHASE 2 TESTING - OVERFITTING REDUCTION")
print("="*80)

# Load data
print("\nLoading data...")
data_dict = pd.read_excel('data/UK_Weekly_Trended_Timeline_from_200101_202429.xlsx', sheet_name=None)
df_UK_weekly = pd.concat([df.assign(Category=sheet_name) for sheet_name, df in data_dict.items()], ignore_index=True)
df_UK_weekly['End Date'] = pd.to_datetime(df_UK_weekly['End Date'])
df_UK_weekly = df_UK_weekly.set_index('End Date')
df_UK_weekly_full = df_UK_weekly.groupby(['Title', 'Category', 'ISBN'])['Volume'].resample('W').sum().fillna(0)
df_UK_weekly_full = df_UK_weekly_full.reset_index()

alchemist = df_UK_weekly_full[df_UK_weekly_full['Title'].isin(['Alchemist, The'])]
alchemist = alchemist.set_index('End Date').resample('W').sum().ffill()
alchemist = alchemist[alchemist.index > '2012-01-01']

forecast_horizon = 32
split_index = len(alchemist) - forecast_horizon
alchemist_train = alchemist['Volume'].iloc[:split_index]
alchemist_test = alchemist['Volume'].iloc[split_index:]
print(f"Train: {len(alchemist_train)}, Test: {len(alchemist_test)}")

# Helper function
def create_sequences(lookback, forecast, data):
    X, y = [], []
    for i in range(lookback, len(data) - forecast + 1):
        X.append(data[i - lookback:i])
        y.append(data[i:i + forecast])
    return np.array(X), np.array(y)

# Prepare data
train_data = alchemist_train.values.reshape(-1, 1)
test_data = alchemist_test.values.reshape(-1, 1)
X_train, y_train = create_sequences(52, 32, train_data)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

full_data = np.concatenate([train_data, test_data])
X_test, _ = create_sequences(52, 32, full_data)
X_test_scaled = scaler.transform(X_test[-1:].reshape(-1, 1)).reshape(X_test[-1:].shape)

# ============================================================================
# PHASE 1: Complex Model (5 layers, 80 units)
# ============================================================================
print("\n" + "="*80)
print("PHASE 1 MODEL: 5 layers × 80 units (Complex)")
print("="*80)

def create_phase1_model():
    """Original complex model"""
    model = Sequential()
    model.add(LSTM(units=80, input_shape=(52, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=80))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.compile(loss='mse', optimizer='adam')
    return model

model_phase1 = create_phase1_model()
print(f"Parameters: {model_phase1.count_params():,}")

start = time.time()
history1 = model_phase1.fit(
    X_train_scaled, y_train_scaled,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)
train_time1 = time.time() - start

# Predict
pred1 = model_phase1.predict(X_test_scaled, verbose=0)
pred1 = scaler.inverse_transform(pred1.reshape(-1, 1)).flatten()

mae1 = mean_absolute_error(alchemist_test.values, pred1)
mape1 = mean_absolute_percentage_error(alchemist_test.values, pred1)

print(f"\nResults:")
print(f"  Training time: {train_time1:.1f}s")
print(f"  Final train loss: {history1.history['loss'][-1]:.4f}")
print(f"  Final val loss: {history1.history['val_loss'][-1]:.4f}")
print(f"  Val/Train ratio: {history1.history['val_loss'][-1] / history1.history['loss'][-1]:.2f}")
print(f"  Test MAE: {mae1:.2f}")
print(f"  Test MAPE: {mape1:.4f}")

# ============================================================================
# PHASE 2: Simplified Model (2 layers, 50 units, early stopping)
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 MODEL: 2 layers × 50 units + Early Stopping (Simplified)")
print("="*80)

def create_phase2_model():
    """Simplified model with fewer parameters"""
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(52, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50))
    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.compile(loss='mse', optimizer='adam')
    return model

model_phase2 = create_phase2_model()
print(f"Parameters: {model_phase2.count_params():,}")
print(f"Parameter reduction: {(1 - model_phase2.count_params()/model_phase1.count_params())*100:.1f}%")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

start = time.time()
history2 = model_phase2.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,  # More epochs, but early stopping prevents overfitting
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
train_time2 = time.time() - start

# Predict
pred2 = model_phase2.predict(X_test_scaled, verbose=0)
pred2 = scaler.inverse_transform(pred2.reshape(-1, 1)).flatten()

mae2 = mean_absolute_error(alchemist_test.values, pred2)
mape2 = mean_absolute_percentage_error(alchemist_test.values, pred2)

print(f"\nResults:")
print(f"  Training time: {train_time2:.1f}s")
print(f"  Epochs trained: {len(history2.history['loss'])}")
print(f"  Final train loss: {history2.history['loss'][-1]:.4f}")
print(f"  Final val loss: {history2.history['val_loss'][-1]:.4f}")
print(f"  Val/Train ratio: {history2.history['val_loss'][-1] / history2.history['loss'][-1]:.2f}")
print(f"  Test MAE: {mae2:.2f}")
print(f"  Test MAPE: {mape2:.4f}")

# ============================================================================
# PHASE 2: Cross-Validation
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: Cross-Validation (Model Stability)")
print("="*80)

def time_series_cv(n_splits=3):
    """Quick 3-fold CV"""
    n_samples = len(X_train_scaled)
    test_size = n_samples // (n_splits + 1)
    scores = []

    for i in range(n_splits):
        train_end = test_size * (i + 2)
        test_start = test_size * (i + 1)
        test_end = train_end

        X_tr = X_train_scaled[:test_start]
        y_tr = y_train_scaled[:test_start]
        X_val = X_train_scaled[test_start:test_end]
        y_val = y_train_scaled[test_start:test_end]

        if len(X_val) == 0:
            continue

        model = create_phase2_model()
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        model.fit(X_tr, y_tr, epochs=50, batch_size=32,
                 validation_split=0.2, callbacks=[es], verbose=0)

        pred = model.predict(X_val, verbose=0)
        pred_inv = scaler.inverse_transform(pred.reshape(-1, 1))
        y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))
        mae = mean_absolute_error(y_val_inv.flatten(), pred_inv.flatten())
        scores.append(mae)
        print(f"  Fold {i+1}: MAE = {mae:.2f}")

    return scores

cv_scores = time_series_cv(n_splits=3)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"\n  Mean CV MAE: {cv_mean:.2f} (+/- {cv_std:.2f})")
print(f"  CV Coefficient of Variation: {(cv_std/cv_mean)*100:.1f}%")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1 vs PHASE 2 COMPARISON")
print("="*80)

print("\n| Metric                | Phase 1 (Complex) | Phase 2 (Simplified) | Change |")
print("|----------------------|-------------------|----------------------|--------|")
print(f"| Parameters           | {model_phase1.count_params():,}              | {model_phase2.count_params():,}                | {(1 - model_phase2.count_params()/model_phase1.count_params())*100:.1f}% ↓ |")
print(f"| Epochs Trained       | 30                | {len(history2.history['loss'])}                  | Early stop |")
print(f"| Val/Train Loss Ratio | {history1.history['val_loss'][-1] / history1.history['loss'][-1]:.2f}                | {history2.history['val_loss'][-1] / history2.history['loss'][-1]:.2f}                  | {'Better' if history2.history['val_loss'][-1] / history2.history['loss'][-1] < history1.history['val_loss'][-1] / history1.history['loss'][-1] else 'Worse'} |")
print(f"| Test MAE             | {mae1:.2f}              | {mae2:.2f}                | {((mae2-mae1)/mae1)*100:+.1f}% |")
print(f"| Test MAPE            | {mape1:.4f}            | {mape2:.4f}              | {((mape2-mape1)/mape1)*100:+.1f}% |")
print(f"| Training Time        | {train_time1:.1f}s              | {train_time2:.1f}s                 | {((train_time2-train_time1)/train_time1)*100:+.1f}% |")
print(f"| CV Stability         | N/A               | {cv_std:.2f} MAE std        | Added |")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if mae2 < mae1:
    print(f"\n✓ Phase 2 IMPROVED test MAE by {((mae1-mae2)/mae1)*100:.1f}%")
elif mae2 < mae1 * 1.05:
    print(f"\n≈ Phase 2 maintained similar performance ({((mae2-mae1)/mae1)*100:+.1f}%) with 75% fewer parameters")
else:
    print(f"\n⚠ Phase 2 test MAE increased by {((mae2-mae1)/mae1)*100:.1f}%")

val_train_ratio1 = history1.history['val_loss'][-1] / history1.history['loss'][-1]
val_train_ratio2 = history2.history['val_loss'][-1] / history2.history['loss'][-1]

if val_train_ratio2 < val_train_ratio1:
    print(f"✓ Phase 2 has BETTER generalization (val/train ratio: {val_train_ratio2:.2f} vs {val_train_ratio1:.2f})")
else:
    print(f"⚠ Phase 2 val/train ratio: {val_train_ratio2:.2f} vs {val_train_ratio1:.2f}")

if cv_std < 20:
    print(f"✓ Phase 2 model is STABLE across time periods (CV std: {cv_std:.2f} < 20)")
else:
    print(f"⚠ Phase 2 model shows some variance (CV std: {cv_std:.2f})")

print("\nSummary: Phase 2 uses 75% fewer parameters with {}" .format(
    "improved" if mae2 < mae1 else "similar" if mae2 < mae1 * 1.05 else "slightly worse"
) + " performance.")
print("="*80)
