"""Configuration constants and book definitions for the forecasting pipeline."""

import multiprocessing
from dataclasses import dataclass


@dataclass(frozen=True)
class BookConfig:
    title_match: str
    short_name: str
    display_name: str


BOOKS = [
    BookConfig('Alchemist, The', 'alchemist', 'The Alchemist'),
    BookConfig('Very Hungry Caterpillar, The', 'caterpillar', 'The Very Hungry Caterpillar'),
]

# Data paths
DATA_DIR = 'data'
SALES_FILE = 'UK_Weekly_Trended_Timeline_from_200101_202429.xlsx'
ISBN_FILE = 'ISBN_List.xlsx'
PLOT_DIR = 'results/plots'

# Runtime artifact directories for the encoder-decoder LSTM showcase
TB_LOG_DIR = 'results/tb_logs'
SAVED_MODEL_DIR = 'results/saved_models'
CHECKPOINT_DIR = 'results/checkpoints'

# Time series parameters
START_DATE = '2012-01-01'
FORECAST_HORIZON_WEEKS = 32
LOOKBACK_WEEKS = 52
MONTHLY_FORECAST_MONTHS = 8
MONTHLY_TRAIN_RATIO = 0.8

# LSTM parameters
LSTM_UNITS = 50
LSTM_DROPOUT = 0.3
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10

# Optuna parameters
OPTUNA_TRIALS = 10
OPTUNA_EPOCHS = 10
OPTUNA_RETRAIN_EPOCHS = 30

# XGBoost parameters
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 7
XGB_LEARNING_RATE = 0.1
XGB_GRID_WINDOW_LENGTHS = [12, 24, 32, 52]

# Monthly XGBoost
MONTHLY_XGB_N_ESTIMATORS = 200
MONTHLY_XGB_MAX_DEPTH = 3
MONTHLY_XGB_LEARNING_RATE = 0.1

# Hybrid model parameters
HYBRID_EPOCHS = 50
HYBRID_PATIENCE = 8
PARALLEL_SARIMA_WEIGHT = 0.7

# Encoder-decoder LSTM (with attention) parameters
ENC_DEC_ENCODER_UNITS = 64
ENC_DEC_DECODER_UNITS = 64
ENC_DEC_VAL_SPLIT = 0.2
ENC_DEC_LR_PATIENCE = 5
ENC_DEC_LR_FACTOR = 0.5
ENC_DEC_MIN_LR = 1e-5

# Hyperopt tuning
HYPEROPT_MAX_EVALS = 100

# CPU cores
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# Plot filename mapping (model_key, book_short_name) -> filename
PLOT_NAMES = {
    ('lstm', 'alchemist'): '01_lstm_alchemist_forecast.png',
    ('lstm', 'caterpillar'): '02_lstm_caterpillar_forecast.png',
    ('optuna', 'alchemist'): '03_optuna_alchemist_forecast.png',
    ('optuna', 'caterpillar'): '04_optuna_caterpillar_forecast.png',
    ('seq_hybrid', 'alchemist'): '05_sequential_hybrid_alchemist.png',
    ('seq_hybrid', 'caterpillar'): '06_sequential_hybrid_caterpillar.png',
    ('par_hybrid', 'alchemist'): '07_parallel_hybrid_alchemist.png',
    ('monthly_xgb', 'alchemist'): '08_monthly_xgboost_alchemist.png',
    ('monthly_xgb', 'caterpillar'): '09_monthly_xgboost_caterpillar.png',
    ('encoder_decoder', 'alchemist'): '10_encoder_decoder_alchemist_forecast.png',
    ('encoder_decoder', 'caterpillar'): '11_encoder_decoder_caterpillar_forecast.png',
}
