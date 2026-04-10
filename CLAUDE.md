# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time series forecasting pipeline for UK book sales (Nielsen BookScan data, weekly 2001–2024). Focuses on two books: *The Alchemist* and *The Very Hungry Caterpillar*. Runs 8 model types per book, evaluates on held-out test data, and saves 9 plots.

## Commands

```bash
# Setup
python bootstrap.py                      # Install deps + create dirs + check data
pip install -r requirements.txt          # Dependencies only

# Run full pipeline (~10-30 min, Optuna trials dominate)
python run_analysis.py

# Tests
python -m pytest tests/ -v
```

Output goes to `results/plots/` (9 PNGs).

## Architecture

```
run_analysis.py          # Entry point (env vars, matplotlib backend)
  └─ pipeline.py         # Orchestration: loops books, calls models, saves plots
       ├─ config.py      # All constants, book definitions, plot name mapping
       ├─ data_loader.py # Excel loading, preprocessing, train/test split
       └─ models.py      # Model builders, training wrappers, evaluation
```

**Data flow:**
1. `data_loader.prepare_all_books()` loads Excel, preprocesses, splits per book
2. `pipeline.run_all_models()` runs each model type for one book, collects results
3. Loop over `config.BOOKS` drives the per-book orchestration
4. Plots saved to `results/plots/` using `config.PLOT_NAMES` mapping

**Models (per book):**
- Auto ARIMA/SARIMA (pmdarima, m=52 weekly / m=12 monthly)
- XGBoost with Hyperopt tuning (100 trials) + sktime grid search
- LSTM (2 layers × 50 units, StandardScaler, lookback=52, forecast=32)
- Optuna-optimized LSTM (10 trials searching units/dropout)
- Sequential hybrid: SARIMA base + LSTM on residuals
- Parallel hybrid: weighted SARIMA + LSTM (0.7/0.3 default)
- Monthly XGBoost (weekly → monthly aggregation)
- Monthly ARIMA (8-month horizon)

**Key parameters** are centralized in `config.py` — not scattered as magic numbers.

## Data Requirements

Two Excel files must exist in `data/`:
- `UK_Weekly_Trended_Timeline_from_200101_202429.xlsx` (~50MB, 4 sheets)
- `ISBN_List.xlsx` (~5MB, 4 sheets)

Both contain Fiction, Non-Fiction, Children's Books, Educational sheets. Not included in repo.

## Environment

- Python 3.8+, 8GB RAM recommended, GPU optional
- TensorFlow 2.15 (pinned), uses `Agg` matplotlib backend (non-interactive)
- `lightgbm` commented out due to macOS build issues
- `venv/` is gitignored — create locally
- macOS threading fix (`OMP_NUM_THREADS=1`) set in `run_analysis.py`
