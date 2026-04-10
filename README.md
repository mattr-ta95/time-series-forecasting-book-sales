# Time Series Forecasting for Book Sales Analysis

A time series forecasting project that benchmarks classical, machine learning, and deep learning models on UK Nielsen BookScan weekly sales data, with a controlled synthetic-data experiment quantifying the training volume at which LSTMs become competitive with Auto ARIMA.

## Project Overview

The goal is to identify sales patterns — particularly seasonal trends and long-term potential — and to understand which class of model is appropriate at the data volume available to small and medium-sized independent publishers.

### Business Context

Nielsen BookScan is the world's largest continuous book sales tracking service. This project develops insights for small to medium-sized independent publishers who need to make stocking, investment, and reprinting decisions from historical sales data.

### Key Features

- **Classical Time Series Analysis**: ARIMA, SARIMA, and seasonal decomposition
- **Machine Learning Models**: XGBoost with hyperparameter tuning
- **Deep Learning**: LSTM networks with Optuna hyperparameter optimization
- **Hybrid Models**: Sequential and parallel combinations of SARIMA and LSTM
- **Multi-frequency Analysis**: Both weekly and monthly forecasting
- **Synthetic Data Scaling Study**: Controlled experiment on how much data LSTMs need to match classical models
- **Comprehensive Evaluation**: MAE, MAPE, RMSE metrics across all models

## Dataset

The project uses two main datasets:
- `UK_Weekly_Trended_Timeline_from_200101_202429.xlsx`: Weekly sales data for various books
- `ISBN_List.xlsx`: Metadata for books including ISBN, Title, Author, etc.

### Sample Books Analyzed
- **The Alchemist** by Paulo Coelho
- **The Very Hungry Caterpillar** by Eric Carle

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mattr-ta95/time-series-forecasting-book-sales.git
cd time-series-forecasting-book-sales
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the datasets and place them in the `data/` directory:
   - `UK_Weekly_Trended_Timeline_from_200101_202429.xlsx`
   - `ISBN_List.xlsx`

## Usage

The pipeline preprocesses weekly data (zero-filling missing weeks, filtering from 2012), then runs classical (Auto ARIMA, STL decomposition, ADF stationarity), machine learning (XGBoost with grid search), deep learning (LSTM with Optuna), and hybrid models. Monthly forecasts (8-month horizon) are produced from aggregated weekly data.

```bash
python run_analysis.py      # Full pipeline
python -m pytest tests/ -v  # Tests
```

## Project Structure

```
time-series-forecasting-book-sales/
├── run_analysis.py              # Entry point — runs the full pipeline
├── synthetic_experiment.py      # Data volume scaling experiment (LSTM vs data size)
├── config.py                    # All constants and book definitions
├── data_loader.py               # Data loading, preprocessing, book extraction
├── models.py                    # Model builders, training functions, evaluation
├── pipeline.py                  # Orchestration: per-book model loop, plots, summary
├── bootstrap.py                 # Environment setup script
├── requirements.txt
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_data_loader.py
│   └── test_models.py
├── data/
│   ├── UK_Weekly_Trended_Timeline_from_200101_202429.xlsx
│   └── ISBN_List.xlsx
├── docs/                        # Supporting documentation
└── results/plots/               # Generated forecast visualizations
```

## Results: A Case Study in Deep Learning at Small Data Scale

**Research question**: How much data does an LSTM need to match classical forecasting on weekly retail sales?

Weekly Nielsen BookScan data gives about 600 observations per book across 2012–2024, well below the sample size at which recurrent architectures typically excel. The headline question was therefore not "which model wins?" but "at what data volume does deep learning become competitive?" The synthetic scaling experiment below is the core finding; the per-book benchmarks show the expected small-sample ordering.

### Synthetic Data Experiment: When Does the LSTM Catch Up?

To isolate data volume from architecture, synthetic book sales series were generated at multiple scales using STL decomposition of the real Caterpillar data, preserving its trend, seasonality, and noise. The same LSTM architecture was then trained at each data size.

```bash
python synthetic_experiment.py
```

| Training Data | Data Points | LSTM MAPE | vs Real LSTM |
|--------------|-------------|-----------|--------------|
| Real data | ~600 | 0.324 | baseline |
| Synthetic 2K | 2,084 | 0.215 | -33% |
| Synthetic 5K | 5,084 | 0.251 | -23% |
| Synthetic 10K | 10,084 | **0.193** | **-40%** |
| Auto ARIMA (real) | ~600 | 0.181 | target |

**Finding**: At roughly 10,000 training points the LSTM converges to within one percentage point of Auto ARIMA (MAPE 0.193 vs 0.181). The bottleneck on the real data is sample size, not architecture or training procedure. Daily sales data (~7,300 observations over 20 years) would put publishers close to the threshold where deep learning becomes a viable alternative to classical methods.

### Baseline Benchmarks on ~600 Data Points

With the data-volume threshold quantified, the per-book tables below show the expected small-sample ordering: classical models should, and do, win. Deep learning rows appear first to reflect the focus of the work; classical baselines appear last.

#### The Alchemist (32-week horizon)

| Model | MAE | MAPE | RMSE |
|-------|-----|------|------|
| LSTM | 223 | 0.379 | 360 |
| Optuna LSTM | 167 | 0.306 | 229 |
| Sequential Hybrid (SARIMA + LSTM residuals) | **155** | **0.298** | **226** |
| Parallel Hybrid (weighted SARIMA + LSTM) | 165 | 0.316 | 237 |
| XGBoost | 156 | 0.330 | 218 |
| **Auto ARIMA** | **155** | **0.298** | **226** |

#### The Very Hungry Caterpillar (32-week horizon)

| Model | MAE | MAPE | RMSE |
|-------|-----|------|------|
| LSTM | 769 | 0.326 | 914 |
| Optuna LSTM | 740 | 0.310 | 910 |
| Sequential Hybrid (SARIMA + LSTM residuals) | **349** | **0.187** | **443** |
| Parallel Hybrid (weighted SARIMA + LSTM) | 382 | 0.191 | 493 |
| XGBoost | 416 | 0.212 | 517 |
| **Auto ARIMA** | **349** | **0.187** | **443** |

Auto ARIMA is the best-performing model at this data scale, matched only by the Sequential Hybrid (which reduces to ARIMA when the LSTM finds no signal in the residuals). This ordering is consistent with the synthetic experiment: at N≈600 the LSTM is data-starved.

### Key Insights

1. **Quantified data threshold (~10K samples).** The synthetic experiment pins down roughly 10,000 training points as the size at which a vanilla LSTM becomes competitive with Auto ARIMA on weekly retail sales — directly actionable for publishers deciding whether to invest in daily-frequency data pipelines.
2. **Architecture ruled out as the cause of LSTM underperformance.** Holding the model fixed and varying only the training set size is a controlled test: LSTM MAPE drops from 0.324 at 600 points to 0.193 at 10K, confirming the bottleneck is data volume, not model capacity.
3. **Daily-frequency data would make LSTMs viable.** Twenty years of daily sales yields about 7,300 observations, close enough to the 10K threshold to expect competitive deep learning performance.
4. **Classical models remain the right choice on weekly data.** Auto ARIMA should continue to be the production baseline at Nielsen BookScan granularity; the Sequential Hybrid adds no signal because the LSTM cannot find structure in the ARIMA residuals at N≈600.
5. **Caterpillar is more predictable than Alchemist** (MAPE 18.7% vs 29.8%) due to stronger seasonal patterns — a children's book with Christmas and Easter spikes.

## Methodology

### Data Preprocessing
Weekly resampling with zero-filling, ISBN normalization, date indexing, filtered for books with data beyond 2024-07-01.

### Model Selection
- **Auto ARIMA**: Automatic parameter selection with seasonal components (`m=52` weekly, `m=12` monthly)
- **XGBoost**: Gradient boosting with detrending and deseasonalization
- **LSTM**: 2 × 50-unit stacked layers with dropout, StandardScaler, lookback=52
- **Hybrid**: Sequential (SARIMA + LSTM residuals) and parallel (weighted ensemble) approaches

### TensorFlow / Deep Learning Work

The deep learning components are the focus of this repo:

- **Multi-step (32-week) LSTM forecasting** with 52-week lookback, StandardScaler, 2 × 50-unit stacked layers, dropout, and early stopping (see `models.py`)
- **Optuna hyperparameter search** over LSTM architecture (units, dropout, learning rate), 10 trials per book
- **Sequential hybrid model**: Auto SARIMA base forecast with an LSTM trained on in-sample residuals, predicting residuals over the 32-week horizon
- **Parallel hybrid model**: weighted ensemble of SARIMA and LSTM forecasts (default 0.7/0.3 split)
- **STL-based synthetic data generation** that scales the training set from 600 to 10,000 points while preserving trend, seasonality, and noise
- **Controlled scaling study** (`synthetic_experiment.py`) holding the LSTM architecture fixed and varying only the training set size to isolate the data-volume bottleneck

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error
- **RMSE (Root Mean Square Error)**: Square root of mean squared error

## Technical Details

### Dependencies
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Time Series**: statsmodels, pmdarima, sktime
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning**: tensorflow, keras
- **Optimization**: hyperopt, optuna

### Hardware Requirements
- Minimum 8GB RAM recommended
- GPU support optional but recommended for LSTM training
- Multi-core CPU for parallel processing

## Contributing

Fork, branch, commit, push, and open a Pull Request.

## License

MIT License — see the [LICENSE](LICENSE) file.

## Acknowledgments

- Nielsen BookScan for the sales data
- Cambridge Institute for the project framework
- Open source community for the libraries used

## Contact

Open a GitHub issue for questions or suggestions.

## Future Enhancements

- [ ] Acquire daily-frequency sales data to test the ~10K threshold predicted by the synthetic experiment
- [ ] Expand the study to additional titles and genres to test whether the data threshold generalizes
- [ ] Benchmark against modern forecasting baselines (Prophet, NeuralProphet, N-BEATS)
- [ ] Add prediction interval estimation for all models
- [ ] Extend the hybrid approach with an encoder-decoder LSTM to better capture long-range dependencies
