# Time Series Forecasting for Book Sales Analysis

A comprehensive time series analysis project that uses various forecasting techniques to predict book sales and demand patterns. This project analyzes historical sales data from Nielsen BookScan to help publishers make data-driven decisions about stock control, investment, and reprinting strategies.

## Project Overview

This project focuses on utilizing time series analysis techniques to forecast sales and demand for books based on historical sales data. The goal is to identify sales patterns, particularly seasonal trends and long-term potential, to inform data-driven decisions for independent publishers.

### Business Context

Nielsen BookScan, the world's largest continuous book sales tracking service, provides detailed and accurate sales information. This project aims to develop insights for small to medium-sized independent publishers to help them make informed decisions based on historical sales data.

### Key Features

- **Classical Time Series Analysis**: ARIMA, SARIMA, and seasonal decomposition
- **Machine Learning Models**: XGBoost with hyperparameter tuning
- **Deep Learning**: LSTM networks with KerasTuner optimization
- **Hybrid Models**: Sequential and parallel combinations of SARIMA and LSTM
- **Multi-frequency Analysis**: Both weekly and monthly forecasting
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
git clone https://github.com/yourusername/time-series-forecasting.git
cd time-series-forecasting
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the datasets and place them in the `data/` directory:
   - `UK_Weekly_Trended_Timeline_from_200101_202429.xlsx`
   - `ISBN_List.xlsx`

## Usage

### Running the Analysis

1. **Data Preparation**:
   - The script automatically loads and preprocesses the data
   - Resamples weekly data to fill missing weeks with zeros
   - Filters data from 2012 onwards for analysis

2. **Classical Techniques**:
   - Time series decomposition (additive/multiplicative)
   - ACF/PACF analysis for seasonality detection
   - Stationarity testing using ADF test
   - Auto ARIMA model selection and forecasting

3. **Machine Learning Models**:
   - XGBoost with grid search hyperparameter tuning
   - LSTM networks with KerasTuner optimization
   - Cross-validation and performance evaluation

4. **Hybrid Models**:
   - Sequential: LSTM forecasts SARIMA residuals
   - Parallel: Weighted combination of SARIMA and LSTM predictions

5. **Monthly Forecasting**:
   - Aggregated weekly data to monthly
   - XGBoost and SARIMA models for 8-month forecasts

### Running the Analysis

```bash
python run_analysis.py
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
time-series-forecasting/
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
├── docs/                        # Phase summaries, contribution guide, historical scripts
├── results/plots/               # Generated forecast visualizations (11 PNGs)
└── time_series_forecasting_analysis_original.py  # Frozen original script
```

## Key Results

### Weekly Forecast Performance (32-week horizon)

#### The Alchemist

| Model | MAE | MAPE | RMSE |
|-------|-----|------|------|
| **Auto ARIMA** | **155** | **0.298** | **226** |
| **Sequential Hybrid** | **155** | **0.298** | **226** |
| XGBoost | 156 | 0.330 | 218 |
| Parallel Hybrid | 165 | 0.316 | 237 |
| Optuna LSTM | 167 | 0.306 | 229 |
| LSTM | 223 | 0.379 | 360 |

#### The Very Hungry Caterpillar

| Model | MAE | MAPE | RMSE |
|-------|-----|------|------|
| **Auto ARIMA** | **349** | **0.187** | **443** |
| **Sequential Hybrid** | **349** | **0.187** | **443** |
| Parallel Hybrid | 382 | 0.191 | 493 |
| XGBoost | 416 | 0.212 | 517 |
| Optuna LSTM | 740 | 0.310 | 910 |
| LSTM | 769 | 0.326 | 914 |

### Key Insights

1. **Classical models win at this data scale**: Auto ARIMA matches or beats every neural network with only ~600 weekly data points
2. **Sequential hybrid adds nothing**: SARIMA + LSTM residuals produces near-identical results to ARIMA alone — the LSTM can't find signal in the residuals at this sample size
3. **Caterpillar is more predictable**: MAPE of 18.7% vs 29.8% for The Alchemist, driven by stronger seasonal patterns (children's book with Christmas/Easter spikes)
4. **XGBoost is competitive weekly but poor monthly**: Matches ARIMA at weekly frequency but the timestamp-as-feature approach fails at monthly granularity

### Synthetic Data Experiment: Does More Data Fix Deep Learning?

The LSTM's poor performance raised the question: is the architecture wrong, or is ~600 data points simply too few? To test this, we generated synthetic book sales data at multiple scales using STL decomposition of the real Caterpillar data, preserving its trend, seasonality, and noise characteristics.

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

**Conclusion**: At ~10,000 data points, the LSTM approaches ARIMA performance (MAPE 0.19 vs 0.18). The bottleneck is data scarcity, not model architecture. For publishers with access to daily sales data (which would provide ~7,300 points over 20 years), deep learning becomes a viable alternative to classical methods.

## Methodology

### Data Preprocessing
- Weekly data resampling with zero-filling for missing weeks
- ISBN conversion to string format
- Date standardization and indexing
- Filtering for books with data beyond 2024-07-01

### Model Selection
- **Auto ARIMA**: Automatic parameter selection with seasonal components
- **XGBoost**: Gradient boosting with detrending and deseasonalization
- **LSTM**: Deep learning with 2-layer architecture and dropout regularization
- **Hybrid**: Combination approaches for improved accuracy

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

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Nielsen BookScan for providing the sales data
- Cambridge Institute for the project framework
- Open source community for the excellent libraries used

## Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].

## Future Enhancements

- [ ] Add more books to the analysis
- [ ] Implement additional forecasting models (Prophet, NeuralProphet)
- [ ] Create interactive dashboards
- [ ] Add real-time forecasting capabilities
- [ ] Implement ensemble methods
- [ ] Add confidence interval visualization
