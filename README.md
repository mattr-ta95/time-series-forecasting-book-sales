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

### Running the Script

```bash
python time_series_forecasting_analysis.py
```

## Project Structure

```
time-series-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── UK_Weekly_Trended_Timeline_from_200101_202429.xlsx
│   └── ISBN_List.xlsx
├── time_series_forecasting_analysis.py
└── results/
    ├── plots/
    └── models/
```

## Key Results

### Model Performance Comparison

| Model | Book | MAE | MAPE | RMSE |
|-------|------|-----|------|------|
| Auto ARIMA | The Alchemist | - | - | - |
| Auto ARIMA | The Very Hungry Caterpillar | - | - | - |
| XGBoost | The Alchemist | - | - | - |
| XGBoost | The Very Hungry Caterpillar | - | - | - |
| LSTM | The Alchemist | - | - | - |
| LSTM | The Very Hungry Caterpillar | - | - | - |
| Hybrid (Sequential) | The Alchemist | - | - | - |
| Hybrid (Parallel) | The Alchemist | - | - | - |

### Key Insights

1. **Seasonality**: Both books show clear seasonal patterns with yearly cycles
2. **Trend Analysis**: Different books exhibit varying trend characteristics
3. **Model Performance**: Hybrid models often outperform individual models
4. **Forecasting Horizon**: 32-week forecasts provide good balance of accuracy and utility

## Methodology

### Data Preprocessing
- Weekly data resampling with zero-filling for missing weeks
- ISBN conversion to string format
- Date standardization and indexing
- Filtering for books with data beyond 2024-07-01

### Model Selection
- **Auto ARIMA**: Automatic parameter selection with seasonal components
- **XGBoost**: Gradient boosting with detrending and deseasonalization
- **LSTM**: Deep learning with bidirectional layers and regularization
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
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: tensorflow, keras, keras-tuner
- **Optimization**: hyperopt

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
