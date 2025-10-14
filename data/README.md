# Data Directory

This directory contains the datasets used for the time series forecasting analysis.

## Required Files

### 1. UK_Weekly_Trended_Timeline_from_200101_202429.xlsx
- **Description**: Weekly sales data for various books from 2001 to 2024
- **Format**: Excel file with multiple sheets
- **Sheets**: Contains 4 different categories/sheets of book sales data
- **Columns**:
  - `End Date`: Weekly end date
  - `Volume`: Sales volume for that week
  - `Title`: Book title
  - `Category`: Book category/genre
  - `ISBN`: International Standard Book Number

### 2. ISBN_List.xlsx
- **Description**: Metadata for books including detailed information
- **Format**: Excel file with multiple sheets
- **Sheets**: Contains 4 different categories/sheets matching the sales data
- **Columns**:
  - `ISBN`: International Standard Book Number
  - `Title`: Book title
  - `Author`: Book author
  - `Publisher`: Publishing company
  - Additional metadata fields

## Data Structure

The datasets are organized into 4 categories/sheets:
1. **Fiction**
2. **Non-Fiction**
3. **Children's Books**
4. **Educational**

Each sheet contains books from that specific category with their corresponding sales data and metadata.

## Data Preprocessing

The analysis script performs the following preprocessing steps:

1. **Data Loading**: Reads all sheets from both Excel files
2. **Combination**: Merges all sheets into single DataFrames with category identifiers
3. **Resampling**: Converts to weekly frequency, filling missing weeks with zeros
4. **Data Type Conversion**: 
   - ISBNs converted to string format
   - Dates converted to datetime objects
5. **Filtering**: Removes books with sales data beyond 2024-07-01
6. **Focus Books**: Extracts data for "The Alchemist" and "The Very Hungry Caterpillar"

## Sample Data

### Sales Data Sample
```
End Date    | Volume | Title                    | Category | ISBN
2020-01-05  | 150    | Alchemist, The          | Fiction  | 9780061122415
2020-01-12  | 200    | Alchemist, The          | Fiction  | 9780061122415
2020-01-19  | 175    | Alchemist, The          | Fiction  | 9780061122415
```

### Metadata Sample
```
ISBN           | Title                    | Author        | Publisher
9780061122415  | Alchemist, The          | Paulo Coelho  | HarperOne
9780399226908  | Very Hungry Caterpillar | Eric Carle    | Philomel Books
```

## Data Quality Notes

- **Missing Values**: Weeks with no sales are represented as missing data points
- **Resampling**: The script fills missing weeks with zero sales
- **Date Range**: Data spans from 2001 to 2024
- **Frequency**: Weekly data points
- **Completeness**: Some books may have gaps in their sales data

## Usage in Analysis

The data is used for:
1. **Exploratory Data Analysis**: Understanding sales patterns and trends
2. **Time Series Decomposition**: Separating trend, seasonal, and residual components
3. **Model Training**: Training various forecasting models
4. **Model Evaluation**: Testing model performance on held-out data
5. **Forecasting**: Predicting future sales for 32 weeks ahead

## Data Privacy and Usage

- This data is provided for educational and research purposes
- Nielsen BookScan data is used under appropriate licensing
- No personal information is included in the datasets
- All analysis is performed on aggregated sales data only

## File Size and Performance

- **UK_Weekly_Trended_Timeline_from_200101_202429.xlsx**: ~50MB
- **ISBN_List.xlsx**: ~5MB
- **Memory Usage**: ~200MB when loaded into pandas DataFrames
- **Processing Time**: Initial data loading takes ~30 seconds

## Troubleshooting

### Common Issues:
1. **File Not Found**: Ensure both Excel files are in the `data/` directory
2. **Memory Issues**: Close other applications if experiencing memory problems
3. **Sheet Access**: Verify that all 4 sheets are accessible in both files
4. **Date Format**: Ensure dates are in proper Excel date format

### Data Validation:
The script includes basic data validation:
- Checks for required columns
- Validates date formats
- Ensures ISBN uniqueness
- Verifies data completeness
