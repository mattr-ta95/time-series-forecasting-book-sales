"""Data loading, preprocessing, and book extraction."""

import os

import numpy as np
import pandas as pd

from config import (
    BOOKS, DATA_DIR, FORECAST_HORIZON_WEEKS, ISBN_FILE, SALES_FILE, START_DATE,
)


def load_raw_data():
    """Load and combine all Excel sheets into unified DataFrames."""
    sales_path = os.path.join(DATA_DIR, SALES_FILE)
    isbn_path = os.path.join(DATA_DIR, ISBN_FILE)

    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"Sales data file not found: {sales_path}")
    if not os.path.exists(isbn_path):
        raise FileNotFoundError(f"ISBN data file not found: {isbn_path}")

    data_dict = pd.read_excel(sales_path, sheet_name=None)
    data_isbn = pd.read_excel(isbn_path, sheet_name=None)

    print(data_dict.keys())
    print(data_isbn.keys())

    df_weekly = pd.concat(
        [df.assign(Category=sheet_name) for sheet_name, df in data_dict.items()],
        ignore_index=True,
    )
    df_isbn = pd.concat(
        [df.assign(Category=sheet_name) for sheet_name, df in data_isbn.items()],
        ignore_index=True,
    )

    df_weekly.info()
    df_isbn.info()

    return df_weekly, df_isbn


def preprocess_weekly(df_weekly):
    """Resample to weekly frequency, fill missing weeks with zero, convert types."""
    df = df_weekly.copy()
    df['End Date'] = pd.to_datetime(df['End Date'])
    df = df.set_index('End Date')

    df_full = df.groupby(['Title', 'Category', 'ISBN'])[['Volume']].resample('W').sum().fillna(0)
    df_full = df_full.reset_index()

    df_full['ISBN'] = df_full['ISBN'].astype(str)
    df_full['End Date'] = pd.to_datetime(df_full['End Date'])

    return df_full


def extract_book_series(df_full, title_match, start_date=START_DATE):
    """Extract, resample, and filter time series for a single book."""
    book = df_full[df_full['Title'].isin([title_match])]
    book = book.set_index('End Date')
    book = book.resample('W').sum().ffill()
    book = book[book.index > start_date]
    return book


def split_train_test(series, horizon=FORECAST_HORIZON_WEEKS):
    """Split a Series by holding out the last `horizon` points for testing."""
    split_idx = len(series) - horizon
    return series.iloc[:split_idx], series.iloc[split_idx:]


def aggregate_monthly(train_series):
    """Aggregate weekly training data to monthly frequency (month-start)."""
    s = train_series.copy()
    if isinstance(s.index, pd.PeriodIndex):
        s.index = s.index.to_timestamp()
    return s.groupby(pd.Grouper(freq='MS')).sum()


def prepare_all_books():
    """Load data and prepare train/test splits for all configured books.

    Returns:
        dict mapping book short_name to {'full_df', 'volume', 'train', 'test', 'config'}.
    """
    df_weekly, _ = load_raw_data()
    df_full = preprocess_weekly(df_weekly)

    books_data = {}
    for book in BOOKS:
        book_df = extract_book_series(df_full, book.title_match)
        volume = book_df['Volume']
        train, test = split_train_test(volume)

        books_data[book.short_name] = {
            'full_df': book_df,
            'volume': volume,
            'train': train,
            'test': test,
            'config': book,
        }
        print(f"  {book.display_name}: {len(train)} train, {len(test)} test weeks")

    return books_data
