"""
==========================================================
Stock Market Index Fetcher & Processor Module
==========================================================

This module fetches monthly stock market index data from 
Yahoo Finance for a set of global indices grouped by currency. 
It structures the output into a time-aligned pandas DataFrame 
with a MultiIndex format for easy analysis.

Structure of the final DataFrame:
- Index: Monthly timestamps.
- Columns (MultiIndex):
    Level 0: Currency (e.g., USD, EUR)
    Level 1: Index Symbol (e.g., GSPC, FTSE)
    Level 2: Metric ('Close' or 'Volume')

Functions:
- fetch_index_data(): Downloads and aligns index data per currency.
- combine_all_data_to_multiindex(): Structures the data into a 
  MultiIndex DataFrame.

Usage:
- Can be run as a standalone script with:
    python index_match.py YYYY-MM-DD YYYY-MM-DD
  (e.g., python index_match.py 2001-01-01 2025-01-01)
- Can also be imported as a module for use in broader
  financial data processing or strategy backtesting pipelines.

Dependencies:
- pandas
- yfinance
- sys
- typing
==========================================================
"""

import pandas as pd
import yfinance as yf
import sys
from typing import Dict, List, Tuple


def fetch_index_data(indexes: Dict[str, List[str]], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download and organize index data by currency.

    Args:
        indexes (dict): Mapping from currency to list of Yahoo Finance index symbols.
        start_date (str): Start date in format 'YYYY-MM-DD'.
        end_date (str): End date in format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: MultiIndex dataframe (Currency, Index, Metric) with monthly data.
    """
    all_data = {}
    timeline = pd.date_range(start=start_date, end=end_date, freq='ME')

    for currency, index_list in indexes.items():
        currency_df = pd.DataFrame()
        print(f"\n Processing currency: {currency}")

        for symbol in index_list:
            try:
                print(f"  ↳ Downloading: {symbol}")
                df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

                if df.empty:
                    print(f"No data for {symbol}")
                    continue

                # Keep only Close and Volume columns
                df = df[["Close", "Volume"]]

                # Rename columns to <Symbol>_<Metric>
                df.columns = [symbol.replace("^", "") + "_" + col for col in df.columns]

                # Resample to month-end frequency and reindex
                df = df.resample("ME").last()
                df = df.reindex(timeline, method="ffill")

                # Merge into the currency-level DataFrame
                currency_df = df if currency_df.empty else pd.concat([currency_df, df], axis=1)

            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                continue

        if not currency_df.empty:
            all_data[currency] = currency_df

    # Combine into final multi-index DataFrame
    combined_df = combine_all_data_to_multiindex(all_data)
    return combined_df


def combine_all_data_to_multiindex(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert all_data dict into a MultiIndex DataFrame.

    Columns are structured as:
        Currency → Index Symbol → Metric

    Args:
        all_data (dict): Dictionary of currency-wise DataFrames with flat column names.

    Returns:
        pd.DataFrame: MultiIndex column DataFrame.
    """
    all_dfs = []

    for currency, df in all_data.items():
        new_columns: List[Tuple[str, str, str]] = []

        for col in df.columns:
            if "_" in col:
                parts = col.split("_")
                symbol = parts[0]
                metric = "_".join(parts[1:])
                new_columns.append((currency, symbol, metric))
            else:
                new_columns.append((currency, col, ""))

        df.columns = pd.MultiIndex.from_tuples(new_columns, names=["Currency", "Index", "Metric"])
        all_dfs.append(df)

    # Concatenate along columns (same time index)
    combined_df = pd.concat(all_dfs, axis=1)
    return combined_df


if __name__ == "__main__":
    # Ensure the user provides start and end dates
    if len(sys.argv) != 3:
        print("Error: Please provide start and end dates as arguments.")
        print("Usage: python index_match.py YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]

    # Define index symbols per currency
    indexes_per_currency = {
        "USD": ['^VIX', 'DX-Y.NYB', '^DJI', '^GSPC', '^RUT', '^IXIC', '^XAX', '^BVSP',
                '^125904-USD-STRD', '^XDB', '^XDE', '^XDN', '^XDA', '^JN0U.JO', '^NYA'],
        "EUR": ['^STOXX50E', '^N100', '^FCHI', '^GDAXI', '^BFX'],
        "GBP": ['^FTSE', '^FTMC', '^FTSE100', '^BUK100P'],
        "CAD": ['^GSPTSE'],
        "BRL": ['^BVSP'],
        "INR": ['^BSESN'],
        "CNY": ['000001.SS'],
        "HKD": ['^HSI'],
        "JPY": ['^N225'],
        "AUD": ['^AXJO', '^AORD'],
        "KRW": ['^KS11'],
        "EGP": ['^EGX30'],
        "ILS": ['^TA125.TA'],
        "CLP": ['^IPSA'],
        "MXN": ['^MXX'],
        "TWD": ['^TWII'],
        "NZD": ['^NZ50'],
        "MYR": ['^KLSE'],
        "IDR": ['^JKSE'],
        "SGD": ['^STI'],
    }

    print(f"\nFetching index data from {start_date} to {end_date}...\n")
    index_data = fetch_index_data(indexes_per_currency, start_date, end_date)

    # Save to file
    output_file = f"data/index_data_{start_date}_to_{end_date}.csv"
    index_data.to_csv(output_file)
    print(f"\nSaved combined index data to '{output_file}'")
