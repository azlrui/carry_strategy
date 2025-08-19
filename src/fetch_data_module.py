"""
==========================================================
FX Data Fetcher & Processor Module
==========================================================

This module fetches monthly foreign exchange (FX) data from 
Yahoo Finance for a list of currency pairs, and structures 
it into a time-aligned pandas DataFrame.

Functions:
- fetch_data(): Downloads and merges FX data across pairs.
- process_data(): (Legacy) Processes data from a dictionary 
  format (e.g., if pulled from another API like Alpha Vantage).

Usage:
- Can be used as a script or imported as a module in a larger 
  FX or carry trading project.

Dependencies:
- pandas
- numpy
- yfinance
- re
- sys
==========================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import re
import sys


def fetch_data(fx_pairs: list[str], start_date: str, end_date: str, freq: str = "1mo") -> pd.DataFrame:
    """
    Fetches FX data from Yahoo Finance and returns a merged DataFrame
    with one column per currency pair, aligned on monthly dates.

    Parameters:
    - fx_pairs (list of str): List of Yahoo FX tickers (e.g., "EURUSD=X")
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format
    - freq (str): Data frequency (default is '1mo')

    Returns:
    - pd.DataFrame: Time-indexed DataFrame with one column per FX pair
    """
    
    # Input checks
    if not all(isinstance(pair, str) for pair in fx_pairs):
        raise ValueError("All FX pairs must be strings.")
    if not all(pair.endswith('=X') for pair in fx_pairs):
        raise ValueError("All FX pairs must end with '=X'.")
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise ValueError("start_date and end_date must be strings in 'YYYY-MM-DD' format.")
    
    print("Fetching data from Yahoo Finance...")

    all_data = None  # Will contain all merged FX data

    for pair in fx_pairs:
        print(f"Processing {pair}...")

        # Download from Yahoo Finance
        df = yf.download(
            pair,
            start=start_date,
            end=end_date,
            interval=freq,
            auto_adjust=True,
            progress=False
        )

        # If Yahoo returns a multi-level column index (can happen with OHLCV), flatten it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        if df.empty:
            print(f"No data found for {pair}. Skipping.")
            continue

        # Extract currency code like "EURUSD" from "EURUSD=X"
        match = re.search(r'([A-Z]{3})([A-Z]{3})', pair)
        if not match:
            print(f"Could not parse currency pair from {pair}. Skipping.")
            continue
        label = match.group(0)

        # Keep only the 'Close' column and rename it
        df = df[['Close']].rename(columns={'Close': label})
        df.index = pd.to_datetime(df.index)  # Ensure datetime index

        # Append to main DataFrame
        if all_data is None:
            all_data = df
        else:
            all_data = pd.concat([all_data, df], axis=1)  # align on index

    # Final formatting
    if all_data is not None:
        all_data.sort_index(inplace=True)
        all_data = all_data.resample('M').last()  # align to month-end
    
    return all_data


def process_data(data: dict) -> pd.DataFrame:
    """
    Processes raw FX data provided in dictionary format 
    (e.g., from Alpha Vantage or other JSON APIs).

    Parameters:
    - data (dict): Dictionary with FX data

    Returns:
    - pd.DataFrame: Processed time-indexed DataFrame
    """
    # Convert nested dict to DataFrame and transpose to get time on index
    df = pd.DataFrame(data["Time Series FX (Monthly)"]).T
    breakpoint()  # Debugging hook if needed

    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df.columns = [col.split(" ")[1] for col in df.columns]  # Clean column names
    df = df.astype(float)  # Convert strings to numeric values
    
    return df


if __name__ == "__main__":
    # Usage: python fetch_data_module.py USD 2003-01-01 2025-01-01
    to_fx = sys.argv[1]           # e.g., 'USD'
    start_date = sys.argv[2]      # e.g., '2003-01-01'
    end_date = sys.argv[3]        # e.g., '2025-01-01'

    fx_list_df = pd.read_csv('data/physical_currency_list.csv')
    list_of_fx = fx_list_df['currency code'].tolist()
    FX_PAIRS = [f"{from_fx}{to_fx}=X" for from_fx in list_of_fx if from_fx != to_fx]

    fx_df = fetch_data(fx_pairs=FX_PAIRS, start_date=start_date, end_date=end_date, freq='1mo')
    fx_df.to_csv(f'data/fx_data_{to_fx}_{start_date}_{end_date}.csv')

    print(f"FX data saved to data/fx_data_{to_fx}_{start_date}_{end_date}.csv")
    print("Data fetching complete.")
    sys.exit(0)