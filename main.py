import pandas as pd
import numpy as np
import yfinance as yf
import re

import pandas as pd
import yfinance as yf
import re

def fetch_data(fx_pairs: list[str], start_date: str, end_date: str, freq: str) -> pd.DataFrame:
    """
    Fetches FX data from Yahoo Finance and returns a merged DataFrame
    with one column per currency pair.
    """
    # Input checks
    if not all(isinstance(pair, str) for pair in fx_pairs):
        raise ValueError("All FX pairs must be strings.")
    if not all(pair.endswith('=X') for pair in fx_pairs):
        raise ValueError("All FX pairs must end with '=X'.")
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise ValueError("start_date and end_date must be strings in 'YYYY-MM-DD' format.")
    
    print("Fetching data from Yahoo Finance...")
    
    all_data = pd.DataFrame()

    for pair in fx_pairs:
        df = yf.download(pair, start=start_date, end=end_date, interval=freq, auto_adjust=True)

        if df.empty:
            print(f"No data found for {pair}. Skipping.")
            continue

        # Extract currency label like "EURUSD" from "EURUSD=X"
        match = re.search(r'([A-Z]{3})([A-Z]{3})', pair)
        if not match:
            print(f"Could not parse currency pair from {pair}. Skipping.")
            continue
        label = match.group(0)

        # Extract the 'Close' column and rename it to the currency label
        df = df[['Close']].rename(columns={'Close': label})

        # Merge into the master dataframe
        if all_data.empty:
            all_data = df
        else:
            all_data = all_data.join(df, how='outer')

    # Restrict index to monthly frequency
    all_data = all_data.resample('M').last()  # monthly end

    return all_data


def process_data(data: dict) -> pd.DataFrame:
    """
    Processes the raw data from the API into a pandas DataFrame
    """
    
    df = pd.DataFrame(data["Time Series FX (Monthly)"]).T
    breakpoint()

    df.index = pd.to_datetime(df.index)
    df.columns = [col.split(" ")[1] for col in df.columns]
    df = df.astype(float)
    
    return df


if __name__ == "__main__":
    
    fx_list_df = pd.read_csv('data/physical_currency_list.csv')

    # DEFINE PARAMATERS OF THE REQUEST
    list_of_fx = fx_list_df['currency code'].tolist()
    to_fx = 'USD'

    FX_PAIRS = [f"{from_fx}{to_fx}=X" for from_fx in list_of_fx if from_fx != "USD" ]

    start_date = "2000-01-01"
    end_date = "2025-01-01"

    fx_df = fetch_data(fx_pairs = FX_PAIRS[:10], start_date=start_date, end_date=end_date, freq='1m')
    breakpoint()