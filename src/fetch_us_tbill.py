"""
fetch_us_tbill.py

Fetches daily US Treasury Bill (T-Bill) rates from the US Treasury website and outputs
a time series of monthly (4-week coupon equivalent) rates, aligned to month-end.

- Input (via command line): start_date, end_date in YYYY-MM-DD format
- Output: CSV file saved in the `../data/` directory
- Designed for integration into larger projects via import or CLI

Usage (from CLI):
    python fetch_us_tbill.py 2003-01-01 2025-01-01

Author: Rui Azevedo
Project: Carry Strategy
"""

import pandas as pd
import numpy as np
import sys
import os
import re


def fetch_tbill_data(start_date: str, end_date: str, output_path: str = "../data") -> pd.DataFrame:
    """
    Downloads and processes US T-Bill data between the given dates.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'
        output_path (str): Directory to save the output CSV

    Returns:
        pd.DataFrame: Cleaned and monthly-resampled T-Bill time series
    """
    # Generate month-end timeline
    timeline = pd.date_range(start=start_date, end=end_date, freq='M')

    # Extract list of unique years from timeline
    years = sorted(set([re.search(r'\d{4}', str(date)).group(0) for date in timeline]))

    all_data = None

    # Loop through each year and fetch the data
    for count, year in enumerate(years):
        print(f"Progress: {count / len(years):.2%} for year {year}")

        url = (
            f"https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/daily-treasury-rates.csv/2003/all?"
            f"type=daily_treasury_bill_rates&field_tdr_date_value={year}&page&_format=csv"
        )

        # Read CSV for the year
        df = pd.read_csv(url, sep=",", decimal=".", parse_dates=['Date'], index_col='Date')

        # Keep only the 4-week coupon equivalent rate
        df = df[['4 WEEKS COUPON EQUIVALENT']]
        df.rename(columns={'4 WEEKS COUPON EQUIVALENT': 'Monthly t-bills'}, inplace=True)

        # Concatenate data
        all_data = df if all_data is None else pd.concat([all_data, df], axis=0)

    # Final formatting: resample to month-end and reindex
    if all_data is not None:
        all_data.index = pd.to_datetime(all_data.index)
        all_data = all_data.resample('M').last()
        all_data.sort_index(inplace=True)
        all_data = all_data.reindex(timeline, method='ffill')
    else:
        all_data = pd.DataFrame(index=timeline, columns=['Monthly t-bills'])
        all_data['Monthly t-bills'] = np.nan

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'us_tbill_data_{start_date}_{end_date}.csv')

    all_data.to_csv(output_file)
    print(f"US Treasury Bill data saved to {output_file}")

    return all_data


if __name__ == "__main__":
    # Called via command line
    if len(sys.argv) != 3:
        print("Usage: python fetch_us_tbill.py <start_date> <end_date>")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]

    fetch_tbill_data(start_date, end_date)
