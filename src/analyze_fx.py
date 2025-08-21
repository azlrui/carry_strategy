import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.Classes.StatisticalAnalysis import StatisticalAnalysis as SA

def clean_dataset(df:pd.DataFrame, filter_start: str, filter_end: str) -> pd.DataFrame :
    start, end = filter_start, filter_end

    df = df.loc[start:end]

    cols_with_nan = df.columns[df.isna().any()]
    df = df.drop(columns=cols_with_nan)
    
    return df

def main():
    # Upload FX_DATASET
    file = "data/fx_data_USD_2001-01-01_2025-01-01_1mo.csv"
    start, end = "2006-06-30", "2025-01-01"

    fx_data = pd.read_csv(file, index_col='Date', parse_dates=True)
    fx_data = clean_dataset(fx_data, start, end)

    fx = SA(fx_data, method = "net")
    ret = fx.compute_returns(method = "log")
    print(ret.head())

if __name__ == "__main__":
    raise SystemExit(main())