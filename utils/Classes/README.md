# API of Analysis

base = Analysis(df)
base.cols_list()     # -> list[str]
base.index_range()   # -> (start_ts, end_ts) ou (None, None)
base.shape()         # -> (T, N)
base.missing_report()# -> DataFrame: n_missing, share_missing (attrs["n_obs"])

# API of StatisticalAnalysis

## Construction
fx = StatisticalAnalysis(
    data=df,
    method="net",     # "net": P_t/P_{t-1}-1 | "gross": P_t/P_{t-1} | "log": ln(P_t)-ln(P_{t-1})
    periods=1,        # k dans P_t/P_{t-k}
    clip_inf=True     # replace ±inf by NaN
)

fx.prices    # DataFrame of prices (float)
fx.returns   # DataFrame of returns (computed lazily using config above)

## Price conversion -> returns
r = fx.compute_returns(method="net")     # simple returns
r_log = fx.compute_returns(method="log") # log returns
r_gross = fx.compute_returns("gross")    # gross level (P_t/P_{t-1})

## Basic Stats
out = fx.basic_stat(
    statistic="ann_std",   # "mean","variance","std","skew","kurtosis","ann_std","ann_variance"
    cols_to_analyze=None,  # iterable[str] -> restrict to a subset
    ranked=True, k=10,     # add rank col and take top-k
    ddof=1,                # sample stats
    numeric_only=True,
    on="returns",          # or "prices"
    winsor=None,           # e.g. 0.01 -> winsorize 1% tails
    min_periods=1,
    ascending=False,       # sorting order
    top=None,              # alternative to k
    bias=True              # for skew/kurtosis
)


## Summary
tab = fx.summary(
    on="returns",            # "prices" | "returns"
    annualize=True,          # annualize mean/std/Sharpe if on="returns"
    ddof=1,
    cols_to_analyze=None,    # subset (validated)
    numeric_only=True
)

### EXAMPLE
# --- 1. Setup
import pandas as pd
from src.StatisticalAnalysis import StatisticalAnalysis

df = pd.read_csv("data/fx_data_USD_2001-01-01_2025-01-01_1mo.csv",
                 index_col="Date", parse_dates=True)

fx = StatisticalAnalysis(df, method="net")

# --- 2. Quick diagnostics
display(pd.DataFrame({
    "n_cols": [len(fx.cols_list())],
    "start": [fx.index_range()[0]],
    "end": [fx.index_range()[1]],
    "shape_TxN": [fx.shape()]
}))
display(fx.missing_report().head())

# --- 3. Summary table (returns)
sum_tab = fx.summary()
display(sum_tab.sort_values("sharpe", ascending=False).head(10))

# --- 4. Basic stats - annualized volatility top-10
ann_vol_top = fx.basic_stat("ann_std", ranked=True, k=10)
display(ann_vol_top)

# --- 5. Subset summary (EUR, GBP, AUD)
subset = fx.summary(cols_to_analyze=["EURUSD","GBPUSD","AUDUSD"])
display(subset)

# --- 6. Price variance (sanity check on levels)
price_var = fx.basic_stat("variance", on="prices", ranked=True, k=5, ascending=True)
display(price_var)
