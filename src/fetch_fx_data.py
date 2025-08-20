#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==========================================================
FX Data Fetcher & Processor (CLI-ready, typed)
==========================================================

This module fetches FX data from Yahoo Finance for a list of
currency pairs and outputs a time-aligned pandas DataFrame.

Key features:
- Strong typing and input validation
- CLI with argparse (Windows .bat friendly)
- Supports Yahoo intervals: 1d, 1wk, 1mo
- Resampling aligned to D / W-FRI / Month-End as needed
- Robust CSV reading for currency list
- Auto-create output directory and descriptive filename

Dependencies:
- pandas
- numpy
- yfinance
- python-dateutil (for robust date parsing if needed; here we stick to str)
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf


YF_ALLOWED_INTERVALS: tuple[str, ...] = ("1d", "1wk", "1mo")
PAIR_REGEX = re.compile(r"^([A-Z]{3})([A-Z]{3})=X$")


def validate_interval(freq: str) -> str:
    """Return a valid Yahoo interval or raise ValueError."""
    if freq not in YF_ALLOWED_INTERVALS:
        raise ValueError(
            f"Unsupported freq '{freq}'. Allowed: {', '.join(YF_ALLOWED_INTERVALS)}"
        )
    return freq


def read_currency_list(csv_path: pathlib.Path, colname: str = "currency code") -> List[str]:
    """Read a CSV column with 3-letter ISO currency codes."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if colname not in df.columns:
        raise KeyError(
            f"Column '{colname}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    series = df[colname].astype(str).str.upper().str.strip()
    # Keep canonical 3-letter codes only
    codes = [c for c in series if re.fullmatch(r"[A-Z]{3}", c)]
    if not codes:
        raise ValueError(f"No valid 3-letter currency codes found in column '{colname}'.")
    return codes


def build_pairs(from_currencies: Sequence[str], to_currency: str) -> List[str]:
    """Build Yahoo tickers like 'EURUSD=X', excluding identity pairs."""
    to_currency = to_currency.upper().strip()
    if not re.fullmatch(r"[A-Z]{3}", to_currency):
        raise ValueError("`to` must be a 3-letter ISO code (e.g., 'USD').")
    pairs = [f"{fx}{to_currency}=X" for fx in from_currencies if fx != to_currency]
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in pairs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    if not uniq:
        raise ValueError("No FX pairs built (check your currency list and `--to`).")
    return uniq


def resample_aligned(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Align index to standard closes consistent with the requested interval.
    - 1d  -> 'D' last
    - 1wk -> 'W-FRI' last (Yahoo weeks typically end Fri)
    - 1mo -> 'ME' last
    """
    # Ensure DateTimeIndex and sorted
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    if interval == "1d":
        return df.resample("D").last()
    if interval == "1wk":
        return df.resample("W-FRI").last()
    if interval == "1mo":
        return df.resample("ME").last()
    # Should have been validated earlier
    raise ValueError(f"Unexpected interval: {interval}")


def fetch_data(
    fx_pairs: Sequence[str],
    start_date: str,
    end_date: str,
    freq: str = "1mo",
) -> pd.DataFrame:
    """
    Fetch FX Close prices from Yahoo Finance and return a merged DataFrame.

    Parameters
    ----------
    fx_pairs : Sequence[str]
        Yahoo FX tickers (e.g., 'EURUSD=X').
    start_date : str
        Start date in 'YYYY-MM-DD'.
    end_date : str
        End date in 'YYYY-MM-DD'.
    freq : str
        Yahoo interval: '1d', '1wk', or '1mo'.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with one column per FX pair, aligned and resampled.
    """
    interval = validate_interval(freq)

    # Basic sanity checks on tickers
    bad = [p for p in fx_pairs if not PAIR_REGEX.match(p)]
    if bad:
        raise ValueError(f"Invalid Yahoo FX tickers: {bad}")

    print(f"[INFO] Downloading FX data ({len(fx_pairs)} pairs), interval={interval}...")
    all_data: Optional[pd.DataFrame] = None

    for pair in fx_pairs:
        print(f"[INFO] {pair} ...")
        df = yf.download(
            pair,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        # Flatten multi-index columns if any (OHLCV)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        if df.empty:
            print(f"[WARN] No data for {pair}. Skipping.")
            continue

        # Extract 'EURUSD' from 'EURUSD...'
        m = re.search(r"([A-Z]{3})([A-Z]{3})", pair)
        if not m:
            print(f"[WARN] Could not parse label from {pair}. Skipping.")
            continue
        label = m.group(0)

        # Keep 'Close' only
        if "Close" not in df.columns:
            print(f"[WARN] 'Close' not in columns for {pair}. Skipping.")
            continue

        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index)
        s.name = label
        s = s[~s.index.duplicated(keep="last")]  # defensive against duplicate stamps

        # Merge
        if all_data is None:
            all_data = s.to_frame()
        else:
            all_data = all_data.join(s, how="outer")

    if all_data is None or all_data.empty:
        raise RuntimeError("No data fetched. Check inputs and connection.")

    # Align to requested frequency (end-of-period)
    all_data = resample_aligned(all_data, interval)

    # Optional: strictly keep rows within [start, end) after resampling
    all_data = all_data.loc[(all_data.index >= pd.to_datetime(start_date)) &
                            (all_data.index <= pd.to_datetime(end_date))]
    return all_data


def process_data(data: dict, col: str) -> pd.DataFrame:
    """
    Process raw FX data provided as a nested dictionary (legacy support).

    Notes
    -----
    - This assumes 'data[col]' is a dict keyed by timestamps/strings.
    - Converts to float and cleans column labels of the form 'XYZ ...' -> 'XYZ'.
    """
    df = pd.DataFrame(data[col]).T
    df.index = pd.to_datetime(df.index)
    # Extract token after first space -> matches original behavior
    df.columns = [c.split(" ")[1] if " " in c else c for c in df.columns]
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch FX data from Yahoo and save as CSV."
    )
    parser.add_argument("--fx-file", type=pathlib.Path, required=True,
                        help="Path to CSV with currency codes (e.g., data/physical_currency_list.csv).")
    parser.add_argument("--colname", type=str, default="currency code",
                        help="Column name in CSV containing 3-letter ISO codes (default: 'currency code').")
    parser.add_argument("--to", type=str, required=True,
                        help="Destination/base currency (3-letter code), e.g., USD.")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD).")
    parser.add_argument("--freq", type=str, default="1mo", choices=list(YF_ALLOWED_INTERVALS),
                        help="Yahoo interval: 1d, 1wk, 1mo (default: 1mo).")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data"),
                        help="Directory to write the output CSV (default: ./data).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Build pairs from CSV list
    cur_list = read_currency_list(args.fx_file, args.colname)
    pairs = build_pairs(cur_list, args.to)
    print(f"[INFO] Built {len(pairs)} Yahoo tickers for base '{args.to}'.")

    # Fetch
    df = fetch_data(
        fx_pairs=pairs,
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
    )

    # Ensure output dir exists
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_name = f"fx_data_{args.to}_{args.start}_{args.end}_{args.freq}.csv"
    out_path = args.outdir / out_name
    df.to_csv(out_path, index=True)
    print(f"[INFO] Saved {len(df)} rows x {len(df.columns)} cols -> {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
