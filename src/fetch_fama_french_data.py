#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==========================================================
Fama-French 3 Factors (Mkt-RF, SMB, HML) + RF (CLI-ready)
==========================================================

Downloads the official zip from Ken French's Data Library,
extracts the CSV member, parses dates based on frequency
(month/week/day), filters to [--start, --end], and writes a CSV.

CLI:
  --start  YYYY-MM-DD
  --end    YYYY-MM-DD
  --freq   {month,week,day,1mo,1w,1d}
  --outdir PATH
  [--as-decimal]  # optional, convert % to decimals

Output:
  <outdir>/ff_factors_<freq>_<start>_to_<end>.csv

Notes:
- The raw files are in percent units. Use --as-decimal to divide by 100.
- The CSV structure sometimes includes footers; we robustly drop non-numeric rows.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import zipfile
from typing import Optional, Sequence, Tuple

import pandas as pd
import requests


URLS = {
    "month": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
    "week":  "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip",
    "day":   "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
}
# Accept aliases to stay consistent with your other scripts
ALIASES = {"1mo": "month", "1w": "week", "1d": "day"}

EXPECTED_COLS = ("Mkt-RF", "SMB", "HML", "RF")


def normalize_freq(freq: str) -> str:
    """Map aliases to canonical values and validate."""
    f = freq.lower()
    f = ALIASES.get(f, f)
    if f not in URLS:
        raise ValueError("Unsupported frequency. Use one of: month, week, day (or 1mo, 1w, 1d).")
    return f


def fetch_zip(url: str) -> bytes:
    """Download a zip file and return raw bytes."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def read_first_csv_from_zip(zbytes: bytes) -> pd.DataFrame:
    """
    Read the first CSV member from a zip archive as a DataFrame.
    We skip the header notes (usually 3-4 lines) by detecting the header row.
    """
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        # Choose first CSV entry
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not csv_name:
            raise RuntimeError("No CSV found inside the zip archive.")

        with zf.open(csv_name) as f:
            # Read raw text
            raw = f.read().decode("latin1")  # Ken French files are often Latin-1
    # Find the header line (where it contains 'Mkt-RF')
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:20]):  # header usually appears early
        if "Mkt-RF" in line and "SMB" in line and "HML" in line:
            header_idx = i
            break
    if header_idx is None:
        # Fallback: assume 3 or 4 line preamble; try 3
        header_idx = 3

    # Re-read via pandas from the header line
    buf = io.StringIO("\n".join(lines[header_idx:]))
    df = pd.read_csv(buf)
    df = df[:-1]
    # Standardize the first column name to 'Date'
    if df.columns[0] != "Date":
        df = df.rename(columns={df.columns[0]: "Date"})
    return df


def clean_ff_dataframe(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Keep only expected columns, coerce numeric, drop footer rows,
    and parse the Date according to frequency.
    """
    # Keep only expected columns when present
    keep = ["Date"] + [c for c in EXPECTED_COLS if c in df.columns]
    df = df[keep].copy()

    # Drop rows where all factor columns are NaN or non-numeric footers
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=keep[1:], how="all")

    # Parse Date depending on frequency (Ken French format)
    if freq == "month":
        # Monthly dates are typically YYYYMM (e.g., 200201)
        df = df[:-102] # This is due to the fact that FF includes annualization of factors in the last 102 lines
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
        # Normalize to month-end
        df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    elif freq in ("week", "day"):
        # Weekly/Daily use YYYYMMDD
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    else:
        raise ValueError(f"Unexpected freq: {freq}")

    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df


def clip_and_convert_units(df: pd.DataFrame, start: str, end: str, as_decimal: bool) -> pd.DataFrame:
    """Clip to [start, end] and optionally convert percent to decimal."""
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    out = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if as_decimal:
        out[EXPECTED_COLS] = out[EXPECTED_COLS].astype(float) / 100.0
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Fama-French 3 factors (+RF) from Ken French (zip), filter by date, and save CSV."
    )
    parser.add_argument("--start", required=True, type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--freq", required=True, type=str, help="month, week, day (or 1mo, 1w, 1d)")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data"), help="Output directory")
    parser.add_argument("--as-decimal", action="store_true", help="Convert % values to decimals")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    freq = normalize_freq(args.freq)
    url = URLS[freq]

    print(f"[INFO] Downloading Fama-French ({freq}) from {url}")
    zbytes = fetch_zip(url)

    print("[INFO] Reading CSV from zip...")
    raw_df = read_first_csv_from_zip(zbytes)

    print("[INFO] Cleaning and parsing dates...")
    clean_df = clean_ff_dataframe(raw_df, freq=freq)

    print(f"[INFO] Clipping to range {args.start} → {args.end} and unit conversion...")
    final_df = clip_and_convert_units(clean_df, args.start, args.end, args.as_decimal)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"ff_factors_{freq}_{args.start}_to_{args.end}.csv"
    final_df.to_csv(out_path, index=True)
    print(f"[INFO] Saved {len(final_df)} rows x {final_df.shape[1]} cols -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
