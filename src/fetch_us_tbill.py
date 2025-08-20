#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==========================================================
US T-Bill Rate Fetcher & Processor (CLI-ready, typed)
==========================================================

Downloads daily US T-Bill COUPON EQUIVALENT rates from the
US Treasury website for a chosen tenor (in weeks), then
resamples to month-end and forward-fills onto a monthly
timeline between --start and --end.

CLI:
  --start YYYY-MM-DD
  --end   YYYY-MM-DD
  --tenor {4,6,8,13,17,26,52}  (weeks)
  --outdir PATH

Notes:
- Only the *COUPON EQUIVALENT* column is retained.
- Output CSV contains one column:
    'TBill_<TENOR>W_CouponEq'
- Values are left as reported by Treasury (percent). If you
  need decimals (e.g., 0.0523 for 5.23%), divide by 100
  downstream.

Dependencies:
  - pandas
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional, Sequence, Dict, List

import pandas as pd


# Mapping from tenor (weeks) to expected US Treasury column names
TENOR_COL_MAP: Dict[int, str] = {
    4:  "4 WEEKS COUPON EQUIVALENT",
    6:  "6 WEEKS COUPON EQUIVALENT",
    8:  "8 WEEKS COUPON EQUIVALENT",
    13: "13 WEEKS COUPON EQUIVALENT",
    17: "17 WEEKS COUPON EQUIVALENT",
    26: "26 WEEKS COUPON EQUIVALENT",
    52: "52 WEEKS COUPON EQUIVALENT",
}


def build_year_range(start_date: str, end_date: str) -> List[int]:
    """Return a sorted list of years intersecting [start_date, end_date]."""
    s = pd.to_datetime(start_date).year
    e = pd.to_datetime(end_date).year
    return list(range(s, e + 1))


def month_end_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Return a month-end index spanning [start_date, end_date]."""
    return pd.date_range(start=start_date, end=end_date, freq="ME")


def treasury_csv_url_for_year(year: int) -> str:
    """
    Construct the official CSV URL for daily T-Bill rates for a given year.

    Important:
    - We query 'daily_treasury_bill_rates' and filter by year.
    - Treasury serves one CSV per request with the 'Date' column.
    """
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/{year}/all?"
        "type=daily_treasury_bill_rates&"
        f"field_tdr_date_value={year}&page&_format=csv"
    )


def fetch_tbill_coupon_eq(
    start_date: str,
    end_date: str,
    tenor_weeks: int,
) -> pd.DataFrame:
    """
    Fetch daily COUPON EQUIVALENT series for the selected tenor and
    return a monthly (month-end) DataFrame with a single column.

    Returns
    -------
    pd.DataFrame
        Index: month-end dates
        Column: 'TBill_<TENOR>W_CouponEq' (values as percent, not decimal)
    """
    if tenor_weeks not in TENOR_COL_MAP:
        raise ValueError(f"tenor must be one of {sorted(TENOR_COL_MAP)}")

    target_col = TENOR_COL_MAP[tenor_weeks]
    years = build_year_range(start_date, end_date)
    monthly_index = month_end_index(start_date, end_date)

    all_daily: list[pd.DataFrame] = []

    for i, y in enumerate(years, start=1):
        print(f"[INFO] Fetching year {y} ({i}/{len(years)}) ...")
        url = treasury_csv_url_for_year(y)

        # Read CSV directly; enforce Date index
        df = pd.read_csv(url, sep=",", decimal=".", parse_dates=["Date"])
        if df.empty:
            print(f"[WARN] Empty CSV for year {y}")
            continue

        # Keep only Date + target COUPON EQUIVALENT column if present
        if target_col not in df.columns:
            print(f"[WARN] Column '{target_col}' missing for year {y}. Available: {list(df.columns)}")
            continue

        df = df[["Date", target_col]].copy()
        df.rename(columns={target_col: f"TBill_{tenor_weeks}W_CouponEq"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Coerce to numeric (Treasury may have 'N/A' or blanks)
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")

        # Restrict to the global [start, end] range
        df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        if not df.empty:
            all_daily.append(df)

    if not all_daily:
        # Return an empty monthly frame with the expected column
        out = pd.DataFrame(index=monthly_index, columns=[f"TBill_{tenor_weeks}W_CouponEq"])
        return out.astype("float64")

    daily = pd.concat(all_daily).sort_index()
    # Resample to month-end last observation, then forward-fill onto the target monthly index
    monthly = daily.resample("ME").last()
    monthly = monthly.reindex(monthly_index, method="ffill")

    return monthly


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch US T-Bill COUPON EQUIVALENT for a given tenor (weeks) and save monthly CSV."
    )
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--tenor",
        type=int,
        choices=sorted(TENOR_COL_MAP),
        required=True,
        help="Tenor in weeks (one of 4, 6, 8, 13, 17, 26, 52).",
    )
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data"),
                        help="Output directory (default: ./data)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    print(f"[INFO] Fetching T-Bill {args.tenor}W coupon-equivalent {args.start} → {args.end}")

    monthly = fetch_tbill_coupon_eq(
        start_date=args.start,
        end_date=args.end,
        tenor_weeks=args.tenor,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"us_tbill_couponEq_{args.tenor}w_{args.start}_to_{args.end}.csv"
    monthly.to_csv(out_path, index=True)
    print(f"[INFO] Saved {len(monthly)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
