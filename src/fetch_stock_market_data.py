#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stock Market Index Fetcher & Processor (CLI-ready, typed)

Enhancements:
- Read universe from a Python-like .env file that contains dict literals, e.g.
    valid_indices = { "USD": [...], "EUR": [...] }
    fundamentally_excluded = { "USD": [...], "GBP": [...] }
- Choose which dict to use via --env-key, and optionally subtract another dict via --exclude-key.
- Subset currencies via --currencies=USD,EUR,...

CLI:
  --start YYYY-MM-DD
  --end   YYYY-MM-DD
  --freq  {1d,1wk,1mo}
  --outdir PATH
  [--env-file PATH] [--env-key NAME] [--exclude-key NAME] [--currencies CSV]
"""

from __future__ import annotations

import argparse
import pathlib
import ast
import re
from typing import Dict, List, Tuple, Optional, Sequence

import pandas as pd
import yfinance as yf

YF_ALLOWED_INTERVALS: tuple[str, ...] = ("1d", "1wk", "1mo")


def sanitize_symbol(symbol: str) -> str:
    """Return a filesystem/column friendly symbol."""
    return symbol.replace("^", "").replace("-", "_").replace(".", "_").upper()


def resample_rule(interval: str) -> str:
    """Map Yahoo interval to a calendar alignment rule."""
    if interval == "1d":
        return "D"
    if interval == "1wk":
        return "W-FRI"
    if interval == "1mo":
        return "ME"
    raise ValueError(f"Unsupported interval: {interval}")


# ---------- New: load universe from a Python-like .env ----------
def _extract_assign_literal(source: str, varname: str) -> Dict[str, List[str]]:
    """
    Parse a Python-like file content and return the literal value of `varname`.
    Only supports simple literal assignments (dict/list/str/number).
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == varname:
                    return ast.literal_eval(node.value)
    raise KeyError(f"Variable '{varname}' not found in the provided file.")


def load_universe_from_env_file(env_path: pathlib.Path, varname: str) -> Dict[str, List[str]]:
    """Load a dict[str, list[str]] from a Python-like .env file (literal assignment)."""
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")
    text = env_path.read_text(encoding="utf-8")
    data = _extract_assign_literal(text, varname)
    if not isinstance(data, dict):
        raise TypeError(f"Variable '{varname}' must be a dict.")
    # Normalize currency keys uppercase, ensure lists of strings
    normalized: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"Value for '{k}' must be a list of strings.")
        normalized[k.upper()] = [str(x).strip() for x in v]
    return normalized


def apply_exclusions(
    base: Dict[str, List[str]],
    excl: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[str]]:
    """Return base minus any symbols listed in excl by matching currency keys."""
    if not excl:
        return base
    out: Dict[str, List[str]] = {}
    for cur, lst in base.items():
        to_drop = set(excl.get(cur, []))
        out[cur] = [s for s in lst if s not in to_drop]
    return out


def subset_currencies(base: Dict[str, List[str]], currencies: Optional[List[str]]) -> Dict[str, List[str]]:
    """Keep only selected currencies if provided."""
    if not currencies:
        return base
    keep = {c.upper().strip() for c in currencies}
    return {c: syms for c, syms in base.items() if c in keep}


# ---------- Existing fetch logic ----------
def fetch_index_data(
    indexes: Dict[str, List[str]],
    start_date: str,
    end_date: str,
    freq: str = "1mo",
) -> pd.DataFrame:
    """Download and organize index data by currency."""
    if freq not in YF_ALLOWED_INTERVALS:
        raise ValueError(f"freq must be one of {YF_ALLOWED_INTERVALS}")
    align = resample_rule(freq)
    timeline = pd.date_range(start=start_date, end=end_date, freq=align)
    all_data: Dict[str, pd.DataFrame] = {}

    for currency, index_list in indexes.items():
        if not index_list:
            continue
        print(f"[INFO] Processing currency: {currency}")
        currency_df = pd.DataFrame(index=timeline)

        for symbol in index_list:
            friendly = sanitize_symbol(symbol)
            try:
                print(f"  ↳ Downloading: {symbol} (interval={freq})")
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    interval=freq,
                    auto_adjust=True,
                    progress=False,
                )
                if df.empty:
                    print(f"  [WARN] No data for {symbol}")
                    continue

                keep_cols = [c for c in ("Close", "Volume") if c in df.columns]
                if not keep_cols:
                    print(f"  [WARN] Neither Close nor Volume for {symbol}")
                    continue

                sub = df[keep_cols].copy()
                sub.index = pd.to_datetime(sub.index)
                sub = sub[~sub.index.duplicated(keep="last")]
                sub = sub.resample(align).last()
                sub = sub.reindex(timeline, method="ffill")
                sub.columns = [f"{friendly}_{col}" for col in sub.columns]
                currency_df = currency_df.join(sub, how="left")
            except Exception as e:
                print(f"  [ERROR] Failed to fetch {symbol}: {e}")
                continue

        valid_cols = [c for c in currency_df.columns if c is not None]
        if valid_cols:
            all_data[currency] = currency_df

    if not all_data:
        raise RuntimeError("No data fetched for any currency.")

    combined_df = combine_all_data_to_multiindex(all_data)
    combined_df = combined_df.loc[
        (combined_df.index >= pd.to_datetime(start_date)) &
        (combined_df.index <= pd.to_datetime(end_date))
    ]
    return combined_df


def combine_all_data_to_multiindex(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert {currency: flat-cols DF} into a MultiIndex-column DataFrame."""
    all_dfs: List[pd.DataFrame] = []
    for currency, df in all_data.items():
        new_columns: List[Tuple[str, str, str]] = []
        for col in df.columns:
            if isinstance(col, str) and "_" in col:
                symbol, metric = col.split("_", 1)
                new_columns.append((currency, symbol, metric))
            else:
                new_columns.append((currency, str(col), ""))
        df2 = df.copy()
        df2.columns = pd.MultiIndex.from_tuples(
            new_columns, names=["Currency", "Index", "Metric"]
        )
        all_dfs.append(df2)
    combined_df = pd.concat(all_dfs, axis=1).sort_index(axis=1)
    return combined_df


# ---------- Defaults if no .env is provided ----------
def default_index_universe() -> Dict[str, List[str]]:
    return {
        "USD": ['^DJI', '^GSPC', '^RUT', '^IXIC', '^XAX', '^NYA', 'DX-Y.NYB', '^JN0U.JO'],
        "EUR": ['^STOXX50E', '^N100', '^FCHI', '^GDAXI', '^BFX'],
        "GBP": ['^FTSE', '^FTMC', '^FTSE100'],
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch global indices from Yahoo Finance and save a MultiIndex CSV."
    )
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--freq", type=str, choices=list(YF_ALLOWED_INTERVALS),
                        default="1mo", help="Yahoo interval: 1d, 1wk, 1mo (default 1mo)")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data"),
                        help="Output directory (default: ./data)")
    # New: universe selection
    parser.add_argument("--env-file", type=pathlib.Path, default=None,
                        help="Path to a Python-like .env containing dict literals (e.g., valid_indices = {...}).")
    parser.add_argument("--env-key", type=str, default="valid_indices",
                        help="Variable name in the .env to use as universe (default: valid_indices).")
    parser.add_argument("--exclude-key", type=str, default=None,
                        help="Variable name in the .env to subtract from universe (e.g., fundamentally_excluded).")
    parser.add_argument("--currencies", type=str, default=None,
                        help="Comma-separated list of currencies to keep (e.g., USD,EUR,GBP).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Load universe
    if args.env_file:
        base_uni = load_universe_from_env_file(args.env_file, args.env_key)
        excl_uni = load_universe_from_env_file(args.env_file, args.exclude_key) if args.exclude_key else None
        idx_map = apply_exclusions(base_uni, excl_uni)
    else:
        idx_map = default_index_universe()

    # Optional: subset currencies
    cur_list = [c.strip() for c in args.currencies.split(",")] if args.currencies else None
    idx_map = subset_currencies(idx_map, cur_list)

    print(f"[INFO] Fetching index data {args.start} → {args.end} (interval={args.freq})")
    df = fetch_index_data(
        indexes=idx_map,
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_file = args.outdir / f"index_data_{args.start}_to_{args.end}_{args.freq}.csv"
    df.to_csv(out_file, index=True)
    print(f"[INFO] Saved {len(df)} rows x {df.shape[1]} cols -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
