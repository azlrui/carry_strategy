import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Optional, Union, Dict, Literal

Number = Union[int, float]

# =========================
# Base class: Analysis
# =========================

@dataclass
class Analysis:
    """
    Container for time series (FX / indices).
    Expects:
      - df: DataFrame with DatetimeIndex
      - columns: series names (e.g., 'EURUSD', 'GBPUSD', ...)
    Provides:
      - basic info helpers and missing-value diagnostics.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        self.df = self._validate_input(data, pd.DataFrame)
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("Index must be of type pd.DatetimeIndex")
        self.df = self.df.sort_index()
        try:
            self._freq = pd.infer_freq(self.df.index)
        except Exception:
            self._freq = None  # may remain None (not critical)

    def _validate_input(self, obj, typ: type) -> object:
        """Return obj if type matches typ; raise otherwise."""
        if isinstance(obj, typ):
            return obj
        raise ValueError(f"Input must be of type {typ.__name__}.")

    # ---------- Basic info ----------
    def cols_list(self):
        """Return list of column names."""
        return list(self.df.columns)

    def index_range(self):
        """Return (start_date, end_date) or (None, None) if empty."""
        if self.df.empty:
            return (None, None)
        return (self.df.index[0], self.df.index[-1])

    def shape(self):
        """Return (T_length, N_assets)."""
        return self.df.shape

    def missing_report(self) -> pd.DataFrame:
        """NaN counts by column and share of missing."""
        n = len(self.df)
        out = pd.DataFrame({
            "n_missing": self.df.isna().sum(),
            "share_missing": self.df.isna().mean()
        })
        out.index.name = "series"
        out.attrs["n_obs"] = n
        return out.sort_values("n_missing", ascending=False)


# =========================
# Derived class: StatisticalAnalysis
# =========================

@dataclass
class StatisticalAnalysis(Analysis):
    """
    Keeps both raw prices and computed returns.
    Lets you run stats on either (on='prices' or on='returns').
    """
    def __init__(
        self,
        data: pd.DataFrame,
        method: Optional[Literal["log", "gross", "net"]] = None,
        periods: int = 1,
        clip_inf: bool = True,
    ) -> None:
        super().__init__(data)
        self.prices: pd.DataFrame = self.df.astype(float)

        # Return engine config (lazy by default)
        self._ret_method: Literal["log", "gross", "net"] = method or "net"
        self._ret_periods: int = periods
        self._ret_clip_inf: bool = clip_inf
        self._returns_cache: Optional[pd.DataFrame] = None

        # Precompute if user explicitly asked
        if method is not None:
            self._returns_cache = self.compute_returns(method, periods, clip_inf)

    # ---------- Internal helpers ----------

    def _ann_factor(self) -> float:
        """
        Infer an annualization factor from index frequency.
        Falls back to 12 (monthly) if unknown.
        """
        freq = self._freq
        if freq in ("B", "D"):
            return 252.0
        if freq and freq.startswith("W"):
            return 52.0
        if freq in ("M", "MS"):
            return 12.0
        if freq in ("Q", "QS"):
            return 4.0
        if freq in ("A", "AS", "Y"):
            return 1.0
        return 12.0  # safe fallback for monthly-type datasets

    def _select_domain(
        self,
        on: Literal["prices", "returns"]
    ) -> pd.DataFrame:
        """Choose the data domain and validate availability."""
        if on == "prices":
            df = self.prices
        else:
            df = self.returns  # property, lazy-computes if needed
        if df is None or df.empty:
            raise ValueError("Input data is empty or returns are not available.")
        return df

    def _validate_and_subset_columns(
        self,
        df: pd.DataFrame,
        cols_to_analyze: Optional[Iterable[str]],
        numeric_only: bool
    ) -> pd.DataFrame:
        """
        Keep numeric columns and optionally subset to requested list with validation.
        """
        if numeric_only:
            df = df.select_dtypes(include=[np.number])

        if cols_to_analyze is not None:
            cols_to_analyze = list(cols_to_analyze)
            missing = [c for c in cols_to_analyze if c not in df.columns]
            if missing:
                raise ValueError(f"Les colonnes suivantes n'existent pas: {missing}")
            df = df[cols_to_analyze]
        return df

    def _winsorize(
        self,
        df: pd.DataFrame,
        winsor: Optional[float]
    ) -> pd.DataFrame:
        """
        Column-wise clipping to limit the impact of extreme outliers.
        winsor: e.g. 0.01 trims 1% tails on both sides.
        """
        if winsor is not None and 0 < winsor < 0.5:
            lo = df.quantile(winsor)
            hi = df.quantile(1 - winsor)
            df = df.clip(lo, hi, axis=1)
        return df

    # ---------- Returns engine ----------

    def compute_returns(
        self,
        method: Literal["log", "gross", "net"] = "net",
        periods: int = 1,
        clip_inf: bool = True
    ) -> pd.DataFrame:
        """
        Convert price-level data to returns.
        - "log"   : log(P_t) - log(P_{t-k})
        - "gross" : P_t / P_{t-k}
        - "net"   : P_t / P_{t-k} - 1
        """
        prices = self.prices.astype(float)

        if method == "log":
            ret = np.log(prices).diff(periods=periods)
        elif method == "gross":
            ret = prices / prices.shift(periods)
        elif method == "net":
            ret = prices.pct_change(periods=periods)
        else:
            raise ValueError("method must be 'log', 'gross', or 'net'.")

        if clip_inf:
            ret = ret.replace([np.inf, -np.inf], np.nan)
        return ret

    @property
    def returns(self) -> pd.DataFrame:
        """
        Lazy access to returns based on the current config
        (method, periods, clip_inf). Computed once and cached.
        """
        if self._returns_cache is None:
            self._returns_cache = self.compute_returns(
                self._ret_method, self._ret_periods, self._ret_clip_inf
            )
        return self._returns_cache

    # ---------- Usual statistics ----------

    def basic_stat(
        self,
        statistic: Literal[
            "mean", "variance", "std", "skew", "kurtosis", "ann_std", "ann_variance"
        ] = "mean",
        cols_to_analyze: Optional[Iterable[str]] = None,
        ranked: bool = False,
        k: int = 5,
        ddof: int = 1,                 # sample stats by default
        numeric_only: bool = True,
        on: Literal["prices", "returns"] = "returns",
        winsor: Optional[float] = None,  # e.g. 0.01 trims 1% tails per column
        min_periods: int = 1,
        ascending: bool = False,       # sort direction for the metric
        top: Optional[int] = None,     # alternative to k; if set, overrides k
        bias: bool = True,             # for skew/kurtosis bias correction
    ) -> pd.DataFrame:
        """
        Compute a per-column statistic on either prices or returns.
        Optionally rank and slice.

        Notes
        -----
        - 'variance'/'std' use ddof; finance convention is ddof=1 (sample).
        - 'kurtosis' uses pandas (Fisher's excess) by default.
        - 'ann_std' and 'ann_variance' annualize ONLY when `on="returns"`.
        - Winsorization is column-wise clipping to limit tail impact.
        """
        # Choose domain and standardize the frame
        df = self._select_domain(on)
        df = self._validate_and_subset_columns(df, cols_to_analyze, numeric_only)
        df = self._winsorize(df, winsor)

        # Compute the chosen metric
        stat_name = statistic
        if statistic == "mean":
            s = df.mean(min_count=min_periods)
        elif statistic == "variance":
            s = df.var(ddof=ddof)
        elif statistic == "std":
            s = df.std(ddof=ddof)
        elif statistic == "skew":
            s = df.skew(bias=bias)          # ddof not relevant
        elif statistic == "kurtosis":
            s = df.kurtosis(bias=bias)      # Fisher's by default (excess)
        elif statistic == "ann_std":
            if on != "returns":
                raise ValueError("Annualized metrics only make sense on returns. Set on='returns'.")
            af = self._ann_factor()
            s = df.std(ddof=ddof) * np.sqrt(af)
            stat_name = "ann_std"
        elif statistic == "ann_variance":
            if on != "returns":
                raise ValueError("Annualized metrics only make sense on returns. Set on='returns'.")
            af = self._ann_factor()
            s = (df.std(ddof=ddof) ** 2) * af
            stat_name = "ann_variance"
        else:
            raise ValueError(f"Unknown statistic '{statistic}'.")

        out = s.to_frame(name=stat_name).sort_values(stat_name, ascending=ascending)

        # Optional ranking + slicing
        if ranked:
            # Highest value rank = 1 if ascending=False (usual for vol/mean)
            out["rank"] = out[stat_name].rank(method="first", ascending=ascending).astype(int)
            out = out.sort_values(["rank", stat_name])

        if top is not None and top > 0:
            out = out.head(top)
        elif ranked and k is not None and k > 0:
            out = out.head(k)

        # Metadata for traceability
        out.index.name = "series"
        out.attrs.update({
            "ddof": ddof,
            "domain": on,
            "winsor": winsor,
            "annualized": statistic in {"ann_std", "ann_variance"},
            "min_periods": min_periods
        })
        return out
    
    # ---------- Summary ----------

    def summary(
        self,
        on: Literal["prices", "returns"] = "returns",
        annualize: bool = True,
        ddof: int = 1,
        cols_to_analyze: Optional[Iterable[str]] = None,
        numeric_only: bool = True
    ) -> pd.DataFrame:
        """
        Produce a compact summary table with key statistics.

        Parameters
        ----------
        on : {"prices","returns"}, default "returns"
            Work on raw price levels or computed returns.
        annualize : bool, default True
            Whether to annualize mean/std/Sharpe (only if on="returns").
        ddof : int, default 1
            Delta degrees of freedom (sample stats).
        cols_to_analyze : iterable of str, optional
            Restrict computation to a subset of columns (validated).
        numeric_only : bool, default True
            Drop non-numeric columns before computation.

        Returns
        -------
        pd.DataFrame : one row per series with columns:
            ["mean","std","skew","kurtosis","sharpe","maxDD"]
        """
        # select data domain
        df = self._select_domain(on)

        # validate subset & numeric restriction
        df = self._validate_and_subset_columns(df, cols_to_analyze, numeric_only)

        if df.empty:
            raise ValueError("No valid columns to analyze.")

        # --- compute stats ---
        mean = df.mean()
        std = df.std(ddof=ddof)
        skew = df.skew()
        kurt = df.kurtosis()

        # Drawdowns
        if on == "returns":
            cum = (1 + df).cumprod()
            roll_max = cum.cummax()
            dd = cum / roll_max - 1.0
            maxDD = dd.min()
        else:
            maxDD = pd.Series(np.nan, index=df.columns)

        # Annualization factor
        af = self._ann_factor() if (annualize and on == "returns") else 1.0

        mean_ann = mean * af
        std_ann = std * np.sqrt(af)
        sharpe = mean_ann / std_ann.replace(0, np.nan)

        out = pd.DataFrame({
            "mean": mean_ann if on == "returns" else mean,
            "std": std_ann if on == "returns" else std,
            "skew": skew,
            "kurtosis": kurt,
            "sharpe": sharpe if on == "returns" else np.nan,
            "maxDD": maxDD
        })
        out.index.name = "series"

        # metadata
        out.attrs.update({
            "domain": on,
            "annualized": (annualize and on == "returns"),
            "ddof": ddof,
            "cols_restricted": cols_to_analyze,
            "numeric_only": numeric_only
        })

        return out
