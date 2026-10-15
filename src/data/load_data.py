"""Data acquisition, cleaning, and feature engineering for commodity ETFs."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from skfin.dataloaders.cache import CacheManager
from skfin.datasets import load_yf_returns

# Major commodity ETFs covering broad commodities, energy, precious metals, and agriculture
DEFAULT_COMMODITY_TICKERS: List[str] = ["DBC", "USO", "GLD", "SLV", "DBA"]


def load_commodity_data(
    tickers: Optional[List[str]] = None,
    start: str = "2007-01-01",
    end: Optional[str] = None,
    cache_dir: str = "data",
    force_reload: bool = False,
    resample_bday: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load and clean commodity ETF return series.

    Ensures synchronous history across all ETFs by dropping time periods prior to
    the newest ETF's inception date (DBA in Jan 2007). Reindexes to business days (B)
    to guarantee consistent chronological frequency and prevent look-ahead bias.

    Args:
        tickers: List of ETF tickers (default: DBC, USO, GLD, SLV, DBA).
        start: Start date string (YYYY-MM-DD).
        end: Optional end date string (YYYY-MM-DD).
        cache_dir: Local cache directory for fast, offline reproducibility.
        force_reload: If True, bypass cache and re-download via yfinance.
        resample_bday: If True, align data to business day ('B') frequency.

    Returns:
        Dictionary containing:
            - 'total_returns': Daily total returns (price + dividend).
            - 'price_returns': Daily price returns.
            - 'dividend_returns': Daily dividend returns.
    """
    if tickers is None:
        tickers = DEFAULT_COMMODITY_TICKERS

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        if (Path("..") / cache_dir).exists():
            cache_dir = str(Path("..") / cache_dir)
        elif Path("data").exists():
            cache_dir = "data"

    raw_data = load_yf_returns(
        tickers=tickers,
        start=start,
        end=end,
        force_reload=force_reload,
        cache_dir=cache_dir,
    )

    cleaned_data: Dict[str, pd.DataFrame] = {}
    for key in ["total_returns", "price_returns", "dividend_returns"]:
        df = raw_data[key].copy()

        # Reorder columns to match requested tickers
        available_cols = [c for c in tickers if c in df.columns]
        df = df[available_cols]

        # Synchronize inception dates: drop dates where any ETF hasn't started yet
        df = df.dropna(how="any")

        if resample_bday:
            # Reindex to full business days index and fill missing days with 0 return
            # (e.g. market holidays where position value didn't change)
            bday_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq="B")
            df = df.reindex(bday_idx).fillna(0.0)

        cleaned_data[key] = df

    return cleaned_data


def compute_descriptive_stats(
    returns: pd.DataFrame,
    num_period_per_year: int = 260,
) -> pd.DataFrame:
    """Compute summary statistics for commodity ETF daily returns.

    Args:
        returns: DataFrame of daily returns.
        num_period_per_year: Annualization factor (260 for business days).

    Returns:
        DataFrame containing Annualized Mean (%), Annualized Vol (%),
        Skewness, Kurtosis, and Min/Max daily returns (%).
    """
    stats = pd.DataFrame(index=returns.columns)
    stats["Mean (Ann. %)"] = returns.mean() * num_period_per_year * 100
    stats["Std (Ann. %)"] = returns.std() * np.sqrt(num_period_per_year) * 100
    stats["Sharpe (RF=0)"] = stats["Mean (Ann. %)"] / stats["Std (Ann. %)"]
    stats["Skewness"] = returns.skew()
    stats["Kurtosis"] = returns.kurtosis()
    stats["Min (%)"] = returns.min() * 100
    stats["Max (%)"] = returns.max() * 100
    return stats.round(2)


def compute_momentum_features(
    returns: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute multi-horizon rolling momentum features strictly point-in-time.

    For each asset and window, momentum at date t uses returns up to date t.
    When predicting target returns at t+1, this preserves zero look-ahead bias.

    Args:
        returns: Daily returns DataFrame.
        windows: List of rolling lookback windows in business days
                 (default: 5d, 21d, 63d, 126d, 252d).

    Returns:
        DataFrame with MultiIndex columns (window, ticker) or concatenated columns.
    """
    if windows is None:
        windows = [5, 21, 63, 126, 252]

    feature_dfs = []
    for w in windows:
        # Rolling mean return over window w
        feat_w = returns.rolling(window=w).mean()
        feat_w.columns = [f"{col}_mom_{w}d" for col in returns.columns]
        feature_dfs.append(feat_w)

    features = pd.concat(feature_dfs, axis=1)
    return features


def prepare_backtest_data(
    returns: pd.DataFrame,
    feature_type: str = "returns",
    momentum_windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare aligned (features, target, returns) for skfin Backtester.

    Following course convention (see 11_Backtesting.ipynb & 15_Linear_models.ipynb):
    - features at date t represent information available at t.
    - target at date t is returns at date t+1 (ret.shift(-1)).
    - returns at date t+1 are multiplied by holdings h_t (pnl = h.shift(1) * ret).

    Args:
        returns: Cleaned daily returns DataFrame.
        feature_type: 'returns' for raw returns, or 'momentum' for multi-horizon momentum.
        momentum_windows: Lookback windows if feature_type == 'momentum'.

    Returns:
        Tuple of (X, y, ret) strictly aligned on common date index.
    """
    if feature_type == "returns":
        features = returns.copy()
    elif feature_type == "momentum":
        features = compute_momentum_features(returns, windows=momentum_windows)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # Target is next-period return: r_{t+1}
    target = returns.shift(-1)

    # Align dates and drop NaN targets at the end
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    X = features.loc[valid_idx]
    y = target.loc[valid_idx]
    ret = returns.loc[valid_idx]

    return X, y, ret
