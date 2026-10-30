"""Evaluation, cost-aware walk-forward backtesting, and statistical validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from skfin.backtesting_with_cost import (
    BacktesterWithCost,
    MeanVarianceWithCost,
    average_holding_period,
    compute_batch_holdings_with_cost,
    compute_pnl_components,
)
from skfin.metrics import drawdown, sharpe_ratio


class WalkForwardMeanVarianceWithCost(MeanVarianceWithCost):
    """Cost-aware Mean-Variance optimizer compatible with multi-day test blocks and risk scaling."""

    @staticmethod
    def compute_batch_holdings(
        pred: np.ndarray,
        V: np.ndarray,
        A: np.ndarray | None,
        risk_target: float | None = None,
        past_h: np.ndarray | None = None,
        vol_liquidity_factor: np.ndarray | None = None,
        lambda_: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        if vol_liquidity_factor is not None and np.ndim(vol_liquidity_factor) > 1:
            vol_liquidity_factor = np.mean(vol_liquidity_factor, axis=0)
        if past_h is not None and np.ndim(past_h) > 1:
            past_h = past_h[-1:]

        h = compute_batch_holdings_with_cost(
            pred=pred,
            V=V,
            A=A,
            past_h=past_h,
            vol_liquidity_factor=vol_liquidity_factor,
            lambda_=lambda_,
            risk_target=None,
        )
        if risk_target is not None:
            port_vol = np.sqrt(np.sum((h @ V) * h, axis=1, keepdims=True))
            port_vol = np.where(port_vol > 0, port_vol, 1.0)
            h = risk_target * h / port_vol
        return h


class MultiAssetBacktesterWithCost(BacktesterWithCost):
    """Walk-forward cost-aware backtester supporting multi-feature X (where X.shape[1] != y.shape[1])."""

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        ret: pd.DataFrame,
    ):
        X_sub = X.loc[: self.end_date]
        y_sub = y.loc[X_sub.index]
        cv = TimeSeriesSplit(
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            n_splits=1 + len(X_sub.loc[self.start_date :]) // self.test_size,
        )

        _h = []
        past_h = self.h_init
        for train, test in cv.split(X_sub):
            m = clone(self.estimator)
            m.fit(X_sub.iloc[train], y_sub.iloc[train])
            if self.vol_liquidity_factor is None:
                vlf = None
            else:
                vlf = np.squeeze(
                    self.vol_liquidity_factor.reindex(X_sub.index).values[test]
                )
            current_h = m.predict(
                X_sub.iloc[test], past_h=past_h, vol_liquidity_factor=vlf
            )
            _h.append(current_h)
            past_h = current_h

        cols = y.columns if isinstance(y, pd.DataFrame) else X.columns
        idx = X_sub.index[np.concatenate([test for _, test in cv.split(X_sub)])]
        h_ = pd.DataFrame(np.concatenate(_h), index=idx, columns=cols)

        self.h_ = h_.loc[self.start_date : self.end_date]
        self.pnl_components_ = compute_pnl_components(
            self.h_, ret, vol_liquidity_factor=self.vol_liquidity_factor
        )
        if self.return_pnl_component:
            self.pnl_ = self.pnl_components_
        else:
            if isinstance(self.pnl_components_, dict):
                self.pnl_ = self.pnl_components_["net = gross - impact cost"]
            else:
                self.pnl_ = self.pnl_components_
        return self


def compute_vol_liquidity_factor(
    ret: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
    cost_scale: float = 2e-3,
) -> pd.DataFrame:
    """Compute point-in-time per-asset quadratic transaction cost factor without look-ahead bias."""
    rolling_vol = ret.rolling(window, min_periods=min_periods).std()
    expanding_vol = ret.expanding(min_periods=2).std()
    initial_vol = ret.iloc[:min_periods].std()
    clean_vol = rolling_vol.fillna(expanding_vol).fillna(initial_vol)
    equal_liquidity_weight = 1.0 / ret.shape[1]
    return (clean_vol / equal_liquidity_weight) * cost_scale


def compute_net_strategy_summary(
    holdings_dict: dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    vol_liquidity_factor: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Gross and Net performance metrics across strategies."""
    rows = []
    for name, h in holdings_dict.items():
        h_clean = h.dropna(how="all")
        comp = compute_pnl_components(
            h_clean, ret, vol_liquidity_factor=vol_liquidity_factor
        )
        gross_pnl = comp["gross"]
        net_pnl = comp["net = gross - impact cost"]
        cost_pnl = comp["impact cost"]

        gross_ret = gross_pnl.mean() * 260 * 100
        net_ret = net_pnl.mean() * 260 * 100
        cost_drag = cost_pnl.mean() * 260 * 100
        ann_vol = net_pnl.std() * np.sqrt(260) * 100
        gross_sr = sharpe_ratio(gross_pnl)
        net_sr = sharpe_ratio(net_pnl)
        max_dd = drawdown(net_pnl, return_in_risk_unit=False).min() * 100
        turnover = h_clean.diff().abs().sum(axis=1).mean() * 260
        ahp = average_holding_period(h_clean)

        rows.append(
            {
                "Strategy": name,
                "Gross Ret (%)": round(float(gross_ret), 2),
                "Net Ret (%)": round(float(net_ret), 2),
                "Cost Drag (%)": round(float(cost_drag), 2),
                "Ann. Vol (%)": round(float(ann_vol), 2),
                "Gross Sharpe": round(float(gross_sr), 2),
                "Net Sharpe": round(float(net_sr), 2),
                "Max DD (Net %)": round(float(max_dd), 2),
                "Ann. Turnover": round(float(turnover), 2),
                "Avg Hold (Days)": round(float(ahp), 1),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy")


def compute_lo_confidence_interval(
    pnl: pd.Series | np.ndarray,
    periods_per_year: int = 260,
    confidence: float = 0.95,
    max_lag: int | None = None,
) -> dict[str, float | int]:
    """Compute Lo (2002) HAC-adjusted standard error and confidence interval for annualized Sharpe ratio."""
    x = pd.Series(pnl).dropna().astype(float).to_numpy()
    n = len(x)
    if n < 3:
        raise ValueError("At least three non-missing PnL observations are required.")
    mu = x.mean()
    sigma = x.std(ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("PnL standard deviation must be positive.")

    # IID standard error: sqrt((1 + 0.5 * SR_daily^2) / n)
    sr_daily = mu / sigma
    se_iid_daily = np.sqrt((1.0 + 0.5 * (sr_daily**2)) / n)

    # First-order influence function for Sharpe = mean / std (HAC / Newey-West)
    centered = x - mu
    var = np.mean(centered**2)
    influence = centered / sigma - (mu / (2.0 * sigma**3)) * (centered**2 - var)
    influence = influence - influence.mean()

    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = min(max(int(max_lag), 0), n - 1)
    long_run_var = float(np.dot(influence, influence) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        autocov = float(np.dot(influence[lag:], influence[:-lag]) / n)
        long_run_var += 2.0 * weight * autocov
    se_hac_daily = np.sqrt(max(long_run_var, 0.0) / n)

    scale = np.sqrt(periods_per_year)
    annual_sr = float(sr_daily * scale)
    se_iid_annual = float(se_iid_daily * scale)
    se_hac_annual = float(se_hac_daily * scale)
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p_value_hac = float(2.0 * (1.0 - norm.cdf(abs(annual_sr) / max(se_hac_annual, 1e-12))))

    return {
        "n_obs": int(n),
        "annualized_sharpe": round(annual_sr, 3),
        "se_iid": round(se_iid_annual, 3),
        "se_hac": round(se_hac_annual, 3),
        "ci_95_lower": round(annual_sr - z * se_hac_annual, 3),
        "ci_95_upper": round(annual_sr + z * se_hac_annual, 3),
        "p_value_hac": round(p_value_hac, 4),
        "hac_max_lag": int(max_lag),
    }
