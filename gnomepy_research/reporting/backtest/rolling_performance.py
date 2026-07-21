"""Rolling performance analysis — alpha decay, regime classification, stability."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gnomepy.reporting.metrics import compute_sharpe

if TYPE_CHECKING:
    from gnomepy.reporting.report import BacktestReport


def compute_rolling_sharpe(
    pnl_curve: pd.Series,
    window: str = "1h",
    bar: str = "10s",
) -> pd.Series:
    """Rolling Sharpe ratio over a sliding window.

    Returns a Series of Sharpe values indexed by the window end timestamp.
    NaN where there are insufficient bars.
    """
    if pnl_curve.empty:
        return pd.Series(dtype=float)

    resampled = pnl_curve.resample(bar).last().dropna()
    bar_returns = resampled.diff().dropna()

    if bar_returns.empty:
        return pd.Series(dtype=float)

    window_td = pd.Timedelta(window)
    bar_td = pd.Timedelta(bar)
    n_window_bars = max(2, int(window_td / bar_td))

    rolling_mean = bar_returns.rolling(n_window_bars, min_periods=max(2, n_window_bars // 2)).mean()
    rolling_std = bar_returns.rolling(n_window_bars, min_periods=max(2, n_window_bars // 2)).std(ddof=1)

    sharpe = rolling_mean / rolling_std.replace(0, float("nan"))
    sharpe.index = bar_returns.index
    return sharpe


def compute_alpha_decay(
    pnl_curve: pd.Series,
    window: str = "1h",
    bar: str = "10s",
) -> pd.Series:
    """Sharpe ratio of each non-overlapping window, ordered chronologically.

    A declining trend indicates alpha decay — edge deteriorating over time.
    """
    if pnl_curve.empty:
        return pd.Series(dtype=float)

    resampled = pnl_curve.resample(bar).last().dropna()
    bar_returns = resampled.diff().dropna()

    if bar_returns.empty:
        return pd.Series(dtype=float)

    window_td = pd.Timedelta(window)
    bar_td = pd.Timedelta(bar)
    n_window_bars = max(2, int(window_td / bar_td))

    chunks = [
        bar_returns.iloc[i : i + n_window_bars]
        for i in range(0, len(bar_returns), n_window_bars)
        if len(bar_returns.iloc[i : i + n_window_bars]) >= 2
    ]
    if not chunks:
        return pd.Series(dtype=float)

    sharpes = []
    timestamps = []
    for chunk in chunks:
        std = float(chunk.std(ddof=1))
        sharpes.append(float(chunk.mean()) / std if std > 0 else 0.0)
        timestamps.append(chunk.index[-1])

    return pd.Series(sharpes, index=pd.DatetimeIndex(timestamps))


_REGIME_LABELS = ("low_vol_tight", "low_vol_wide", "high_vol_tight", "high_vol_wide")


def detect_regimes(
    market_df: pd.DataFrame,
    vol_window: str = "5min",
) -> pd.Series:
    """Classify each market tick into one of four regimes.

    Regimes: {low_vol_tight, low_vol_wide, high_vol_tight, high_vol_wide}
    based on realized volatility and bid-ask spread, each split at the median.
    """
    if market_df.empty:
        return pd.Series(dtype=str)

    mkt = market_df.sort_index().copy()

    if "mid_price" not in mkt.columns:
        if "bid_price_0" in mkt.columns and "ask_price_0" in mkt.columns:
            mkt["mid_price"] = (mkt["bid_price_0"].astype(float) + mkt["ask_price_0"].astype(float)) / 2.0
        else:
            return pd.Series("unknown", index=mkt.index, dtype=str)

    mid = mkt["mid_price"].astype(float)
    vol_window_td = pd.Timedelta(vol_window)
    bar_td = pd.Timedelta("1s")
    n_bars = max(2, int(vol_window_td / bar_td))

    returns = mid.pct_change().fillna(0.0)
    realized_vol = returns.rolling(n_bars, min_periods=2).std(ddof=1).fillna(0.0)

    if "bid_price_0" in mkt.columns and "ask_price_0" in mkt.columns:
        bid = mkt["bid_price_0"].astype(float)
        ask = mkt["ask_price_0"].astype(float)
        spread = (ask - bid) / mid.replace(0, float("nan")) * 10_000
    else:
        spread = pd.Series(0.0, index=mkt.index)

    spread = spread.fillna(0.0)

    vol_med = realized_vol.median()
    spread_med = spread.median()

    high_vol = realized_vol >= vol_med
    wide_spread = spread >= spread_med

    labels = pd.Series("low_vol_tight", index=mkt.index, dtype=str)
    labels[~high_vol & wide_spread] = "low_vol_wide"
    labels[high_vol & ~wide_spread] = "high_vol_tight"
    labels[high_vol & wide_spread] = "high_vol_wide"

    return labels


def pnl_by_regime(
    pnl_curve: pd.Series,
    regimes: pd.Series,
) -> dict[str, dict]:
    """Split PnL contribution by regime.

    Returns a dict mapping regime label to {'final_pnl': ..., 'sharpe': ..., 'pct_time': ...}.
    """
    if pnl_curve.empty or regimes.empty:
        return {}

    bar_returns = pnl_curve.resample("10s").last().dropna().diff().dropna()

    regime_resampled = regimes.resample("10s").last().ffill().reindex(bar_returns.index).ffill()

    results: dict[str, dict] = {}
    for label in _REGIME_LABELS:
        mask = regime_resampled == label
        if not mask.any():
            continue
        regime_returns = bar_returns[mask]
        pct_time = float(mask.mean())
        std = float(regime_returns.std(ddof=1))
        sharpe = float(regime_returns.mean()) / std if std > 0 else 0.0
        results[label] = {
            "final_pnl": float(regime_returns.sum()),
            "sharpe": sharpe,
            "pct_time": pct_time,
            "n_bars": int(mask.sum()),
        }

    return results


def plot_rolling_performance(
    report: "BacktestReport",
    window: str = "1h",
    bar: str = "10s",
    title: str | None = None,
) -> go.Figure:
    """Rolling Sharpe + alpha decay + regime breakdown in one figure."""
    pnl = report.pnl_curve
    mkt = report._market_df

    rolling = compute_rolling_sharpe(pnl, window=window, bar=bar)
    decay = compute_alpha_decay(pnl, window=window, bar=bar)
    regimes = detect_regimes(mkt)
    regime_stats = pnl_by_regime(pnl, regimes)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Rolling Sharpe", "Alpha Decay (per window)", "PnL by Regime"),
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.1,
    )

    if not rolling.empty:
        fig.add_trace(
            go.Scatter(
                x=rolling.index, y=rolling.values, mode="lines",
                name=f"rolling Sharpe ({window})",
                line=dict(width=1.2, color="#6366f1"),
            ),
            row=1, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5, row=1, col=1)

    if not decay.empty:
        colors = ["#2ca02c" if s > 0 else "#d62728" for s in decay.values]
        fig.add_trace(
            go.Bar(
                x=decay.index, y=decay.values, name="window Sharpe",
                marker_color=colors,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5, row=2, col=1)

    if regime_stats:
        labels = list(regime_stats.keys())
        pnls = [regime_stats[r]["final_pnl"] for r in labels]
        bar_colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
        pct_times = [f"{regime_stats[r]['pct_time']:.0%}" for r in labels]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=pnls,
                name="PnL by regime",
                marker_color=bar_colors,
                text=pct_times,
                textposition="outside",
            ),
            row=3, col=1,
        )

    fig.update_layout(
        height=700,
        title=title or "Rolling Performance Analysis",
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def rolling_performance_section(report: "BacktestReport") -> go.Figure | None:
    """ReportSection render function."""
    pnl = report.pnl_curve
    if pnl.empty or len(pnl) < 10:
        return None
    duration = (pnl.index.max() - pnl.index.min()).total_seconds()
    if duration < 600:
        return None
    return plot_rolling_performance(report)
