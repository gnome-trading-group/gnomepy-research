"""Adverse selection analysis — post-fill mid movement at multiple horizons."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from gnomepy_research.reporting.backtest.metrics import _is_buy

if TYPE_CHECKING:
    from gnomepy_research.reporting.backtest.report import BacktestReport

DEFAULT_HORIZONS_MS = [100, 500, 1_000, 5_000, 10_000, 30_000, 60_000]


def compute_adverse_selection(
    fills: pd.DataFrame,
    market_df: pd.DataFrame,
    horizons_ms: list[int] | None = None,
) -> pd.DataFrame:
    """Compute post-fill mid movement at each horizon for every fill.

    Returns a DataFrame indexed by fill timestamp with columns:
    ``side``, ``fill_price``, ``mid_at_fill``, and one
    ``adverse_{horizon}`` column per horizon (in the fill's favorable
    direction: positive = good, negative = adversely selected).
    """
    if horizons_ms is None:
        horizons_ms = DEFAULT_HORIZONS_MS

    if fills.empty or market_df.empty or "mid_price" not in market_df.columns:
        return pd.DataFrame()

    # Build a clean mid series for lookups.
    mid_df = market_df[["mid_price"]].sort_index().copy()
    mid_df = mid_df.reset_index()
    mid_df.columns = ["timestamp", "mid_price"]
    mid_df["mid_price"] = mid_df["mid_price"].astype(float)

    # Fill timestamps.
    fill_ts = pd.DataFrame({"timestamp": fills.index}).reset_index(drop=True)

    # Mid at fill time.
    at_fill = pd.merge_asof(
        fill_ts.sort_values("timestamp"),
        mid_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    mid_at_fill = at_fill["mid_price"].values

    # Sign: +1 for buys, -1 for sells.
    sign = fills["side"].map(lambda s: 1.0 if _is_buy(s) else -1.0).values

    # Compute mid at each horizon.
    result = pd.DataFrame({
        "side": fills["side"].values,
        "fill_price": fills["fill_price"].values,
        "mid_at_fill": mid_at_fill,
    }, index=fills.index)

    for h_ms in horizons_ms:
        shifted_ts = pd.DataFrame({
            "timestamp": fills.index + pd.Timedelta(milliseconds=h_ms),
        }).reset_index(drop=True)

        at_horizon = pd.merge_asof(
            shifted_ts.sort_values("timestamp"),
            mid_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        mid_at_h = at_horizon["mid_price"].values
        mid_change_bps = (mid_at_h - mid_at_fill) / mid_at_fill * 10_000 * sign

        label = _horizon_label(h_ms)
        result[f"adverse_{label}"] = mid_change_bps

    return result


def _horizon_label(ms: int) -> str:
    if ms >= 1000:
        s = ms / 1000
        return f"{s:g}s"
    return f"{ms}ms"


def plot_adverse_selection(
    report: "BacktestReport",
    horizons_ms: list[int] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Average post-fill mid movement curve across horizons."""
    if horizons_ms is None:
        horizons_ms = DEFAULT_HORIZONS_MS

    df = compute_adverse_selection(report.fills, report._market_df, horizons_ms)

    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            title=title or "Adverse Selection (no fills)",
            height=400,
        )
        return fig

    adverse_cols = [c for c in df.columns if c.startswith("adverse_")]
    means = df[adverse_cols].mean()
    labels = [c.replace("adverse_", "") for c in adverse_cols]

    # Overall average.
    fig.add_trace(go.Scatter(
        x=labels, y=means.values, mode="lines+markers",
        name="all fills", line=dict(width=2),
    ))

    # Split by side.
    buy_mask = df["side"].map(_is_buy)
    if buy_mask.any():
        buy_means = df.loc[buy_mask, adverse_cols].mean()
        fig.add_trace(go.Scatter(
            x=labels, y=buy_means.values, mode="lines+markers",
            name="buys", line=dict(width=1.2, dash="dash", color="#2ca02c"),
        ))
    if (~buy_mask).any():
        sell_means = df.loc[~buy_mask, adverse_cols].mean()
        fig.add_trace(go.Scatter(
            x=labels, y=sell_means.values, mode="lines+markers",
            name="sells", line=dict(width=1.2, dash="dash", color="#d62728"),
        ))

    # Zero line.
    fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5)

    n_fills = len(df)
    pct_adverse_1s = 0.0
    if "adverse_1s" in df.columns and len(df) > 0:
        pct_adverse_1s = (df["adverse_1s"] < 0).mean() * 100

    fig.update_layout(
        title=title or f"Adverse Selection Profile ({n_fills} fills, {pct_adverse_1s:.0f}% adverse at 1s)",
        height=400,
        xaxis_title="horizon",
        yaxis_title="avg mid movement (bps, favorable direction)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def adverse_selection_section(report: "BacktestReport") -> go.Figure | None:
    """ReportSection render function."""
    if report.fills.empty:
        return None
    return plot_adverse_selection(report)
