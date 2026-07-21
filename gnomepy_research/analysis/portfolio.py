"""Cross-strategy portfolio analysis — correlation and combined Sharpe."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _load_pnl_curve(results_dir: str | Path) -> pd.Series:
    """Load PnL curve from a results directory (parquet + market.parquet)."""
    results_dir = Path(results_dir)
    market_path = results_dir / "market.parquet"
    fills_path = results_dir / "fills.parquet"

    if not market_path.exists():
        return pd.Series(dtype=float)

    from gnomepy.reporting.metrics import build_curves

    market_df = pd.read_parquet(market_path)
    fills_df = pd.read_parquet(fills_path) if fills_path.exists() else pd.DataFrame()
    curves = build_curves(market_df, fills_df)
    return curves.pnl


def compute_cross_strategy_correlation(
    results_dirs: list[str | Path],
    session_names: list[str] | None = None,
    resample_bar: str = "1min",
) -> pd.DataFrame:
    """Compute pairwise PnL correlation across strategy sessions.

    results_dirs: list of paths to completed backtest result directories
    session_names: optional labels (defaults to directory names)
    Returns a square correlation DataFrame.
    """
    if session_names is None:
        session_names = [Path(d).parent.name or Path(d).name for d in results_dirs]

    pnl_series: dict[str, pd.Series] = {}
    for name, d in zip(session_names, results_dirs):
        pnl = _load_pnl_curve(d)
        if not pnl.empty:
            pnl_series[name] = pnl.resample(resample_bar).last().dropna()

    if not pnl_series:
        return pd.DataFrame()

    all_ts = sorted(set().union(*[s.index for s in pnl_series.values()]))
    aligned = pd.DataFrame(index=pd.DatetimeIndex(all_ts))
    for name, series in pnl_series.items():
        aligned[name] = series.reindex(aligned.index).ffill()

    bar_returns = aligned.diff().dropna()
    return bar_returns.corr()


def combined_sharpe(
    pnl_curves: list[pd.Series],
    weights: list[float] | None = None,
    bar: str = "1min",
) -> float:
    """Portfolio-level Sharpe under specified capital weights.

    Weights default to equal allocation. Each PnL curve is normalized
    to unit final PnL before weighting, then combined.
    """
    from gnomepy.reporting.metrics import compute_sharpe

    if not pnl_curves:
        return 0.0

    if weights is None:
        weights = [1.0 / len(pnl_curves)] * len(pnl_curves)

    if len(weights) != len(pnl_curves):
        raise ValueError("weights length must match pnl_curves length")

    bar_returns_list = []
    for pnl, w in zip(pnl_curves, weights):
        if pnl.empty:
            continue
        resampled = pnl.resample(bar).last().dropna()
        ret = resampled.diff().dropna()
        final = float(pnl.iloc[-1])
        if final != 0:
            ret = ret / abs(final) * w
        bar_returns_list.append(ret)

    if not bar_returns_list:
        return 0.0

    all_ts = sorted(set().union(*[r.index for r in bar_returns_list]))
    combined = pd.Series(0.0, index=pd.DatetimeIndex(all_ts))
    for ret in bar_returns_list:
        combined = combined.add(ret.reindex(combined.index).fillna(0.0))

    std = float(combined.std(ddof=1))
    return float(combined.mean()) / std if std > 0 else 0.0


def plot_correlation_matrix(
    corr: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Heatmap of the cross-strategy correlation matrix."""
    if corr.empty:
        return go.Figure()

    labels = list(corr.columns)
    z = corr.values

    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="correlation"),
        )
    )
    fig.update_layout(
        title=title or "Cross-Strategy PnL Correlation",
        height=max(300, 80 * len(labels)),
        width=max(400, 100 * len(labels)),
    )
    return fig
