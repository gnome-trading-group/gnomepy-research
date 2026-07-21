from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.datastore import DataStore
from gnomepy.java.enums import SchemaType
from gnomepy.java.market_data import MarketDataClient
from gnomepy.java.recorder import BacktestResults
from gnomepy.java.statics import Scales
from gnomepy.registry.api import RegistryClient
from gnomepy.reporting.report import BacktestReport
from gnomepy_research.signals.fair_value.base import FairValueSignal


def _parse_dt(dt: str | datetime) -> datetime:
    if isinstance(dt, datetime):
        return dt
    return datetime.fromisoformat(dt)


def _resolve_listing(listing_id: int) -> tuple[int, int]:
    registry = RegistryClient()
    listings = registry.get_listing(listing_id=listing_id)
    if not listings:
        raise ValueError(f"Listing {listing_id} not found in registry")
    listing = listings[0]
    return listing.security_id, listing.exchange_id


def _downsample(series: pd.Series, max_points: int) -> pd.Series:
    if len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series.iloc[::step]


def load_datastore(
    listing_id: int,
    start: str | datetime,
    end: str | datetime,
    schema_type: str = "MBP_10",
) -> DataStore:
    """Load market data as a DataStore for signal computation.

    Args:
        listing_id: Registry listing ID.
        start: Start datetime (ISO string or datetime).
        end: End datetime (ISO string or datetime).
        schema_type: SBE schema type name (default "MBP_10").

    Returns:
        DataStore ready for iteration via compute_signals().
    """
    ensure_jvm_started()
    security_id, exchange_id = _resolve_listing(listing_id)
    schema_enum = SchemaType[schema_type]
    client = MarketDataClient()
    return client.load(
        security_id=security_id,
        exchange_id=exchange_id,
        schema_type=schema_enum,
        start=_parse_dt(start),
        end=_parse_dt(end),
    )


def load_market_data(
    listing_id: int,
    start: str | datetime,
    end: str | datetime,
    schema_type: str = "MBP_10",
) -> pd.DataFrame:
    """Load market data as a DataFrame with derived columns.

    Adds ``mid_price``, ``spread``, ``spread_bps``, and ``microprice``
    derived from top-of-book bid/ask. Index is the event timestamp.

    Args:
        listing_id: Registry listing ID.
        start: Start datetime (ISO string or datetime).
        end: End datetime (ISO string or datetime).
        schema_type: SBE schema type name (default "MBP_10").

    Returns:
        DataFrame indexed by timestamp with market data columns.
    """
    ds = load_datastore(listing_id, start, end, schema_type)
    df = ds.to_df()
    if df.empty:
        return df

    if "timestamp_event" in df.columns:
        df = df.set_index("timestamp_event")

    if "bid_price_0" in df.columns and "ask_price_0" in df.columns:
        bid = df["bid_price_0"].astype(float)
        ask = df["ask_price_0"].astype(float)
        mid = (bid + ask) / 2.0
        df["mid_price"] = mid
        df["spread"] = ask - bid
        valid_mid = mid > 0
        df["spread_bps"] = np.where(valid_mid, df["spread"] / mid * 10_000, np.nan)

        if "bid_size_0" in df.columns and "ask_size_0" in df.columns:
            bid_sz = df["bid_size_0"].astype(float)
            ask_sz = df["ask_size_0"].astype(float)
            total_sz = bid_sz + ask_sz
            df["microprice"] = np.where(
                total_sz > 0,
                (bid * ask_sz + ask * bid_sz) / total_sz,
                mid,
            )

    return df


def compute_signals(
    source: DataStore,
    signals: dict,
) -> pd.DataFrame:
    """Feed market data through signals and return a DataFrame of values.

    FairValueSignal outputs are automatically scaled to human-readable prices
    (divided by Scales.PRICE). All other signal types return values as-is.

    Args:
        source: DataStore of market data (from load_datastore()).
        signals: Dict mapping column name to Signal instance.

    Returns:
        DataFrame indexed by event timestamp, one column per signal.
        Rows where a signal is not yet ready contain NaN.
    """
    signal_list = list(signals.items())
    is_fv = {name: isinstance(sig, FairValueSignal) for name, sig in signal_list}
    price_scale = Scales.PRICE

    rows = []
    for schema in source:
        ts = schema.timestamp_event
        for _, sig in signal_list:
            sig.update(ts, schema)

        row: dict = {"timestamp": pd.Timestamp(ts, unit="ns", tz="UTC")}
        for name, sig in signal_list:
            if sig.is_ready():
                val = sig.value()
                row[name] = val / price_scale if is_fv[name] else val
            else:
                row[name] = np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("timestamp")


def plot_signals(
    market_df: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
    signals_to_plot: list[str] | None = None,
    title: str | None = None,
    max_points: int = 10_000,
) -> go.Figure:
    """Plot mid price and signal values with a shared time axis.

    Args:
        market_df: DataFrame from load_market_data() (must have mid_price column).
        signals_df: DataFrame from compute_signals(). If None, plots price only.
        signals_to_plot: Subset of signals_df columns to show. Default: all columns.
        title: Figure title.
        max_points: Max data points per trace (uniform downsampling).

    Returns:
        Plotly Figure: price panel on top, signals panel below.
    """
    cols: list[str] = []
    if signals_df is not None and not signals_df.empty:
        cols = signals_to_plot or list(signals_df.columns)
        cols = [c for c in cols if c in signals_df.columns]

    n_rows = 2 if cols else 1
    row_heights = [0.45, 0.55] if cols else [1.0]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=(["Price", "Signals"] if cols else ["Price"]),
    )

    if "mid_price" in market_df.columns:
        mid = _downsample(market_df["mid_price"].dropna(), max_points)
        fig.add_trace(
            go.Scatter(x=mid.index, y=mid.values, name="mid price",
                       line=dict(width=0.8, color="#1f77b4")),
            row=1, col=1,
        )

    if cols and signals_df is not None:
        for col in cols:
            series = _downsample(signals_df[col].dropna(), max_points)
            fig.add_trace(
                go.Scatter(x=series.index, y=series.values, name=col,
                           line=dict(width=0.8)),
                row=2, col=1,
            )
        fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.4, row=2, col=1)

    fig.update_layout(
        title=title or "Market Data & Signals",
        height=500 if n_rows == 1 else 650,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def load_results(path: str) -> BacktestReport:
    """Load a backtest result directory as a BacktestReport.

    Args:
        path: Local directory path or S3 URI containing backtest results.

    Returns:
        BacktestReport with all curves and metrics ready to use.
    """
    results = BacktestResults.from_parquet(path)
    return BacktestReport(results)


def compare_results(*paths: str, metric: str = "sharpe") -> pd.DataFrame:
    """Load multiple backtest result directories and return a comparison table.

    Args:
        *paths: Local or S3 paths to backtest result directories.
        metric: Column to sort by (descending). Default: "sharpe".

    Returns:
        DataFrame with one row per backtest, sorted by metric descending.
    """
    rows = []
    for path in paths:
        try:
            report = load_results(path)
            summary = report.summary()
            pnl = report.pnl_curve
            max_dd = float((pnl - pnl.cummax()).min()) if not pnl.empty else float("nan")
            rows.append({
                "path": path,
                "final_pnl": summary.get("final_pnl"),
                "sharpe": summary.get("sharpe"),
                "sortino": summary.get("sortino"),
                "fill_count": summary.get("fill_count"),
                "max_drawdown": max_dd,
                "total_fees": summary.get("total_fees"),
                "pct_positive_buckets": summary.get("pct_positive_buckets"),
            })
        except Exception as exc:
            rows.append({"path": path, "error": str(exc)})

    df = pd.DataFrame(rows)
    if metric in df.columns and df[metric].notna().any():
        df = df.sort_values(metric, ascending=False)
    return df.reset_index(drop=True)
