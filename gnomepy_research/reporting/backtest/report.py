"""BacktestReport — tick-level curves, scalar summaries, plots, and HTML export."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from gnomepy_research.reporting.backtest.metrics import Curves, build_curves, compute_sharpe

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from gnomepy.java.recorder import BacktestResults


def _with_mid_price(market_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure market_df has a ``mid_price`` column.

    The gnomepy recorder writes depth-indexed BBO (``bid_price_0`` /
    ``ask_price_0``) rather than a pre-computed mid. Plot and metric code
    assumes ``mid_price`` exists, so derive it once here.
    """
    if market_df is None or market_df.empty:
        return market_df
    if "mid_price" in market_df.columns:
        return market_df
    if "bid_price_0" in market_df.columns and "ask_price_0" in market_df.columns:
        df = market_df.copy()
        df["mid_price"] = (
            df["bid_price_0"].astype(float) + df["ask_price_0"].astype(float)
        ) / 2.0
        return df
    return market_df


def _extract_config(backtest) -> dict:
    """Build a config dict from a Backtest object's attributes."""
    from dataclasses import asdict

    config: dict = {}

    strategy = getattr(backtest, "_strategy", None)
    if strategy is not None:
        if isinstance(strategy, str):
            config["java_strategy"] = strategy
        else:
            cls = type(strategy)
            config["strategy"] = f"{cls.__module__}:{cls.__name__}"

    schema = getattr(backtest, "_schema_type", None)
    if schema is not None:
        config["schema_type"] = schema.name

    start = getattr(backtest, "_start_date", None)
    if start is not None:
        config["start_date"] = str(start)

    end = getattr(backtest, "_end_date", None)
    if end is not None:
        config["end_date"] = str(end)

    exchanges = getattr(backtest, "_exchanges", None)
    if exchanges:
        config["exchanges"] = [asdict(e) for e in exchanges]

    bucket = getattr(backtest, "_bucket", None)
    if bucket and bucket != "gnome-market-data-prod":
        config["bucket"] = bucket

    return config


class BacktestReport:
    """Wraps ``BacktestResults`` and exposes tick-level curves plus
    scalar summaries.

    Every metric is available as a total (``*_curve``) and a per-symbol
    breakdown (``*_by_symbol``).

    Usage::

        results = backtest.run()
        report = BacktestReport(results)
        report.pnl_curve              # total PnL
        report.pnl_by_symbol          # PnL per (exchange_id, security_id)
        report.pnl_by_symbol[(1, 1)]  # PnL for a specific symbol
        report.summary_df()           # scalar metrics
    """

    def __init__(
        self,
        results: "BacktestResults",
        start_date: pd.Timestamp | str | None = None,
        end_date: pd.Timestamp | str | None = None,
        config: dict | None = None,
        backtest: "Backtest | None" = None,
    ):
        self._results = results
        # Config priority: explicit > preset stashed on backtest > auto-extract from backtest
        if config is not None:
            self._config = config
        elif backtest is not None and hasattr(backtest, "_preset_config"):
            self._config = backtest._preset_config
        elif backtest is not None:
            self._config = _extract_config(backtest)
        else:
            self._config = None
        self._market_df: pd.DataFrame = _with_mid_price(results.market_records_df())
        self._exec_df: pd.DataFrame = results.fills_df()
        self._intent_df: pd.DataFrame = results.intent_records_df()

        if start_date is not None or end_date is not None:
            start = pd.Timestamp(start_date) if start_date is not None else self._market_df.index.min()
            end = pd.Timestamp(end_date) if end_date is not None else self._market_df.index.max()
            self._market_df = self._market_df.loc[start:end]
            if not self._exec_df.empty:
                self._exec_df = self._exec_df.loc[start:end]
            if not self._intent_df.empty:
                self._intent_df = self._intent_df.loc[start:end]

    @classmethod
    def from_dataframes(
        cls,
        market_df: pd.DataFrame,
        exec_df: pd.DataFrame,
        intent_df: pd.DataFrame | None = None,
        config: dict | None = None,
    ) -> "BacktestReport":
        """Create a report from pre-loaded DataFrames (e.g. from parquet files)."""
        obj = object.__new__(cls)
        obj._results = None
        obj._config = config
        obj._market_df = _with_mid_price(market_df)
        obj._exec_df = exec_df
        obj._intent_df = intent_df if intent_df is not None else pd.DataFrame()
        return obj

    @cached_property
    def _curves(self) -> Curves:
        return build_curves(self._market_df, self._exec_df)

    # -- Total curves --------------------------------------------------------

    @property
    def pnl_curve(self) -> pd.Series:
        return self._curves.pnl

    @property
    def position_curve(self) -> pd.Series:
        return self._curves.position

    @property
    def fees_curve(self) -> pd.Series:
        return self._curves.fees

    @property
    def volume_curve(self) -> pd.Series:
        return self._curves.volume

    # -- Per-symbol breakdowns -----------------------------------------------

    @property
    def pnl_by_symbol(self) -> pd.DataFrame:
        return self._curves.pnl_by_symbol

    @property
    def position_by_symbol(self) -> pd.DataFrame:
        return self._curves.position_by_symbol

    @property
    def fees_by_symbol(self) -> pd.DataFrame:
        return self._curves.fees_by_symbol

    @property
    def volume_by_symbol(self) -> pd.DataFrame:
        return self._curves.volume_by_symbol

    @property
    def notional_curve(self) -> pd.Series:
        return self._curves.notional

    @property
    def notional_by_symbol(self) -> pd.DataFrame:
        return self._curves.notional_by_symbol

    # -- Fills ---------------------------------------------------------------

    @property
    def fills(self) -> pd.DataFrame:
        return self._curves.fills

    # -- Market Making -------------------------------------------------------

    def mm_stats(self) -> dict:
        """Market-making specific metrics. Empty dict if not an MM strategy."""
        from gnomepy_research.reporting.backtest.market_making import compute_mm_stats
        return compute_mm_stats(self)

    def plot_mm_dashboard(self, **kwargs) -> "go.Figure":
        """Quoted spread, position histogram, fill side bar chart."""
        from gnomepy_research.reporting.backtest.market_making import plot_mm_dashboard
        return plot_mm_dashboard(self, **kwargs)

    # -- Sharpe --------------------------------------------------------------

    def sharpe_metrics(self, bar: str = "10s", n_buckets: int = 5) -> dict:
        """Compute Sharpe, Sortino, and per-bucket stability metrics."""
        return compute_sharpe(self.pnl_curve, bar=bar, n_buckets=n_buckets)

    # -- Scalars -------------------------------------------------------------

    def summary(self) -> dict:
        c = self._curves
        duration = 0.0
        if not self._market_df.empty:
            duration = float(
                (self._market_df.index.max() - self._market_df.index.min()).total_seconds()
            )

        final_pnl = float(c.pnl.iloc[-1]) if not c.pnl.empty else 0.0
        total_fees = float(c.fees.iloc[-1]) if not c.fees.empty else 0.0
        total_volume = float(c.volume.iloc[-1]) if not c.volume.empty else 0.0
        total_notional = float(c.notional.iloc[-1]) if not c.notional.empty else 0.0
        final_position = float(c.position.iloc[-1]) if not c.position.empty else 0.0

        # Per-symbol breakdowns.
        if not c.pnl_by_symbol.empty:
            final_pnl_by_symbol = {col: float(c.pnl_by_symbol[col].iloc[-1]) for col in c.pnl_by_symbol.columns}
        else:
            final_pnl_by_symbol = {}

        if not c.position_by_symbol.empty:
            final_positions = {col: float(c.position_by_symbol[col].iloc[-1]) for col in c.position_by_symbol.columns}
        else:
            final_positions = {}

        sharpe = self.sharpe_metrics()

        return {
            "duration_seconds": duration,
            "market_record_count": int(self._results.market_record_count) if self._results else len(self._market_df),
            "intent_record_count": int(self._results.intent_record_count) if self._results else len(self._intent_df),
            "fill_count": c.fill_count,
            "total_volume": total_volume,
            "total_notional": total_notional,
            "final_position": final_position,
            "final_positions": final_positions,
            "total_fees": total_fees,
            "final_pnl": final_pnl,
            "final_pnl_by_symbol": final_pnl_by_symbol,
            "sharpe": sharpe["sharpe"],
            "sortino": sharpe["sortino"],
            "sharpe_std": sharpe["sharpe_std"],
            "pct_positive_buckets": sharpe["pct_positive_buckets"],
        }

    def summary_df(self) -> pd.Series:
        s = self.summary()
        # Exclude nested dicts from the Series view.
        flat = {k: v for k, v in s.items() if not isinstance(v, dict)}
        return pd.Series(flat, name="backtest_report")

    # -- Plots ---------------------------------------------------------------

    def plot_pnl(self, **kwargs) -> "go.Figure":
        """PnL + drawdown chart with mid overlay and fill markers."""
        from gnomepy_research.reporting.backtest.plots import plot_pnl
        return plot_pnl(self, **kwargs)

    def plot_position(self, **kwargs) -> "go.Figure":
        """Position, cumulative fees, and cumulative volume subplots."""
        from gnomepy_research.reporting.backtest.plots import plot_position
        return plot_position(self, **kwargs)

    def plot_adverse_selection(self, **kwargs) -> "go.Figure":
        """Average post-fill mid movement curve across horizons."""
        from gnomepy_research.reporting.backtest.adverse_selection import plot_adverse_selection
        return plot_adverse_selection(self, **kwargs)

    def plot_cross_exchange_spread(self, **kwargs) -> "go.Figure":
        """Spread between two exchanges for the same security (bps)."""
        from gnomepy_research.reporting.backtest.plots import plot_cross_exchange_spread
        return plot_cross_exchange_spread(self, **kwargs)

    def plot_spread(self, **kwargs) -> "go.Figure":
        """Bid-ask spread over time."""
        from gnomepy_research.reporting.backtest.plots import plot_spread
        return plot_spread(self, **kwargs)

    def plot_pnl_by_symbol(self, **kwargs) -> "go.Figure":
        """Per-symbol PnL curves on a single chart."""
        from gnomepy_research.reporting.backtest.plots import plot_pnl_by_symbol
        return plot_pnl_by_symbol(self, **kwargs)

    # -- HTML export ---------------------------------------------------------

    def save_html(
        self,
        path: str | Path,
        *,
        exclude: list[str] | None = None,
        extra_sections: list | None = None,
        extra_figs: list["go.Figure"] | None = None,
        max_points: int | None = None,
    ) -> None:
        """Write a standalone HTML report with configurable sections.

        Parameters
        ----------
        exclude : list[str], optional
            Section names to skip (e.g. ``["spread", "fills"]``).
        extra_sections : list[ReportSection], optional
            Additional sections appended after the defaults.
        extra_figs : list[go.Figure], optional
            Extra plotly figures appended as unnamed sections.
        max_points : int, optional
            Downsample chart traces to at most this many points.
            ``None`` (default) keeps all data points.
        """
        from gnomepy_research.reporting.backtest.plots import assemble_html, resolve_sections

        # Merge config-driven sections with explicit args.
        cfg_exclude, cfg_sections, cfg_max_points = resolve_sections(self._config or {})

        merged_exclude = list(exclude or []) + cfg_exclude
        merged_sections = list(extra_sections or []) + cfg_sections
        if max_points is None:
            max_points = cfg_max_points

        Path(path).write_text(assemble_html(
            self, exclude=merged_exclude or None, extra_sections=merged_sections or None,
            extra_figs=extra_figs, max_points=max_points,
        ))

    def __repr__(self) -> str:
        s = self.summary()
        lines = [
            "BacktestReport",
            f"  duration:        {s['duration_seconds']:.2f} s",
            f"  market records:  {s['market_record_count']:,}",
            f"  intents:         {s['intent_record_count']:,}",
            f"  fills:           {s['fill_count']:,}",
            f"  total volume:    {s['total_volume']:.4f}",
            f"  final position:  {s['final_position']:.4f}",
            f"  total fees:      {s['total_fees']:.4f}",
            f"  final PnL:       {s['final_pnl']:.4f}",
        ]
        for sym, pnl in s.get("final_pnl_by_symbol", {}).items():
            lines.append(f"    {sym}: {pnl:.4f}")
        return "\n".join(lines)
