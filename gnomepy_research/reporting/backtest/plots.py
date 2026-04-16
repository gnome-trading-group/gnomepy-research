"""Plotly chart builders for BacktestReport."""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from gnomepy_research.reporting.backtest.metrics import _is_buy

if TYPE_CHECKING:
    from gnomepy_research.reporting.backtest.report import BacktestReport


@dataclass
class ReportSection:
    """A configurable section of the HTML report.

    ``render`` should accept a ``BacktestReport`` and return:
    - ``str`` — raw HTML inserted inside a ``<section>`` wrapper
    - ``go.Figure`` — auto-converted to embedded plotly HTML
    - ``None`` — section is silently skipped
    """

    name: str
    title: str
    render: Callable[..., str | go.Figure | None]

pio.templates.default = "ggplot2"


_MUTED_COLORS = [
    "rgba(150,150,150,0.5)", "rgba(100,149,237,0.5)", "rgba(180,120,80,0.5)",
    "rgba(120,180,120,0.5)", "rgba(180,100,180,0.5)", "rgba(200,180,100,0.5)",
]


def _iter_symbols(market_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Get unique (exchange_id, security_id) pairs from market data."""
    if market_df.empty:
        return []
    return list(
        market_df.groupby(["exchange_id", "security_id"], sort=False).ngroups
        and market_df.groupby(["exchange_id", "security_id"], sort=False).groups.keys()
    )


def _sym_label(eid: int, sid: int) -> str:
    return f"{eid}/{sid}"


def _downsample(series: pd.Series, max_points: int | None) -> pd.Series:
    """Evenly downsample a Series if it exceeds *max_points*."""
    if max_points is None or len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series.iloc[::step]


def plot_pnl(
    report: "BacktestReport",
    *,
    show_mid: bool = True,
    show_fills: bool = True,
    title: str | None = None,
    max_points: int | None = None,
) -> go.Figure:
    """PnL + drawdown chart with optional mid-price overlay and fill markers."""
    pnl = _downsample(report.pnl_curve, max_points)
    peak = pnl.cummax()
    dd = pnl - peak

    specs = [[{"secondary_y": show_mid}], [{"secondary_y": False}]]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Mark-to-market PnL", "Drawdown"),
        specs=specs,
    )

    # Mid price — one line per symbol, behind everything.
    # Dedupe timestamps for cleaner display (keep last mid per nanosecond).
    if show_mid and not report._market_df.empty:
        symbols = _iter_symbols(report._market_df)
        for i, (eid, sid) in enumerate(symbols):
            sym_mkt = report._market_df[
                (report._market_df["exchange_id"] == eid)
                & (report._market_df["security_id"] == sid)
            ].sort_index()
            sym_mkt = sym_mkt[~sym_mkt.index.duplicated(keep="last")]
            mid = _downsample(sym_mkt["mid_price"].astype(float), max_points)
            color = _MUTED_COLORS[i % len(_MUTED_COLORS)]
            name = "mid price" if len(symbols) == 1 else f"mid {_sym_label(eid, sid)}"
            fig.add_trace(
                go.Scatter(
                    x=mid.index, y=mid.values, mode="lines", name=name,
                    line=dict(width=0.8, color=color),
                    hoverinfo="skip",
                ),
                row=1, col=1, secondary_y=True,
            )

    fig.add_trace(
        go.Scatter(x=pnl.index, y=pnl.values, mode="lines", name="PnL",
                   line=dict(width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown",
                   line=dict(width=1, color="crimson"), fill="tozeroy",
                   fillcolor="rgba(220,20,60,0.2)"),
        row=2, col=1,
    )

    # Fill markers on the PnL curve.
    fills = report.fills
    if show_fills and not fills.empty:
        pnl_df = pd.DataFrame({"timestamp": pnl.index, "pnl_val": pnl.values})
        fills_df = pd.DataFrame({"timestamp": fills.index}).reset_index(drop=True)
        eq_merged = pd.merge_asof(
            fills_df.sort_values("timestamp"),
            pnl_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        eq_at_fills = eq_merged["pnl_val"].values
        buy_mask = fills["side"].map(_is_buy).values
        fig.add_trace(
            go.Scatter(
                x=fills.index[buy_mask], y=eq_at_fills[buy_mask], mode="markers",
                name="buy",
                marker=dict(symbol="triangle-up", size=7, color="#2ca02c"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fills.index[~buy_mask], y=eq_at_fills[~buy_mask], mode="markers",
                name="sell",
                marker=dict(symbol="triangle-down", size=7, color="#d62728"),
            ),
            row=1, col=1,
        )

    fig.update_layout(
        height=600,
        title=title or "Mark-to-market PnL",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    fig.update_yaxes(title_text="PnL", row=1, col=1, secondary_y=False)
    if show_mid:
        fig.update_yaxes(title_text="mid", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="DD", row=2, col=1)
    fig.update_xaxes(title_text="time", row=2, col=1)
    return fig


_POSITION_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_position(
    report: "BacktestReport",
    *,
    title: str | None = None,
    max_points: int | None = None,
) -> go.Figure:
    """Position, cumulative fees, and cumulative volume subplots."""
    pos_by_sym = report.position_by_symbol
    fees = _downsample(report.fees_curve, max_points)
    vol = _downsample(report.volume_curve, max_points)

    multi = len(pos_by_sym.columns) > 1

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Position", "Cumulative fees", "Cumulative volume"),
    )

    if multi:
        # Per-symbol position lines.
        for i, col in enumerate(pos_by_sym.columns):
            label = _sym_label(*col) if isinstance(col, tuple) else str(col)
            s = _downsample(pos_by_sym[col], max_points)
            color = _POSITION_COLORS[i % len(_POSITION_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values, mode="lines", name=label,
                    line=dict(width=1, shape="hv", color=color),
                ),
                row=1, col=1,
            )
        # Also plot total as dashed.
        total = _downsample(report.position_curve, max_points)
        fig.add_trace(
            go.Scatter(
                x=total.index, y=total.values, mode="lines", name="total",
                line=dict(width=1.5, shape="hv", color="#333", dash="dash"),
            ),
            row=1, col=1,
        )
    else:
        pos = _downsample(report.position_curve, max_points)
        fig.add_trace(
            go.Scatter(
                x=pos.index, y=pos.values, mode="lines", name="position",
                line=dict(width=1, shape="hv", color="#1f77b4"),
                fill="tozeroy", fillcolor="rgba(31,119,180,0.2)",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=fees.index, y=fees.values, mode="lines", name="fees",
            line=dict(width=1, color="#ff7f0e"),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vol.index, y=vol.values, mode="lines", name="volume",
            line=dict(width=1, color="#9467bd"),
        ),
        row=3, col=1,
    )
    fig.update_layout(
        height=650, hovermode="x unified",
        showlegend=multi,
        legend=dict(orientation="h", y=1.05) if multi else None,
        title=title or "Position / Fees / Volume",
    )
    return fig


def plot_pnl_by_symbol(
    report: "BacktestReport",
    *,
    title: str | None = None,
    max_points: int | None = None,
) -> go.Figure:
    """Per-symbol PnL curves on a single chart."""
    by_sym = report.pnl_by_symbol
    fig = go.Figure()
    for col in by_sym.columns:
        label = f"{col[0]}/{col[1]}" if isinstance(col, tuple) else str(col)
        s = _downsample(by_sym[col], max_points)
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines",
                name=label, line=dict(width=1.2),
            ),
        )
    fig.update_layout(
        height=450,
        title=title or "PnL by symbol",
        hovermode="x unified",
        yaxis_title="PnL",
        xaxis_title="time",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def _fmt(key: str, value) -> str:
    """Format a metric value using _METRIC_FORMAT or a sensible default."""
    fmt = _METRIC_FORMAT.get(key)
    if fmt:
        return f"{value:{fmt}}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _signed_color(value: float) -> str:
    return "#16a34a" if value >= 0 else "#dc2626"


_KPI_KEYS = [
    ("final_pnl", "Final PnL", True),
    ("sharpe", "Sharpe", True),
    ("sortino", "Sortino", True),
    ("total_fees", "Total Fees", True),
    ("fill_count", "Fills", False),
    ("final_position", "Final Position", False),
    ("total_volume", "Total Volume", False),
    ("total_notional", "Notional Volume", False),
    ("duration_seconds", "Duration", False),
]


def _kpi_cards_html(summary: dict) -> str:
    """Render key metrics as styled cards."""
    cards = []
    for key, label, colored in _KPI_KEYS:
        value = summary.get(key, 0)
        formatted = _fmt(key, value)
        if key == "duration_seconds":
            formatted += "s"
        color = _signed_color(value) if colored else "#334155"
        cards.append((label, formatted, color))

    items = []
    for label, value, color in cards:  # noqa: E501
        items.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>'
        )
    return '<div class="kpi-row">' + "".join(items) + '</div>'


_SPREAD_COLORS = ["#6366f1", "#e11d48", "#059669", "#d97706", "#7c3aed"]


def plot_spread(
    report: "BacktestReport",
    *,
    title: str | None = None,
    max_points: int | None = None,
) -> go.Figure:
    """Bid-ask spread over time, one line per symbol."""
    mkt = report._market_df.sort_index()
    symbols = _iter_symbols(mkt)
    multi = len(symbols) > 1

    fig = go.Figure()
    for i, (eid, sid) in enumerate(symbols):
        sym_mkt = mkt[(mkt["exchange_id"] == eid) & (mkt["security_id"] == sid)]
        if "spread" in sym_mkt.columns:
            spread = _downsample(sym_mkt["spread"].astype(float), max_points)
        else:
            spread = _downsample(
                (sym_mkt["ask_price_0"] - sym_mkt["bid_price_0"]).astype(float),
                max_points,
            )
        name = f"spread {_sym_label(eid, sid)}" if multi else "spread"
        color = _SPREAD_COLORS[i % len(_SPREAD_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=spread.index, y=spread.values, mode="lines", name=name,
                line=dict(width=0.8, color=color),
            ),
        )

    fig.update_layout(
        height=350,
        title=title or "Bid-Ask Spread",
        hovermode="x unified",
        yaxis_title="spread",
        xaxis_title="time",
        showlegend=multi,
    )
    return fig


def plot_cross_exchange_spread(
    report: "BacktestReport",
    exchange_id_a: int | None = None,
    exchange_id_b: int | None = None,
    security_id: int | None = None,
    *,
    title: str | None = None,
    max_points: int | None = None,
) -> go.Figure:
    """Spread between two exchanges for the same security (in bps).

    If exchange/security IDs are not provided, auto-detects from the first
    two distinct exchange_ids in the market data.
    """
    mkt = report._market_df.sort_index()
    symbols = _iter_symbols(mkt)

    # Auto-detect exchanges if not specified.
    if exchange_id_a is None or exchange_id_b is None:
        exchange_ids = sorted(set(eid for eid, _ in symbols))
        if len(exchange_ids) < 2:
            fig = go.Figure()
            fig.update_layout(title="Cross-Exchange Spread (need 2+ exchanges)", height=350)
            return fig
        exchange_id_a = exchange_ids[0]
        exchange_id_b = exchange_ids[1]
    if security_id is None:
        security_id = symbols[0][1] if symbols else 1

    mkt_a = mkt[(mkt["exchange_id"] == exchange_id_a) & (mkt["security_id"] == security_id)]
    mkt_b = mkt[(mkt["exchange_id"] == exchange_id_b) & (mkt["security_id"] == security_id)]

    if mkt_a.empty or mkt_b.empty:
        fig = go.Figure()
        fig.update_layout(title="Cross-Exchange Spread (missing data)", height=350)
        return fig

    # Align via merge_asof — for each tick of A, find latest B mid.
    a_df = pd.DataFrame({"timestamp": mkt_a.index, "mid_a": mkt_a["mid_price"].astype(float).values}).reset_index(drop=True)
    b_df = pd.DataFrame({"timestamp": mkt_b.index, "mid_b": mkt_b["mid_price"].astype(float).values}).reset_index(drop=True)

    merged = pd.merge_asof(
        a_df.sort_values("timestamp"),
        b_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).dropna()

    avg_mid = (merged["mid_a"] + merged["mid_b"]) / 2
    valid = avg_mid > 0
    spread_bps = pd.Series(
        ((merged["mid_b"] - merged["mid_a"]) / avg_mid * 10_000).values,
        index=pd.DatetimeIndex(merged["timestamp"].values),
    )[valid.values]

    spread_bps = _downsample(spread_bps, max_points)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spread_bps.index, y=spread_bps.values, mode="lines",
            name=f"{exchange_id_b} - {exchange_id_a}",
            line=dict(width=1, color="#6366f1"),
        ),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.4)

    fig.update_layout(
        height=350,
        title=title or f"Cross-Exchange Spread (ex {exchange_id_a} vs {exchange_id_b}, bps)",
        hovermode="x unified",
        yaxis_title="spread (bps)",
        xaxis_title="time",
    )
    return fig


MAX_FILLS_DISPLAY = 500


def _fills_table_html(report: "BacktestReport") -> str:
    """Render fills as a scrollable HTML table, capped at MAX_FILLS_DISPLAY rows."""
    fills = report.fills
    if fills.empty:
        return '<p class="muted">No fills.</p>'

    total = len(fills)
    truncated = total > MAX_FILLS_DISPLAY
    df = fills.head(MAX_FILLS_DISPLAY) if truncated else fills

    cols = ["side", "fill_price", "fill_qty", "fee"]
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()
    df.index = df.index.strftime("%Y-%m-%d %H:%M:%S.%f")
    df.index.name = "timestamp"

    html = df.to_html(
        classes="fills-table",
        float_format=lambda x: f"{x:,.6f}",
    )
    note = ""
    if truncated:
        note = f'<p class="muted">Showing first {MAX_FILLS_DISPLAY:,} of {total:,} fills.</p>'
    return f'<div class="fills-scroll">{html}</div>{note}'


_METRIC_FORMAT: dict[str, str] = {
    "duration_seconds": ",.1f",
    "market_record_count": ",",
    "intent_record_count": ",",
    "fill_count": ",",
    "total_volume": ",.2f",
    "total_notional": ",.2f",
    "final_position": ",.4f",
    "total_fees": ",.2f",
    "final_pnl": ",.2f",
    "sharpe": ",.3f",
    "sortino": ",.3f",
    "sharpe_std": ",.3f",
    "pct_positive_buckets": ".1%",
}


def _summary_table_html(summary: dict) -> str:
    """Render the full summary as a styled table, excluding nested dicts."""
    rows = []
    for k, v in summary.items():
        if isinstance(v, dict):
            continue
        fmt = _METRIC_FORMAT.get(k)
        if fmt:
            formatted = f"{v:{fmt}}"
        elif isinstance(v, float):
            formatted = f"{v:,.4f}"
        elif isinstance(v, int):
            formatted = f"{v:,}"
        else:
            formatted = str(v)
        label = k.replace("_", " ").title()
        rows.append(f"<tr><th>{label}</th><td>{formatted}</td></tr>")
    return f'<table class="summary-table"><tbody>{"".join(rows)}</tbody></table>'


def _render_config(report: "BacktestReport") -> str | None:
    """Render the backtest config as a YAML code block."""
    config = getattr(report, "_config", None)
    if not config:
        return None
    try:
        import yaml
        config_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    except ImportError:
        import json
        config_str = json.dumps(config, indent=2, default=str)
    return f'<pre class="config-block"><code>{config_str}</code></pre>'


def _render_kpi(report: "BacktestReport") -> str:
    return _kpi_cards_html(report.summary())


def _render_summary_table(report: "BacktestReport") -> str:
    return _summary_table_html(report.summary())


def _render_pnl(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_pnl(report, **kwargs)


def _render_position(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_position(report, **kwargs)


def _render_spread(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_spread(report, **kwargs)


def _render_per_symbol(report: "BacktestReport", **kwargs) -> go.Figure | None:
    if len(report.pnl_by_symbol.columns) <= 1:
        return None
    return plot_pnl_by_symbol(report, **kwargs)


def _render_fills(report: "BacktestReport") -> str:
    return _fills_table_html(report)


DEFAULT_SECTIONS: list[ReportSection] = [
    ReportSection("kpi", "Summary", _render_kpi),
    ReportSection("config", "Configuration", _render_config),
    ReportSection("summary_table", "Details", _render_summary_table),
    ReportSection("pnl", "Performance", _render_pnl),
    ReportSection("position", "Position & Execution", _render_position),
    ReportSection("spread", "Market Conditions", _render_spread),
    ReportSection("per_symbol", "Per-Symbol Breakdown", _render_per_symbol),
    ReportSection("fills", "Fills", _render_fills),
]


def _render_cross_exchange(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_cross_exchange_spread(report, **kwargs)


def _render_adverse_selection(report: "BacktestReport", **kwargs) -> "go.Figure | None":
    from gnomepy_research.reporting.backtest.adverse_selection import adverse_selection_section
    return adverse_selection_section(report)


def _render_market_making(report: "BacktestReport", **kwargs) -> "str | None":
    from gnomepy_research.reporting.backtest.market_making import market_making_section
    return market_making_section(report)


SECTION_REGISTRY: dict[str, ReportSection] = {
    "adverse_selection": ReportSection("adverse_selection", "Adverse Selection", _render_adverse_selection),
    "market_making": ReportSection("market_making", "Market Making", _render_market_making),
    "cross_exchange_spread": ReportSection("cross_exchange_spread", "Cross-Exchange Spread", _render_cross_exchange),
}


def resolve_sections(config: dict) -> tuple[list[str], list[ReportSection], int | None]:
    """Read the ``report`` key from a backtest config and resolve sections.

    Returns ``(exclude, extra_sections, max_points)``.
    """
    report_cfg = config.get("report") if config else None
    if not report_cfg:
        return [], [], None

    exclude = list(report_cfg.get("exclude") or [])
    max_points = report_cfg.get("max_points")

    extra: list[ReportSection] = []
    for entry in report_cfg.get("sections") or []:
        name = entry if isinstance(entry, str) else entry.get("name")
        if not name:
            continue
        section = SECTION_REGISTRY.get(name)
        if not section:
            continue
        args = entry.get("args", {}) if isinstance(entry, dict) else {}
        if args:
            bound_render = functools.partial(section.render, **args)
            extra.append(ReportSection(section.name, section.title, bound_render))
        else:
            extra.append(section)

    return exclude, extra, max_points


def assemble_html(
    report: "BacktestReport",
    *,
    exclude: list[str] | None = None,
    extra_sections: list[ReportSection] | None = None,
    extra_figs: list[go.Figure] | None = None,
    max_points: int | None = None,
) -> str:
    """Build a standalone HTML report with configurable sections.

    Parameters
    ----------
    report : BacktestReport
    exclude : list[str], optional
        Section names to skip (e.g. ``["spread", "fills"]``).
    extra_sections : list[ReportSection], optional
        Additional sections appended after the defaults.
    extra_figs : list[go.Figure], optional
        Legacy — extra plotly figures appended as unnamed sections.
    max_points : int, optional
        Downsample chart traces to at most this many points.
        ``None`` (default) keeps all data points.
    """
    from datetime import datetime as dt

    excluded = set(exclude or [])

    all_sections = list(DEFAULT_SECTIONS)
    if extra_sections:
        all_sections.extend(extra_sections)
    if extra_figs:
        for i, fig in enumerate(extra_figs):
            all_sections.append(ReportSection(
                f"_extra_{i}", f"Additional Chart {i + 1}", lambda r, f=fig: f,
            ))

    # Date range from market data.
    if not report._market_df.empty:
        start = report._market_df.index.min()
        end = report._market_df.index.max()
        date_range = f"{start} &mdash; {end}"
    else:
        date_range = "N/A"

    # Render sections.
    plotlyjs_included = False
    body_parts: list[str] = []

    for sec in all_sections:
        if sec.name in excluded:
            continue
        try:
            result = sec.render(report, max_points=max_points)
        except TypeError:
            result = sec.render(report)
        if result is None:
            continue

        if isinstance(result, go.Figure):
            chart_height = result.layout.height or 500
            html = result.to_html(
                full_html=False,
                include_plotlyjs=not plotlyjs_included,
                config={"responsive": True},
                default_height=f"{chart_height}px",
                default_width="100%",
            )
            plotlyjs_included = True
        else:
            html = result

        body_parts.append(f'<section><h2>{sec.title}</h2>\n{html}\n</section>')

    body_html = "\n".join(body_parts)

    generated_at = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from importlib.metadata import version
        ver = version("gnomepy_research")
    except Exception:
        ver = "unknown"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 24px;
    background: #f8f9fa; color: #1e293b;
  }}
  header {{
    border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px;
  }}
  header h1 {{ margin: 0 0 4px 0; font-size: 1.5rem; color: #0f172a; }}
  header .meta {{ font-size: 0.85rem; color: #64748b; }}
  .kpi-row {{
    display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px;
  }}
  .kpi-card {{
    flex: 1 1 150px; background: #fff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 16px 20px; min-width: 140px;
  }}
  .kpi-value {{
    font-size: 1.35rem; font-weight: 700; font-family: "SF Mono", "Fira Code", monospace;
  }}
  .kpi-label {{
    font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.05em; margin-top: 4px;
  }}
  section {{ margin-bottom: 36px; }}
  h2 {{
    font-size: 1.1rem; color: #334155; border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px; margin-bottom: 16px;
  }}
  .summary-table {{
    border-collapse: collapse; width: 100%; margin: 16px 0;
    font-size: 0.9rem;
  }}
  .summary-table th, .summary-table td {{
    border: 1px solid #e2e8f0; padding: 8px 14px;
  }}
  .summary-table th {{
    background: #f1f5f9; text-align: left; font-weight: 600; width: 220px;
  }}
  .summary-table td {{
    text-align: right; font-family: "SF Mono", "Fira Code", monospace; font-size: 0.85rem;
  }}
  .summary-table tr:nth-child(even) {{ background: #f8fafc; }}
  .fills-scroll {{
    max-height: 400px; overflow-y: auto; border: 1px solid #e2e8f0;
    border-radius: 8px; margin-top: 12px;
  }}
  .fills-table {{
    border-collapse: collapse; width: 100%; font-size: 0.8rem;
  }}
  .fills-table th, .fills-table td {{
    border-bottom: 1px solid #f1f5f9; padding: 6px 12px; text-align: right;
  }}
  .fills-table th {{
    background: #f1f5f9; position: sticky; top: 0; text-align: left;
    font-weight: 600; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .fills-table td {{ font-family: "SF Mono", "Fira Code", monospace; }}
  .fills-table tr:hover {{ background: #f8fafc; }}
  .plotly-graph-div {{ width: 100% !important; min-height: 350px; }}
  .js-plotly-plot .plotly {{ min-height: 350px; }}
  .config-block {{
    background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 16px 20px; overflow-x: auto;
    font-family: "SF Mono", "Fira Code", monospace; font-size: 0.82rem;
    line-height: 1.5; color: #334155;
  }}
  .config-block code {{ background: none; padding: 0; }}
  .muted {{ color: #94a3b8; font-style: italic; }}
  footer {{
    margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e8f0;
    font-size: 0.75rem; color: #94a3b8;
  }}
</style>
</head>
<body>
<header>
  <h1>Backtest Report</h1>
  <div class="meta">{date_range}</div>
</header>

{body_html}

<footer>
  Generated {generated_at} &middot; gnomepy-research {ver}
</footer>
</body>
</html>"""
