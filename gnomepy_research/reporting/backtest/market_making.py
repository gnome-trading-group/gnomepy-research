"""Market-making specific report metrics and visualization."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gnomepy_research.reporting.backtest.metrics import _is_buy

if TYPE_CHECKING:
    from gnomepy_research.reporting.backtest.report import BacktestReport


def _is_mm_strategy(report: "BacktestReport") -> bool:
    """Check if the strategy quoted passively (has intents with bid/ask sizes)."""
    intents = report._intent_df
    if intents.empty:
        return False
    return (intents["bid_size"] > 0).any() or (intents["ask_size"] > 0).any()


def compute_mm_stats(report: "BacktestReport") -> dict:
    """Compute market-making specific scalar metrics."""
    intents = report._intent_df
    fills = report.fills
    market = report._market_df
    position = report.position_curve

    if intents.empty:
        return {}

    total_intents = len(intents)
    fill_count = len(fills)

    # Quoting activity — fraction of market ticks covered by an active quote.
    # For each market tick, find the most recent intent via merge_asof.
    # If that intent has bid_size > 0 or ask_size > 0, the strategy was quoting.
    has_bid = intents["bid_size"] > 0
    has_ask = intents["ask_size"] > 0
    has_both = has_bid & has_ask

    if not market.empty and not intents.empty:
        mkt_sorted = market.sort_index()
        mkt_ts = pd.DataFrame({"timestamp": mkt_sorted.index}).reset_index(drop=True)
        intent_state = pd.DataFrame({
            "timestamp": intents.sort_index().index,
            "has_bid": has_bid.sort_index().values.astype(float),
            "has_ask": has_ask.sort_index().values.astype(float),
        }).reset_index(drop=True)
        merged_state = pd.merge_asof(
            mkt_ts,
            intent_state,
            on="timestamp",
            direction="backward",
        ).fillna(0.0)
        quoting_either = (merged_state["has_bid"] > 0) | (merged_state["has_ask"] > 0)
        quoting_both = (merged_state["has_bid"] > 0) & (merged_state["has_ask"] > 0)
        time_quoting = float(quoting_either.mean())
        time_quoting_both = float(quoting_both.mean())
    else:
        time_quoting = 0.0
        time_quoting_both = 0.0

    # Avg quoted spread vs market spread (where both sides active).
    both_sided = intents[has_both]
    if not both_sided.empty and not market.empty:
        quoted_spread_raw = both_sided["ask_price"].astype(float) - both_sided["bid_price"].astype(float)
        # Get market data at each intent time via merge_asof.
        intent_ts = pd.DataFrame({"timestamp": both_sided.index}).reset_index(drop=True)
        mkt_cols = market[["mid_price", "bid_price_0", "ask_price_0"]].sort_index().reset_index()
        mkt_cols.columns = ["timestamp", "mid_price", "mkt_bid", "mkt_ask"]
        merged_mkt = pd.merge_asof(
            intent_ts.sort_values("timestamp"),
            mkt_cols.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        mid_at_intent = merged_mkt["mid_price"].astype(float).values
        mkt_bid = merged_mkt["mkt_bid"].astype(float).values
        mkt_ask = merged_mkt["mkt_ask"].astype(float).values
        mkt_spread_raw = mkt_ask - mkt_bid

        valid = mid_at_intent > 0
        if valid.any():
            quoted_spread_bps = quoted_spread_raw.values[valid] / mid_at_intent[valid] * 10_000
            market_spread_bps = mkt_spread_raw[valid] / mid_at_intent[valid] * 10_000

            avg_quoted_spread_bps = float(quoted_spread_bps.mean())
            avg_market_spread_bps = float(market_spread_bps.mean())
            # Ratio: <1 means quoting tighter than the market.
            spread_ratio = avg_quoted_spread_bps / avg_market_spread_bps if avg_market_spread_bps > 0 else 0.0

            # Bid/ask improvement: how much tighter are our quotes vs the market?
            our_bid = both_sided["bid_price"].astype(float).values[valid]
            our_ask = both_sided["ask_price"].astype(float).values[valid]
            bid_improvement_bps = float(((our_bid - mkt_bid[valid]) / mid_at_intent[valid] * 10_000).mean())
            ask_improvement_bps = float(((mkt_ask[valid] - our_ask) / mid_at_intent[valid] * 10_000).mean())
        else:
            avg_quoted_spread_bps = 0.0
            avg_market_spread_bps = 0.0
            spread_ratio = 0.0
            bid_improvement_bps = 0.0
            ask_improvement_bps = 0.0
    else:
        avg_quoted_spread_bps = 0.0
        avg_market_spread_bps = 0.0
        spread_ratio = 0.0
        bid_improvement_bps = 0.0
        ask_improvement_bps = 0.0

    # Quote-to-fill ratio.
    quote_to_fill = fill_count / total_intents if total_intents > 0 else 0.0

    # Fill rate by side.
    if not fills.empty:
        buy_mask = fills["side"].map(_is_buy)
        buy_count = int(buy_mask.sum())
        sell_count = int((~buy_mask).sum())
        buy_sell_ratio = buy_count / sell_count if sell_count > 0 else float("inf")
    else:
        buy_sell_ratio = 0.0

    # Inventory stats.
    if not position.empty:
        abs_pos = position.abs()
        max_abs_position = float(abs_pos.max())
        mean_abs_position = float(abs_pos.mean())
    else:
        max_abs_position = 0.0
        mean_abs_position = 0.0

    total_volume = float(report.volume_curve.iloc[-1]) if not report.volume_curve.empty else 0.0
    position_turnover = total_volume / mean_abs_position if mean_abs_position > 0 else 0.0

    # Edge captured per fill — how far was the fill price from mid (in bps).
    # Buys: edge = (mid - fill_price) / mid * 10_000  (positive = bought below mid)
    # Sells: edge = (fill_price - mid) / mid * 10_000  (positive = sold above mid)
    if not fills.empty and not market.empty:
        fill_ts = pd.DataFrame({"timestamp": fills.index}).reset_index(drop=True)
        mid_df = market[["mid_price"]].sort_index().reset_index()
        mid_df.columns = ["timestamp", "mid_price"]
        merged = pd.merge_asof(
            fill_ts.sort_values("timestamp"),
            mid_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        mid_at_fill = merged["mid_price"].astype(float).values
        fill_price = fills["fill_price"].astype(float).values
        sign = fills["side"].map(lambda s: 1.0 if _is_buy(s) else -1.0).values

        valid = mid_at_fill > 0
        if valid.any():
            edge_bps = (mid_at_fill[valid] - fill_price[valid]) * sign[valid] / mid_at_fill[valid] * 10_000
            avg_edge_bps = float(edge_bps.mean())
        else:
            avg_edge_bps = 0.0

        avg_fee_per_fill = float(fills["fee"].fillna(0).abs().mean()) if "fee" in fills.columns else 0.0
        avg_fee_bps = float(avg_fee_per_fill / mid_at_fill[valid].mean() * 10_000) if valid.any() and mid_at_fill[valid].mean() > 0 else 0.0
        avg_net_edge_bps = avg_edge_bps - avg_fee_bps
    else:
        avg_edge_bps = 0.0
        avg_net_edge_bps = 0.0

    return {
        "total_intents": total_intents,
        "time_quoting": time_quoting,
        "time_quoting_both_sides": time_quoting_both,
        "avg_quoted_spread_bps": avg_quoted_spread_bps,
        "avg_market_spread_bps": avg_market_spread_bps,
        "quoted_vs_market_spread": spread_ratio,
        "bid_improvement_bps": bid_improvement_bps,
        "ask_improvement_bps": ask_improvement_bps,
        "quote_to_fill_ratio": quote_to_fill,
        "buy_sell_fill_ratio": buy_sell_ratio,
        "max_abs_position": max_abs_position,
        "mean_abs_position": mean_abs_position,
        "position_turnover": position_turnover,
        "avg_edge_captured_bps": avg_edge_bps,
        "avg_net_edge_captured_bps": avg_net_edge_bps,
    }


def _mm_stats_table_html(stats: dict) -> str:
    """Render MM stats as a styled HTML table."""
    fmt_map = {
        "total_intents": ",",
        "time_quoting": ".1%",
        "time_quoting_both_sides": ".1%",
        "avg_quoted_spread_bps": ",.2f",
        "avg_market_spread_bps": ",.2f",
        "quoted_vs_market_spread": ",.2f",
        "bid_improvement_bps": ",.2f",
        "ask_improvement_bps": ",.2f",
        "quote_to_fill_ratio": ".4f",
        "buy_sell_fill_ratio": ",.2f",
        "max_abs_position": ",.4f",
        "mean_abs_position": ",.4f",
        "position_turnover": ",.1f",
        "avg_edge_captured_bps": ",.2f",
        "avg_net_edge_captured_bps": ",.2f",
    }
    rows = []
    for k, v in stats.items():
        fmt = fmt_map.get(k, ",.4f")
        formatted = f"{v:{fmt}}"
        label = k.replace("_", " ").title()
        rows.append(f"<tr><th>{label}</th><td>{formatted}</td></tr>")
    return f'<table class="summary-table"><tbody>{"".join(rows)}</tbody></table>'


def plot_mm_dashboard(
    report: "BacktestReport",
    *,
    title: str | None = None,
    max_points: int | None = None,
) -> go.Figure:
    """Quoted spread over time, position histogram, fill side bar chart."""
    intents = report._intent_df
    fills = report.fills
    position = report.position_curve

    market = report._market_df

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Quoted spread (bps)", "Position distribution",
            "Bid/Ask vs best (bps)", "",
        ),
        column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
    )

    # 1. Quoted spread over time.
    both = intents[(intents["bid_size"] > 0) & (intents["ask_size"] > 0)]
    if not both.empty:
        spread = both["ask_price"].astype(float) - both["bid_price"].astype(float)
        mid = (both["ask_price"].astype(float) + both["bid_price"].astype(float)) / 2
        spread_bps = spread / mid * 10_000
        if max_points and len(spread_bps) > max_points:
            step = max(1, len(spread_bps) // max_points)
            spread_bps = spread_bps.iloc[::step]
        fig.add_trace(
            go.Scatter(x=spread_bps.index, y=spread_bps.values, mode="lines",
                       name="quoted spread", line=dict(width=0.8, color="#6366f1")),
            row=1, col=1,
        )

    # 2. Position histogram.
    if not position.empty:
        fig.add_trace(
            go.Histogram(x=position.values, name="position",
                         marker_color="#1f77b4", opacity=0.7, nbinsx=30),
            row=1, col=2,
        )

    # 3. Bid/ask improvement over time (bps relative to best bid/ask).
    if not intents.empty and not market.empty:
        # Build a flat DataFrame with positional alignment — avoids dup index issues.
        active_mask = (intents["bid_size"] > 0) | (intents["ask_size"] > 0)
        active = intents[active_mask].copy()

        if not active.empty:
            active_reset = active.reset_index()
            active_reset.columns = ["timestamp"] + list(active.columns)
            mkt_cols = market[["mid_price", "bid_price_0", "ask_price_0"]].sort_index().reset_index()
            mkt_cols.columns = ["timestamp", "mid", "mkt_bid", "mkt_ask"]
            merged = pd.merge_asof(
                active_reset.sort_values("timestamp"),
                mkt_cols.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            mid_vals = merged["mid"].astype(float).values
            valid = mid_vals > 0
            ts = merged["timestamp"].values

            # Bid improvement: (our_bid - best_bid) / mid * 10_000
            bid_active = (merged["bid_size"] > 0).values
            if (valid & bid_active).any():
                mask = valid & bid_active
                bid_imp_vals = (merged["bid_price"].astype(float).values[mask] - merged["mkt_bid"].astype(float).values[mask]) / mid_vals[mask] * 10_000
                bid_imp = pd.Series(bid_imp_vals, index=pd.DatetimeIndex(ts[mask]))
                if max_points and len(bid_imp) > max_points:
                    bid_imp = bid_imp.iloc[::max(1, len(bid_imp) // max_points)]
                fig.add_trace(
                    go.Scatter(x=bid_imp.index, y=bid_imp.values, mode="lines",
                               name="bid vs best bid", line=dict(width=0.8, color="#2ca02c")),
                    row=2, col=1,
                )

            # Ask improvement: (best_ask - our_ask) / mid * 10_000
            ask_active = (merged["ask_size"] > 0).values
            if (valid & ask_active).any():
                mask = valid & ask_active
                ask_imp_vals = (merged["mkt_ask"].astype(float).values[mask] - merged["ask_price"].astype(float).values[mask]) / mid_vals[mask] * 10_000
                ask_imp = pd.Series(ask_imp_vals, index=pd.DatetimeIndex(ts[mask]))
                if max_points and len(ask_imp) > max_points:
                    ask_imp = ask_imp.iloc[::max(1, len(ask_imp) // max_points)]
                fig.add_trace(
                    go.Scatter(x=ask_imp.index, y=ask_imp.values, mode="lines",
                               name="ask vs best ask", line=dict(width=0.8, color="#d62728")),
                    row=2, col=1,
                )

        fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.4, row=2, col=1)

    fig.update_yaxes(title_text="bps", row=2, col=1)
    fig.update_layout(
        height=600,
        title=title or "Market Making Dashboard",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def market_making_section(report: "BacktestReport") -> str | None:
    """ReportSection render function. Returns HTML with stats + chart."""
    if not _is_mm_strategy(report):
        return None

    stats = compute_mm_stats(report)
    table_html = _mm_stats_table_html(stats)
    fig = plot_mm_dashboard(report)
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    return f"{table_html}\n{chart_html}"
