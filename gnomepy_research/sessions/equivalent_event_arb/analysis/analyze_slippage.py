"""
Slippage analysis for EquivalentEventArb backtest results.

For each taker order, computes:
  - Actual slippage (avg_fill_price - best_price at order submission)
  - Fill ratio at 1c, 2c, 3c, 5c windows (order_qty / cheap_depth)
  - Whether a depth-coverage gate with various parameters would have blocked the order

Usage:
  poetry run python3 gnomepy_research/sessions/equivalent_event_arb/analyze_slippage.py <output_dir> [<output_dir2> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TICK = 0.01  # prediction market tick size ($0.01)


def load_backtest(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fills = pd.read_parquet(output_dir / "fills.parquet")
    orders = pd.read_parquet(output_dir / "orders.parquet")
    market = pd.read_parquet(output_dir / "market.parquet")
    return fills, orders, market


def get_book_at(market: pd.DataFrame, exchange_id: int, security_id: int, ts: pd.Timestamp) -> pd.Series | None:
    """Return the most recent book snapshot at or before ts for the given listing."""
    mask = (market["exchange_id"] == exchange_id) & (market["security_id"] == security_id)
    sub = market.loc[mask]
    idx = sub.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return sub.iloc[idx]


def cheap_depth(row: pd.Series, side: str, window_cents: float) -> float:
    """Sum of depth within window_cents of best price on the relevant side."""
    if side == "Bid":  # buying, care about asks
        best = row.get("ask_price_0", float("nan"))
        ceiling = best + window_cents * TICK
        total = 0.0
        for i in range(10):
            p = row.get(f"ask_price_{i}", float("nan"))
            s = row.get(f"ask_size_{i}", 0.0)
            if pd.isna(p) or p <= 0:
                break
            if p > ceiling:
                break
            total += s
        return total
    else:  # selling, care about bids
        best = row.get("bid_price_0", float("nan"))
        floor = best - window_cents * TICK
        total = 0.0
        for i in range(10):
            p = row.get(f"bid_price_{i}", float("nan"))
            s = row.get(f"bid_size_{i}", 0.0)
            if pd.isna(p) or p <= 0:
                break
            if p < floor:
                break
            total += s
        return total


def best_price(row: pd.Series, side: str) -> float:
    if side == "Bid":
        return row.get("ask_price_0", float("nan"))
    else:
        return row.get("bid_price_0", float("nan"))


def analyze(output_dir: Path) -> pd.DataFrame:
    fills, orders, market = load_backtest(output_dir)

    print(f"\n{'='*60}")
    print(f"Event: {output_dir}")
    print(f"Market columns (first 12): {market.columns.tolist()[:12]}")
    print(f"Orders: {len(orders)}, Fills: {len(fills)}")

    results = []
    windows = [1, 2, 3, 5]

    for ts, order in orders.iterrows():
        if order["filled_qty"] <= 0:
            continue

        eid = int(order["exchange_id"])
        sid = int(order["security_id"])
        side = order["side"]
        qty = order["filled_qty"]
        avg_fill = order["avg_fill_price"]
        limit_price = order["submit_price"]

        # Determine if taker: ack_timestamp is epoch (1970) for taker orders
        is_taker = order["ack_timestamp"].year == 1970

        # Book snapshot at order submission
        snap = get_book_at(market, eid, sid, ts)
        if snap is None:
            continue

        bp = best_price(snap, side)
        if pd.isna(bp) or bp <= 0:
            continue

        # Actual slippage
        if side == "Bid":
            slippage_cents = (avg_fill - bp) * 100  # positive = paid above best ask
        else:
            slippage_cents = (bp - avg_fill) * 100   # positive = received below best bid

        # Fill ratios at various windows
        ratios = {}
        depths = {}
        for w in windows:
            d = cheap_depth(snap, side, w)
            depths[f"depth_{w}c"] = d
            ratios[f"fill_ratio_{w}c"] = qty / d if d > 0 else float("inf")

        row = {
            "event": output_dir.name,
            "exchange_id": eid,
            "security_id": sid,
            "side": side,
            "timestamp": ts,
            "qty": qty,
            "limit_price": limit_price,
            "best_price_at_submit": bp,
            "avg_fill_price": avg_fill,
            "slippage_cents": slippage_cents,
            "is_taker": is_taker,
            **depths,
            **ratios,
        }
        results.append(row)

    df = pd.DataFrame(results)
    return df


def coverage_threshold_analysis(df: pd.DataFrame, window: int, multipliers: list[float]) -> None:
    """For each coverage multiplier, compute what fraction of fills would be blocked and the P&L impact."""
    takers = df[df["is_taker"]].copy()
    if takers.empty:
        print("  No taker fills found.")
        return

    ratio_col = f"fill_ratio_{window}c"
    print(f"\n  Coverage gate analysis (window={window}c):")
    print(f"  {'Multiplier':>12} | {'Blocked':>8} | {'Blocked%':>9} | {'Avg slip blocked':>17} | {'Avg slip passed':>16} | {'FP (good trades blocked)':>24}")
    print("  " + "-" * 100)

    for mult in multipliers:
        blocked = takers[ratio_col] > mult
        n_blocked = blocked.sum()
        pct_blocked = 100 * n_blocked / len(takers)
        avg_slip_blocked = takers.loc[blocked, "slippage_cents"].mean() if n_blocked > 0 else float("nan")
        avg_slip_passed = takers.loc[~blocked, "slippage_cents"].mean() if (~blocked).sum() > 0 else float("nan")
        # False positives: blocked trades with negative slippage (i.e., good fills)
        fp = (blocked & (takers["slippage_cents"] <= 0)).sum()
        print(f"  {mult:>12.1f} | {n_blocked:>8} | {pct_blocked:>8.1f}% | {avg_slip_blocked:>16.2f}c | {avg_slip_passed:>15.2f}c | {fp:>24}")


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SLIPPAGE ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nTotal fills analyzed: {len(df)}")
    takers = df[df["is_taker"]]
    makers = df[~df["is_taker"]]
    print(f"  Taker fills: {len(takers)}")
    print(f"  Maker fills: {len(makers)}")

    if takers.empty:
        print("\nNo taker fills found across all events.")
        return

    print("\n--- Taker fills detail ---")
    display_cols = ["event", "exchange_id", "security_id", "side", "qty",
                    "best_price_at_submit", "avg_fill_price", "slippage_cents",
                    "fill_ratio_1c", "fill_ratio_2c", "fill_ratio_3c", "fill_ratio_5c"]
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(takers[display_cols].to_string(index=False))

    print("\n--- Correlation: fill ratio vs slippage (taker fills only) ---")
    for w in [1, 2, 3, 5]:
        col = f"fill_ratio_{w}c"
        valid = takers[[col, "slippage_cents"]].replace([float("inf"), float("-inf")], float("nan")).dropna()
        if len(valid) > 1:
            corr = valid[col].corr(valid["slippage_cents"])
            print(f"  fill_ratio_{w}c vs slippage_cents: r = {corr:.3f}  (n={len(valid)})")

    print("\n--- Coverage gate analysis (what would be blocked at various thresholds) ---")
    for w in [2, 3, 5]:
        coverage_threshold_analysis(takers, w, [1.0, 2.0, 3.0, 5.0, 10.0])

    print("\n--- Slippage distribution (taker fills) ---")
    print(takers["slippage_cents"].describe().round(3))

    print("\n--- By event ---")
    for event, group in takers.groupby("event"):
        print(f"\n  Event: {event} ({len(group)} taker fills)")
        print(f"    Avg slippage: {group['slippage_cents'].mean():.3f}c")
        print(f"    Max slippage: {group['slippage_cents'].max():.3f}c")
        print(f"    Fill ratio 2c: {group['fill_ratio_2c'].replace(float('inf'), float('nan')).mean():.3f}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir> [<output_dir2> ...]")
        sys.exit(1)

    all_dfs = []
    for path in sys.argv[1:]:
        output_dir = Path(path)
        jobs_dir = output_dir / "jobs"
        if jobs_dir.exists():
            for job_dir in sorted(jobs_dir.iterdir()):
                if (job_dir / "fills.parquet").exists():
                    df = analyze(job_dir)
                    all_dfs.append(df)
        elif (output_dir / "fills.parquet").exists():
            df = analyze(output_dir)
            all_dfs.append(df)

    if not all_dfs:
        print("No backtest data found.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    report(combined)


if __name__ == "__main__":
    main()
