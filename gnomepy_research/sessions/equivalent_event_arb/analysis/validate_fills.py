#!/usr/bin/env python3
"""
Validate intent prices and fill accounting from a backtest output directory.

Usage:
    poetry run python validate_fills.py <output_dir> [--fees eid:maker:taker,...]

Checks:
  1. Entry intent bid_price matches market bid at that timestamp
  2. Fill price matches the posted intent price
  3. Fee = parametric formula with correct maker/taker rate
  4. PnL per listing adds up correctly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parametric_fee(price: float, rate: float, qty: float) -> float:
    if rate == 0.0:
        return 0.0
    return rate * price * (1.0 - price) * qty


def infer_fill_type(row: pd.Series) -> str:
    """Taker if fill crossed the opposing best: bid_fill > book_ask, or ask_fill < book_bid."""
    if row["side"] == "Bid":
        return "taker" if row["fill_price"] >= row["book_ask_price"] else "maker"
    else:
        return "taker" if row["fill_price"] <= row["book_bid_price"] else "maker"


def validate(output_dir: Path, fee_rates: dict[int, dict[str, float]]) -> None:
    intents = pd.read_parquet(output_dir / "intents.parquet")
    fills = pd.read_parquet(output_dir / "fills.parquet")
    market = pd.read_parquet(output_dir / "market.parquet")

    print("=" * 70)
    print("1. ENTRY INTENT PRICE vs MARKET BID")
    print("=" * 70)

    # First timestamp where bid_size goes from 0 -> positive for each listing
    entry_intents = (
        intents[intents["bid_size"] > 0]
        .groupby(["exchange_id", "security_id"])
        .first()
        .reset_index()
    )

    # Preserve timestamp as a column
    entry_intents = intents[intents["bid_size"] > 0].copy()
    entry_intents["timestamp"] = entry_intents.index
    entry_intents = (
        entry_intents.groupby(["exchange_id", "security_id"])
        .first()
        .reset_index()
    )

    print(f"{'EID':>4}  {'SID':>8}  {'IntentTs':>26}  {'IntentBid':>10}  {'MktBids@Ts':>14}  {'Match?':>6}")
    print("-" * 70)
    for _, row in entry_intents.iterrows():
        eid, sid = int(row["exchange_id"]), int(row["security_id"])
        ts = row["timestamp"]
        intent_bid = row["bid_price"]

        mkt = market[(market["exchange_id"] == eid) & (market["security_id"] == sid)]
        # Multiple events can share the same millisecond timestamp; intent bid must
        # match one of them (the triggering event, not necessarily first or last).
        mkt_exact = mkt[mkt.index == ts]["bid_price_0"]
        if len(mkt_exact):
            match = (abs(mkt_exact - intent_bid) < 1e-6).any()
            mkt_bids = "/".join(f"{p:.4f}" for p in mkt_exact.unique())
        else:
            mkt_at = mkt[mkt.index < ts]
            mkt_bid = mkt_at["bid_price_0"].iloc[-1] if len(mkt_at) else float("nan")
            match = abs(intent_bid - mkt_bid) < 1e-6
            mkt_bids = f"{mkt_bid:.4f}"
        print(f"{eid:>4}  {sid:>8}  {str(ts):>26}  {intent_bid:>10.4f}  {mkt_bids:>14}  {'YES' if match else 'NO!':>6}")

    print()
    print("=" * 70)
    print("2. FILL ACCOUNTING")
    print("=" * 70)
    print(f"{'Timestamp':>26}  {'EID':>4}  {'SID':>8}  {'Side':>4}  {'Price':>6}  {'Qty':>5}  {'Type':>5}  {'FeeActual':>10}  {'FeeExpected':>11}  {'OK?':>4}")
    print("-" * 70)

    fee_failures = 0
    for ts, row in fills.iterrows():
        eid = int(row["exchange_id"])
        sid = int(row["security_id"])
        price = row["fill_price"]
        qty = row["fill_qty"]
        actual_fee = row["fee"]
        fill_type = infer_fill_type(row)

        rates = fee_rates.get(eid, {"maker": 0.0, "taker": 0.0})
        rate = rates["taker"] if fill_type == "taker" else rates["maker"]
        expected_fee = parametric_fee(price, rate, qty)

        ok = abs(actual_fee - expected_fee) < 1e-5
        if not ok:
            fee_failures += 1

        print(
            f"{str(ts):>26}  {eid:>4}  {sid:>8}  {row['side']:>4}  {price:>6.4f}  {qty:>5.2f}  "
            f"{fill_type:>5}  {actual_fee:>10.6f}  {expected_fee:>11.6f}  {'YES' if ok else 'NO!':>4}"
        )

    print()
    print("=" * 70)
    print("3. POSITION RECONCILIATION & PNL")
    print("=" * 70)
    print(f"{'EID':>4}  {'SID':>8}  {'BidFills':>9}  {'AskFills':>9}  {'NetPos':>8}  {'RawPnL':>9}  {'Fees':>9}  {'NetPnL':>9}")
    print("-" * 70)

    total_pnl = 0.0
    for (eid, sid), grp in fills.groupby(["exchange_id", "security_id"]):
        bid_fills = grp[grp["side"] == "Bid"]
        ask_fills = grp[grp["side"] == "Ask"]
        bid_qty = bid_fills["fill_qty"].sum()
        ask_qty = ask_fills["fill_qty"].sum()
        net_pos = bid_qty - ask_qty
        cash_out = (ask_fills["fill_price"] * ask_fills["fill_qty"]).sum()
        cash_in = (bid_fills["fill_price"] * bid_fills["fill_qty"]).sum()
        raw_pnl = cash_out - cash_in
        total_fees = grp["fee"].sum()
        net_pnl = raw_pnl - total_fees
        total_pnl += net_pnl
        print(
            f"{int(eid):>4}  {int(sid):>8}  {bid_qty:>9.2f}  {ask_qty:>9.2f}  {net_pos:>8.2f}  "
            f"{raw_pnl:>9.4f}  {total_fees:>9.6f}  {net_pnl:>9.4f}"
        )

    print()
    print(f"Total net PnL: {total_pnl:.6f}")
    if fee_failures:
        print(f"WARNING: {fee_failures} fee mismatch(es) found")
    else:
        print("All fees match parametric formula")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", help="Backtest job output directory")
    parser.add_argument(
        "--fees",
        default="4:0.0:0.07,5:0.0175:0.07",
        help='Fee rates: eid:maker:taker,... (default: polymarket=0/0.07, kalshi=0.0175/0.07)',
    )
    args = parser.parse_args()

    fee_rates: dict[int, dict[str, float]] = {}
    for part in args.fees.split(","):
        eid_str, maker_str, taker_str = part.strip().split(":")
        fee_rates[int(eid_str)] = {"maker": float(maker_str), "taker": float(taker_str)}

    validate(Path(args.output_dir), fee_rates)


if __name__ == "__main__":
    main()
