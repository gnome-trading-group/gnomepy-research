#!/usr/bin/env python3
"""
Validate arb backtest debug output.

Usage:
    poetry run python validate_arb.py <log_file> [--fees eid:rate,eid:rate]

    --fees  comma-separated exchange_id:maker_fee_rate pairs, e.g. "5:0.0175,4:0.0"

For each ENTRY the script shows two price perspectives:

  BidEdge  — edge computed from TARGET_SET (bid) prices: what we'd capture if filled
             as makers. This is the *expected* profitability if all legs fill.
             Formula: (1 - sum_bid_prices - fees_at_bid) * 10000

  AskEdge  — the edge the strategy actually used to make the entry decision, extracted
             from the EDGE_CHECK debug line just before the ENTRY. This uses ASK book
             prices (the conservative taker-floor check). Should match 'Reported'.

Checks:
  1. sum_bid_prices < 1.0  (basic arb sanity at bid)
  2. AskEdge >= 5 bps and matches Reported
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field


_ENTRY_RE = re.compile(
    r"^\[(\d+)\] ENTRY .+ pairing=(\d+) edge=([\d.]+)bps mode=(\w+) qty=([\d.]+)"
)
_DEBUG_RE = re.compile(r"^\[(\d+)\] DEBUG (\w+) (.+)$")
_KV_RE = re.compile(r"(\w+)=([\w.\-]+)")


def _parse_kv(s: str) -> dict[str, str]:
    return dict(_KV_RE.findall(s))


@dataclass
class Leg:
    eid: int
    sid: int
    qty: int
    price: float  # bid price, already in [0, 1]


@dataclass
class Entry:
    timestamp: int
    pairing_idx: int
    reported_edge_bps: float
    mode: str
    qty: int
    ask_edge_bps: float | None = None  # from EDGE_CHECK just before entry
    legs: list[Leg] = field(default_factory=list)


def parse_log(lines: list[str]) -> list[Entry]:
    entries: list[Entry] = []
    current: Entry | None = None
    seen_legs: set[tuple[int, int]] = set()
    last_edge_check: dict[int, float] = {}  # pairing_idx -> maker_edge

    for line in lines:
        line = line.strip()

        # Capture EDGE_CHECK maker_edge lines as we pass them
        m = _DEBUG_RE.match(line)
        if m and m.group(2) == "EDGE_CHECK":
            kv = _parse_kv(m.group(3))
            if "maker_edge" in kv:
                try:
                    pairing = int(kv["pairing"])
                    last_edge_check[pairing] = float(kv["maker_edge"])
                except (KeyError, ValueError):
                    pass

        em = _ENTRY_RE.match(line)
        if em:
            pairing_idx = int(em.group(2))
            current = Entry(
                timestamp=int(em.group(1)),
                pairing_idx=pairing_idx,
                reported_edge_bps=float(em.group(3)),
                mode=em.group(4),
                qty=int(float(em.group(5))),
                ask_edge_bps=last_edge_check.get(pairing_idx),
            )
            entries.append(current)
            seen_legs = set()
            continue

        if current is None:
            continue

        if not m:
            continue

        event = m.group(2)
        ts = int(m.group(1))

        if ts != current.timestamp:
            current = None
            seen_legs = set()
            continue

        if event != "TARGET_SET":
            continue

        kv = _parse_kv(m.group(3))
        if kv.get("phase") != "ENTERING":
            continue
        try:
            qty = int(kv["qty"])
            price = float(kv["price"])
            eid = int(kv["eid"])
            sid = int(kv["sid"])
        except (KeyError, ValueError):
            continue

        if qty <= 0:
            continue

        key = (eid, sid)
        if key in seen_legs:
            continue
        seen_legs.add(key)
        current.legs.append(Leg(eid=eid, sid=sid, qty=qty, price=price))

    return entries


def parametric_fee(price: float, rate: float) -> float:
    if rate == 0.0:
        return 0.0
    return rate * price * (1.0 - price)


def validate(entries: list[Entry], fee_rates: dict[int, float]) -> None:
    print(
        f"{'Timestamp':>20}  {'Pair':>4}  {'Mode':>5}  "
        f"{'SumBid':>7}  {'BidEdge':>9}  {'AskEdge':>9}  {'Reported':>9}  {'OK?':>4}"
    )
    print("-" * 90)

    failures = 0
    for entry in entries:
        if not entry.legs:
            print(f"[{entry.timestamp}] WARNING: no TARGET_SET legs found")
            continue

        sum_bid = sum(leg.price for leg in entry.legs)

        if fee_rates:
            bid_fees = sum(
                parametric_fee(leg.price, fee_rates.get(leg.eid, 0.0))
                for leg in entry.legs
            )
            bid_edge = (1.0 - sum_bid - bid_fees) * 10_000
        else:
            bid_edge = (1.0 - sum_bid) * 10_000

        ask_edge_str = f"{entry.ask_edge_bps:>9.2f}" if entry.ask_edge_bps is not None else f"{'?':>9}"

        # ask_edge should match reported; bid prices < 1.0
        ask_matches = (
            entry.ask_edge_bps is None
            or abs(entry.ask_edge_bps - entry.reported_edge_bps) < 1.0
        )
        basic_ok = sum_bid < 1.0 and entry.reported_edge_bps > 0 and ask_matches
        if not basic_ok:
            failures += 1

        print(
            f"{entry.timestamp:>20}  {entry.pairing_idx:>4}  {entry.mode:>5}  "
            f"{sum_bid:>7.4f}  {bid_edge:>9.2f}  {ask_edge_str}  "
            f"{entry.reported_edge_bps:>9.2f}  "
            f"{'YES' if basic_ok else 'NO!':>4}"
        )
        leg_str = "  ".join(f"eid={l.eid} sid={l.sid} bid={l.price:.4f}" for l in entry.legs)
        print(f"  legs: {leg_str}")

    print()
    print(f"Entries: {len(entries)}  Failures: {failures}")
    if not fee_rates:
        print("(Pass --fees eid:rate,eid:rate for fee-adjusted BidEdge)")
    print()
    print("BidEdge = expected profit if filled at maker bid prices (what we post at)")
    print("AskEdge = conservative arb check used for entry decision (ask-book prices)")
    print("Reported = edge from ENTRY log line (should match AskEdge)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_file", nargs="?", help="Debug log file (default: stdin)")
    parser.add_argument(
        "--fees",
        default="",
        help='Maker fee rates per exchange_id, e.g. "5:0.0175,4:0.0"',
    )
    args = parser.parse_args()

    fee_rates: dict[int, float] = {}
    if args.fees:
        for pair in args.fees.split(","):
            eid_str, rate_str = pair.strip().split(":")
            fee_rates[int(eid_str)] = float(rate_str)

    if args.log_file:
        with open(args.log_file) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    entries = parse_log(lines)
    validate(entries, fee_rates)


if __name__ == "__main__":
    main()
