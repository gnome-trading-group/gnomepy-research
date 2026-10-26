"""Synthetic series books for both teams of one CS2 match, identity-free, for the old and new cs2_series replays.

Team A's mid sweeps 0.22 -> 0.88 -> 0.22 over the entry window so the model finds each side cheap at some point;
team B's book mirrors it with a 1c overround. Three ask and bid levels deepen away from the touch.
"""
import json
import math
from datetime import datetime, timedelta, timezone

PRICE = 1_000_000_000
SIZE = 1_000_000
TICK = 0.01

MATCH_ID = 2399331
KICKOFF = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)
START = KICKOFF - timedelta(hours=12)
END = KICKOFF - timedelta(hours=3)
STEP = timedelta(minutes=2)


def book(mid: float) -> dict:
    bid0 = round(mid - TICK, 2)
    ask0 = round(mid + TICK, 2)
    levels = {}
    for i in range(3):
        levels[f"bid_price_{i}"] = int(round((bid0 - i * TICK) * PRICE))
        levels[f"ask_price_{i}"] = int(round((ask0 + i * TICK) * PRICE))
        levels[f"bid_size_{i}"] = (50 * (i + 1)) * SIZE
        levels[f"ask_size_{i}"] = (50 * (i + 1)) * SIZE
    return levels


def main() -> None:
    records = {"a": [], "b": []}
    steps = int((END - START) / STEP)
    sequence = 0
    for k in range(steps):
        at = START + k * STEP + timedelta(seconds=1)
        ns = int(at.timestamp() * 1e9)
        mid_a = round(0.55 - 0.33 * math.cos(2 * math.pi * k / steps), 2)
        mid_b = round(1.01 - mid_a, 2)
        for team, mid in (("a", mid_a), ("b", mid_b)):
            sequence += 1
            records[team].append({
                "minute": at.strftime("%Y-%m-%dT%H:%M"),
                "fields": {"timestamp_event": ns, "timestamp_recv": ns, "sequence": sequence, "action": "Add",
                           "side": "Bid", "price": book(mid)["bid_price_0"], "size": 50 * SIZE, "depth": 0,
                           **book(mid)},
            })
    spec = {"match_id": MATCH_ID, "start": START.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": END.strftime("%Y-%m-%dT%H:%M:%S"), "records": records}
    with open("cs2_spec.json", "w") as f:
        json.dump(spec, f)
    print("records per team", len(records["a"]))


if __name__ == "__main__":
    main()
