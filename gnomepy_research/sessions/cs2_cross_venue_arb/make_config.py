"""
Write a multi-scenario backtest config for CS2CrossVenueArb, one scenario per market, from a hand-made market map and
PandaScore's match times.

Each scenario starts when PandaScore says the series (or the market's map) began, plus a feed delay, as the launcher
will start a live session; markets PandaScore shows cancelled, forfeited or never begun are left out. It ends at the
last minute any of its streams collected.

    python -m gnomepy_research.sessions.cs2_cross_venue_arb.make_config MARKETS PS_MAP OUT.yaml \
        [--from 2026-11-05] [--to 2026-11-08] [--limit N] [--arg key=value ...] [--taker-delay-ms VENUE=MS ...]
        [--strategy module:Class] [--scope match] [--require-venue polymarket] [--record-depth 10]
        [--sweep key=[v1, v2] ...]

--taker-delay-ms overrides an exchange's simulated taker delay (Polymarket International holds marketable orders on
these markets for its secondsDelay, 1 s); unset, the registry's simulation parameters apply. --scope keeps markets of
one scope (match, or map for every map); --require-venue keeps markets that venue lists, for strategies that need it.

MARKETS is a JSON list of {teams: [a, b], scope, start, legs: [{venue, side, listing, s3last}]} (side is the team
name); PS_MAP maps "a|b|start" to {status, forfeit, begin, games: {"1": {begin, forfeit}}} with times in ns.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import yaml

STRATEGY = "gnomepy_research.sessions.cs2_cross_venue_arb.strategy:CS2CrossVenueArb"
VENUES = {"POLYMARKET_INTL": "polymarket", "KALSHI": "kalshi", "POLYMARKET_US": "polymarket_us"}


def begin_ns(market: dict, ps: dict) -> int | None:
    p = ps.get(f"{market['teams'][0]}|{market['teams'][1]}|{market['start']}")
    if p is None or p["status"] == "canceled" or p["forfeit"]:
        return None
    if market["scope"] == "match":
        return p["begin"]
    game = p["games"].get(market["scope"].split(":")[1])
    return game["begin"] if game and not game["forfeit"] else None


def legs_of(market: dict) -> list[dict]:
    a, b = market["teams"]
    by_venue: dict[str, dict] = {}
    for leg in market["legs"]:
        by_venue.setdefault(leg["venue"], {})[leg["side"]] = leg["listing"]
    return [{"venue": VENUES[v], "listing_team_a": d[a], "listing_team_b": d[b]}
            for v, d in sorted(by_venue.items()) if a in d and b in d and v in VENUES]


def iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def scenario(market: dict, ps: dict, delay_s: float, require_venue: str | None = None) -> tuple[str, dict] | None:
    begin = begin_ns(market, ps)
    legs = legs_of(market)
    if begin is None or len(legs) < 2:
        return None
    if require_venue and require_venue not in {leg["venue"] for leg in legs}:
        return None
    last = max(datetime.strptime(leg["s3last"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
               for leg in market["legs"]) + timedelta(minutes=1)
    start = begin + int(delay_s * 1e9)
    if last.timestamp() * 1e9 <= start:
        return None
    name = f"{'_'.join(market['teams'])}_{market['scope'].replace(':', '')}_{market['start'][:10]}"
    listings = sorted({lid for leg in legs for lid in (leg["listing_team_a"], leg["listing_team_b"])})
    return name, {
        "start_date": iso(start),
        "end_date": last.strftime("%Y-%m-%dT%H:%M:%S"),
        "listings": [{"listing_id": lid} for lid in listings],
        "strategy_args": {"legs": legs},
    }


def parse_arg(text: str) -> tuple[str, object]:
    key, value = text.split("=", 1)
    return key, yaml.safe_load(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("markets")
    p.add_argument("ps_map")
    p.add_argument("out")
    p.add_argument("--from", dest="day_from")
    p.add_argument("--to", dest="day_to")
    p.add_argument("--limit", type=int)
    p.add_argument("--delay-seconds", type=float, default=30.0)
    p.add_argument("--arg", action="append", default=[], type=parse_arg)
    p.add_argument("--taker-delay-ms", action="append", default=[], type=parse_arg)
    p.add_argument("--strategy", default=STRATEGY)
    p.add_argument("--scope", choices=["match", "map"])
    p.add_argument("--require-venue")
    p.add_argument("--record-depth", type=int, default=10)
    p.add_argument("--sweep", action="append", default=[], type=parse_arg)
    args = p.parse_args()

    markets = json.load(open(args.markets))
    ps = json.load(open(args.ps_map))
    scenarios, skipped = {}, 0
    for m in markets:
        day = m["start"][:10]
        if (args.day_from and day < args.day_from) or (args.day_to and day > args.day_to):
            continue
        if args.scope and m["scope"].split(":")[0] != args.scope:
            continue
        s = scenario(m, ps, args.delay_seconds, args.require_venue)
        if s is None:
            skipped += 1
            continue
        scenarios[s[0]] = s[1]
        if args.limit and len(scenarios) >= args.limit:
            break
    config = {
        "strategy": {"class_name": args.strategy, "args": {"legs": [], **dict(args.arg)}},
        "record_depth": args.record_depth,
        "scenarios": scenarios,
    }
    if args.sweep:
        config["sweep"] = {"strategy": dict(args.sweep)}
    if args.taker_delay_ms:
        config["exchanges"] = {
            venue: {"order_processing_latency": {"type": "maker_taker", "base_nanos": 5_000_000,
                                                 "taker_delay_nanos": int(ms * 1_000_000), "maker_delay_nanos": 0}}
            for venue, ms in args.taker_delay_ms
        }
    with open(args.out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"{len(scenarios)} scenarios, {skipped} markets skipped (not begun per PandaScore, one venue, or without "
          f"the required venue)")


if __name__ == "__main__":
    main()
