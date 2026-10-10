"""
Build a run config for CS2PreMatch: one scenario per match, listings as inputs.

For each match with a prediction, find its Polymarket series and map-1 markets,
resolve each team's token to its registry listing, and emit a scenario whose
window covers the series entry window plus a margin after kickoff for fills and
settlement. Matches whose markets cannot be matched unambiguously, or whose
tokens are not in the registry, are left out and counted.

    poetry run python -m gnomepy_research.sessions.cs2_prematch.make_config \\
        --predictions <path> --start 2026-09-01 --end 2026-09-29 --out configs/sep.yaml
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
import yaml

from gnomepy.registry import RegistryClient

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.sessions.cs2_prematch import polymarket, predictions
from gnomepy_research.sessions.cs2_prematch.cs2_prematch import DEFAULT_WINDOWS

logger = logging.getLogger(__name__)

STRATEGY_CLASS = "gnomepy_research.sessions.cs2_prematch.cs2_prematch:CS2PreMatch"
POLYMARKET_PROFILE = {
    "fee_model": {"type": "parametric", "taker_fee_rate": 0.07, "maker_fee_rate": 0.0},
    "network_latency": {"type": "static", "latency_nanos": 50_000_000},
    "order_processing_latency": {"type": "maker_taker", "base_nanos": 5_000_000,
                                 "taker_delay_nanos": 50_000_000, "maker_delay_nanos": 0},
    "queue_model": {"type": "risk_averse"},
}


def build_scenarios(preds: pd.DataFrame, history: pd.DataFrame, pm: pd.DataFrame, resolve,
                    markets: tuple[str, ...] = ("series", "game1"),
                    settle_check: bool = True) -> tuple[dict, dict]:
    """
    (scenarios keyed by name, counts of what was skipped and why).

    `resolve(condition_id, token)` returns the registry listing id or None.
    """
    history = history.assign(match_date=pd.to_datetime(history.match_date))
    first = history.sort_values("map_position_in_series").drop_duplicates("match_id").set_index("match_id")
    series_won = history.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() > len(w) - w.sum()))
    map1_won = history[history.map_position_in_series == 1].set_index("match_id").team_a_won
    have = set(zip(preds.match_id, preds.market))
    lead = max(DEFAULT_WINDOWS[m][0] for m in markets)

    scenarios, skipped = {}, {}
    for match_id in sorted(set(preds.match_id)):
        if match_id not in first.index:
            skipped["not in HLTV history"] = skipped.get("not in HLTV history", 0) + 1
            continue
        h = first.loc[match_id]
        kickoff = pd.Timestamp(h.match_time)
        legs = []
        for market in markets:
            if (match_id, market) not in have:
                continue
            won = (series_won if market == "series" else map1_won).get(match_id) if settle_check else None
            cond, tok_a, tok_b, reason = polymarket.match_market(pm, h.team_a_name, h.team_b_name,
                                                                 h.match_date.date(), market, won)
            listing_a = listing_b = None
            if reason is None:
                listing_a, listing_b = resolve(cond, tok_a), resolve(cond, tok_b)
                if listing_a is None or listing_b is None:
                    reason = "token not in registry"
            if reason:
                skipped[f"{market}: {reason}"] = skipped.get(f"{market}: {reason}", 0) + 1
                continue
            legs.append({"match_id": int(match_id), "market": market,
                         "listing_team_a": listing_a, "listing_team_b": listing_b})
        if not legs:
            continue
        scenarios[f"m{match_id}"] = {
            "start_date": (kickoff - pd.Timedelta(hours=lead, minutes=15)).tz_localize(None).isoformat(),
            "end_date": (kickoff + pd.Timedelta(minutes=30)).tz_localize(None).isoformat(),
            "listings": [{"listing_id": leg[k], "profile": "polymarket"}
                         for leg in legs for k in ("listing_team_a", "listing_team_b")],
            "strategy_args": {"markets": legs},
        }
    return scenarios, skipped


def build_config(scenarios: dict, predictions_path: str, strategy_args: dict | None = None) -> dict:
    return {
        "strategy": {"class_name": STRATEGY_CLASS,
                     "args": {"predictions_path": predictions_path, "markets": [], "reload_minutes": 0,
                              **(strategy_args or {})}},
        "record_depth": 10,
        "profiles": {"polymarket": POLYMARKET_PROFILE},
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="parquet path or DatasetStore name")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-settle-check", action="store_true", help="for upcoming matches with unsettled markets")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    preds = predictions.load(args.predictions)
    history = DatasetStore().load("cs2_match_history")
    history = history[pd.to_datetime(history.match_date).between(start, end)]
    preds = preds[preds.match_id.isin(history.match_id)]
    pm = polymarket.fetch_markets(start - pd.Timedelta(days=10), end + pd.Timedelta(days=2))
    registry = RegistryClient()
    scenarios, skipped = build_scenarios(preds, history, pm,
                                         lambda cond, tok: polymarket.resolve_listing(cond, tok, registry),
                                         settle_check=not args.no_settle_check)
    with open(args.out, "w") as f:
        yaml.safe_dump(build_config(scenarios, args.predictions), f, sort_keys=False)
    logger.info("wrote %d scenarios to %s; skipped: %s", len(scenarios), args.out, skipped)
