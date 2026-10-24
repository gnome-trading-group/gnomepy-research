"""
Build, check and (with --send) queue the direct_launch triggers that start the CFB pregame paper sessions.

    AWS_PROFILE=AWSAdministratorAccess-241533121172 REGISTRY_API_KEY_ID=<CloudFormation export RegistryApiKeyId> \\
    poetry run python -m gnomepy_research.sessions.cfb_pregame.launch --start 2026-10-24T14:30Z --end 2026-10-25T06:00Z \\
        --anchor-strategy 6 --raw-strategy 7 --research-commit <sha> [--predictions FILE.parquet] [--send]

One session per variant (anchor, raw) per venue (Kalshi, Polymarket US), each holding every game of the slate it
can trade. The slate is the newest prediction per game in cfb_pregame_predictions (or a parquet file of the same
columns) kicking off in [start, end). Each game's venue market is matched and oriented (slate.py), its two listings
resolved in the registry, and its legs cross-checked against the books: venues that disagree on the home team's
price by more than 0.10, or a market more than 0.35 from the model, mean a likely flip, and the game or leg is left
out. Games whose entry cutoff is under 15 minutes away are left out too.

Without --send nothing leaves this machine but reads: it prints the slate and the triggers. With --send each trigger
goes onto the launcher's FIFO launch queue as gnome-launcher's own scanners send them (launcher/queues.py), for its
direct_launch rule to launch. Paper mode only.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone

import boto3
import pandas as pd
import requests
from gnomepy.data import DatasetStore
from gnomepy.registry import RegistryClient

from gnomepy_research.sessions.cfb_pregame.prediction_feed import PREDICTIONS
from gnomepy_research.sessions.cfb_pregame.slate import kalshi_legs, pmus_legs

STRATEGY_CLASS = "gnomepy_research.sessions.cfb_pregame.strategy:CFBPregame"
RULE_TYPE = "direct_launch"
VARIANTS = {"anchor": {"anchor_weight": 0.25, "min_edge": 0.01}, "raw": {"anchor_weight": 1.0, "min_edge": 0.03},
            "anchor_tp": {"anchor_weight": 0.25, "min_edge": 0.01, "take_profit": True},
            "raw_tp": {"anchor_weight": 1.0, "min_edge": 0.03, "take_profit": True}}
EXCHANGES = {"kalshi": "Kalshi", "polymarket_us": "Polymarket (US)"}
CUTOFF = pd.Timedelta(minutes=30)
MIN_LEAD = pd.Timedelta(minutes=15)
MAX_VENUE_GAP = 0.10
MAX_MODEL_GAP = 0.35
INSTANCE = "t3.small"
QUEUE_PREFIX = "Prod-LauncherStack-LauncherFifoQueue"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
PMUS = "https://gateway.polymarket.us"
TIMEOUT = 30
Drop = tuple[int, str, str]


def resolve(legs: pd.DataFrame, registry) -> tuple[pd.DataFrame, list[Drop]]:
    """Each leg with its registry listing_home and listing_away; legs missing either are dropped and reported."""
    ids = {e.exchange_name: int(e.exchange_id) for e in registry.get_exchange()}
    kept, drops = [], []
    for r in legs.itertuples(index=False):
        exchange = ids[EXCHANGES[r.venue]]
        found = {}
        for side, esid in (("listing_home", r.esid_home), ("listing_away", r.esid_away)):
            hits = registry.get_listing(exchange_id=exchange, exchange_security_id=esid)
            if not hits:
                drops.append((int(r.game_id), r.venue, f"{esid} not in the registry"))
                break
            found[side] = int(hits[0].listing_id)
        else:
            kept.append({**r._asdict(), **found})
    return pd.DataFrame(kept, columns=[*legs.columns, "listing_home", "listing_away"]), drops


def cross_check(legs: pd.DataFrame, mids: dict[tuple[int, str], float | None],
                preds: pd.DataFrame) -> tuple[pd.DataFrame, list[Drop]]:
    """Legs whose market price of the home team is consistent across venues and with the model."""
    p_home = preds.set_index("game_id")["p_home"]
    drops: list[Drop] = []
    keep = []
    for r in legs.itertuples(index=False):
        if mids.get((r.game_id, r.venue)) is None:
            drops.append((int(r.game_id), r.venue, "no two-sided book"))
        else:
            keep.append(r)
    by_game: dict[int, list] = {}
    for r in keep:
        by_game.setdefault(r.game_id, []).append(r)
    out = []
    for game_id, rows in by_game.items():
        prices = {r.venue: mids[(game_id, r.venue)] for r in rows}
        if len(prices) > 1 and max(prices.values()) - min(prices.values()) > MAX_VENUE_GAP:
            text = ", ".join(f"{v} {m:.2f}" for v, m in prices.items())
            drops += [(int(game_id), r.venue, f"venues disagree: {text}") for r in rows]
            continue
        for r in rows:
            p, m = float(p_home[game_id]), prices[r.venue]
            if abs(p - m) > MAX_MODEL_GAP:
                drops.append((int(game_id), r.venue, f"model {p:.2f} vs market {m:.2f}: likely flipped"))
            else:
                out.append(r)
    return pd.DataFrame(out, columns=legs.columns), drops


def triggers(resolved: pd.DataFrame, preds: pd.DataFrame, *, strategy_ids: dict[str, int], research_commit: str,
             bankroll_usd: float, now: pd.Timestamp, cutoff: pd.Timedelta = CUTOFF) -> list[dict]:
    """One direct_launch trigger per variant in strategy_ids per venue: {venue, variant, deadline, data}; entries close
    `cutoff` before each kickoff."""
    kickoffs = preds.set_index("game_id")["kickoff"].map(pd.Timestamp)
    out = []
    for venue in dict.fromkeys(resolved["venue"]):
        rows = resolved[resolved["venue"] == venue].assign(kickoff=lambda d: d["game_id"].map(kickoffs))
        rows = rows[rows["kickoff"] - cutoff - now >= MIN_LEAD].sort_values(["kickoff", "game_id"])
        if rows.empty:
            continue
        games = [{"game_id": int(r.game_id), "listing_home": int(r.listing_home), "listing_away": int(r.listing_away),
                  "kickoff": pd.Timestamp(r.kickoff).isoformat()} for r in rows.itertuples()]
        listings = [lid for g in games for lid in (g["listing_home"], g["listing_away"])]
        deadline = (rows["kickoff"].max() - cutoff).isoformat()
        for variant in strategy_ids:
            args = VARIANTS[variant]
            out.append({"venue": venue, "variant": variant, "deadline": deadline, "data": {
                "strategy_id": int(strategy_ids[variant]), "strategy_type": "python", "strategy_class": STRATEGY_CLASS,
                "mode": "paper", "research_commit": research_commit, "instance_type": INSTANCE, "listings": listings,
                "strategy_args": {"games": games, "bankroll_usd": float(bankroll_usd),
                                  "entry_cutoff_minutes": cutoff.total_seconds() / 60, **args},
            }})
    return out


def send(found: list[dict], sqs, queue_url: str, now: pd.Timestamp) -> list[str]:
    """Queues each trigger as launcher/queues.send_trigger does; returns the trigger ids."""
    ids = []
    for t in found:
        trigger_id = str(uuid.uuid4())
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps({
            "trigger_id": trigger_id, "rule_type": RULE_TYPE, "action": "launch", "timestamp": now.isoformat(),
            "data": t["data"], "deadline": t["deadline"],
        }), MessageGroupId=RULE_TYPE, MessageDeduplicationId=trigger_id)
        ids.append(trigger_id)
    return ids


# ---- venues -------------------------------------------------------------------------------

def _get(url: str, **params) -> dict:
    for attempt in range(4):
        resp = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 429 and resp.status_code < 500:
            break
        time.sleep(2 ** attempt)
    resp.raise_for_status()
    return resp.json()


def kalshi_markets() -> list[dict]:
    out, cursor = [], None
    while True:
        body = _get(f"{KALSHI}/markets", series_ticker="KXNCAAFGAME", status="open", limit=1000,
                    **({"cursor": cursor} if cursor else {}))
        out += body.get("markets", [])
        cursor = body.get("cursor")
        if not cursor or not body.get("markets"):
            return out


def pmus_events() -> list[dict]:
    out, offset = [], 0
    while True:
        page = _get(f"{PMUS}/v2/leagues/cfb/events", limit=100, offset=offset).get("events", [])
        out += page
        if len(page) < 100:
            return out
        offset += 100


def kalshi_mid(body: dict) -> float | None:
    """A team market's mid from its yes and no bids (a no bid at p is a yes offer at 1 - p)."""
    book = body.get("orderbook_fp") or {}
    yes = [float(p) for p, _ in book.get("yes_dollars") or []]
    no = [float(p) for p, _ in book.get("no_dollars") or []]
    if not yes or not no:
        return None
    return (max(yes) + 1 - max(no)) / 2


def pmus_mid(body: dict) -> float | None:
    """A market's mid in long terms."""
    data = body.get("marketData") or {}
    bids = [float(lv["px"]["value"]) for lv in data.get("bids") or []]
    offers = [float(lv["px"]["value"]) for lv in data.get("offers") or []]
    if not bids or not offers:
        return None
    return (max(bids) + min(offers)) / 2


def leg_mids(legs: pd.DataFrame) -> dict[tuple[int, str], float | None]:
    """The market's P(home) per leg: Kalshi's twin-combined team mids, Polymarket US's long mid in home terms (its
    public book is cached for 30 s, which a sanity check can live with)."""
    out = {}
    for r in legs.itertuples(index=False):
        if r.venue == "kalshi":
            h = kalshi_mid(_get(f"{KALSHI}/markets/{r.esid_home}/orderbook"))
            a = kalshi_mid(_get(f"{KALSHI}/markets/{r.esid_away}/orderbook"))
            out[(r.game_id, r.venue)] = None if h is None or a is None else (h + 1 - a) / 2
        else:
            slug = r.esid_home.rpartition(":")[0]
            m = pmus_mid(_get(f"{PMUS}/v1/markets/{slug}/book"))
            out[(r.game_id, r.venue)] = None if m is None else (m if r.esid_home.endswith(":long") else 1 - m)
    return out


def queue_url(sqs) -> str:
    urls = sqs.list_queues(QueueNamePrefix=QUEUE_PREFIX).get("QueueUrls", [])
    if len(urls) != 1:
        raise SystemExit(f"expected one launcher launch queue named {QUEUE_PREFIX}*, found {urls}")
    return urls[0]


def load_predictions(source: str | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = pd.read_parquet(source) if source else DatasetStore().load(PREDICTIONS)
    rows = rows.assign(kickoff=pd.to_datetime(rows["kickoff"], utc=True))
    rows = rows.sort_values("generated_at").drop_duplicates("game_id", keep="last")
    return rows[(rows["kickoff"] >= start) & (rows["kickoff"] < end)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--anchor-strategy", type=int, required=True)
    ap.add_argument("--raw-strategy", type=int, required=True)
    ap.add_argument("--anchor-tp-strategy", type=int, help="the anchored entries with take-profit exits")
    ap.add_argument("--raw-tp-strategy", type=int, help="the raw entries with take-profit exits")
    ap.add_argument("--research-commit", required=True)
    ap.add_argument("--venues", nargs="+", default=list(EXCHANGES), choices=list(EXCHANGES))
    ap.add_argument("--bankroll", type=float, default=10_000.0)
    ap.add_argument("--cutoff-minutes", type=float, default=CUTOFF.total_seconds() / 60,
                    help="entries close this long before each kickoff (0 = trade up to kickoff)")
    ap.add_argument("--predictions", help="a parquet file of cfb_pregame_predictions rows instead of the table")
    ap.add_argument("--send", action="store_true", help="queue the triggers on the launcher's launch queue")
    a = ap.parse_args()
    now = pd.Timestamp(datetime.now(timezone.utc))
    preds = load_predictions(a.predictions, pd.Timestamp(a.start), pd.Timestamp(a.end))
    games = preds[["game_id", "kickoff", "home_team", "away_team"]]
    found = []
    if "kalshi" in a.venues:
        found.append(kalshi_legs(games, kalshi_markets()))
    if "polymarket_us" in a.venues:
        found.append(pmus_legs(games, pmus_events()))
    legs = pd.concat(found, ignore_index=True)
    unmatched = [(int(g), v, "no venue market matched") for g in games["game_id"] for v in a.venues
                 if not ((legs["game_id"] == g) & (legs["venue"] == v)).any()]
    resolved, missing = resolve(legs, RegistryClient())
    checked, flagged = cross_check(resolved, leg_mids(resolved), preds)
    ids = {"anchor": a.anchor_strategy, "raw": a.raw_strategy, "anchor_tp": a.anchor_tp_strategy, "raw_tp": a.raw_tp_strategy}
    out = triggers(checked, preds, strategy_ids={k: v for k, v in ids.items() if v is not None},
                   research_commit=a.research_commit, bankroll_usd=a.bankroll, now=now,
                   cutoff=pd.Timedelta(minutes=a.cutoff_minutes))

    names = preds.set_index("game_id")
    print(f"{len(preds)} predicted games, {len(checked)} legs kept on {checked['game_id'].nunique()} games")
    for venue in a.venues:
        rows = checked[checked["venue"] == venue]
        print(f"  {venue}: {len(rows)} games")
    print("left out:")
    for game_id, venue, reason in sorted(unmatched + missing + flagged):
        g = names.loc[game_id]
        print(f"  {game_id} {g.away_team} at {g.home_team} [{venue}]: {reason}")
    for t in out:
        n = len(t["data"]["strategy_args"]["games"])
        print(f"trigger {t['variant']}/{t['venue']}: {n} games, deadline {t['deadline']}")
    print(json.dumps([{"rule_type": RULE_TYPE, "action": "launch", "deadline": t["deadline"], "data": t["data"]}
                      for t in out], indent=1))
    if a.send:
        sqs = boto3.client("sqs", region_name="us-east-1")
        print("queued:", send(out, sqs, queue_url(sqs), now))


if __name__ == "__main__":
    main()
