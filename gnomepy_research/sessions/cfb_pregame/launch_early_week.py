"""
Build, check and (with --send) queue the direct_launch triggers that start the CFB early-week paper sessions
(early_week.py) on Kalshi.

    AWS_PROFILE=AWSAdministratorAccess-241533121172 REGISTRY_API_KEY_ID=<CloudFormation export RegistryApiKeyId> \\
    poetry run python -m gnomepy_research.sessions.cfb_pregame.launch_early_week \\
        --slate ~/cfb-live-data/<week>/week<N>_e11_e15.csv --games ~/cfb-live-data/<week>/preds_C2.parquet \\
        --model e11 --strategy <id> --research-commit "$(git rev-parse HEAD)" [--start ISO --end ISO] [--send]

The slate is the weekly model file (game_id and one column per model, from the live overlay pipeline), with kickoffs
and CFBD team names from the week's C2 predictions. The model column chosen is frozen into each session's trigger as
the game's p_home. Kalshi's markets are matched, oriented, resolved and cross-checked as launch.py does. A game with
fewer than 66 hours to kickoff at launch is left out: its 72-hour decision time is more than the 6 hours late the
strategy allows. With --include-late such a game is traded anyway, marked off_protocol (the strategy decides it at its
first look and flags its decision row), as long as more than 7 hours remain (entries close 6 hours out). Sessions hold
at most --max-games games each (the registry's heartbeat limit).

Without --send nothing leaves this machine but reads. With --send each trigger goes onto the launcher's launch queue.
Paper mode only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
from gnomepy.registry import RegistryClient

from gnomepy_research.sessions.cfb_pregame import launch
from gnomepy_research.sessions.cfb_pregame.slate import kalshi_legs

STRATEGY_CLASS = "gnomepy_research.sessions.cfb_pregame.early_week:CFBEarlyWeek"
DECIDE = pd.Timedelta(hours=72)
MAX_LATE = pd.Timedelta(hours=6)
ENTRY_CLOSE = pd.Timedelta(hours=6)
MIN_ENTRY = pd.Timedelta(hours=1)
MAX_GAMES = 15


def load_slate(slate_csv: str | Path, games_parquet: str | Path, model: str) -> pd.DataFrame:
    """game_id, kickoff, home_team, away_team and p_home (the model's column) for each game the slate prices."""
    slate = pd.read_csv(slate_csv)
    if model not in slate.columns:
        raise SystemExit(f"{slate_csv} has no column {model!r}; it has {list(slate.columns)}")
    games = pd.read_parquet(games_parquet)
    if "generated_at" in games.columns:
        games = games.sort_values("generated_at")
    games = games.drop_duplicates("game_id", keep="last")[["game_id", "kickoff", "home_team", "away_team"]]
    out = slate[["game_id", model]].rename(columns={model: "p_home"}).merge(games, on="game_id")
    out["kickoff"] = pd.to_datetime(out["kickoff"], utc=True)
    return out[["game_id", "kickoff", "home_team", "away_team", "p_home"]].dropna().reset_index(drop=True)


def triggers(legs: pd.DataFrame, slate: pd.DataFrame, *, strategy_id: int, research_commit: str, model: str,
             now: pd.Timestamp, max_games: int, args: dict,
             include_late: bool = False) -> tuple[list[dict], list[launch.Drop]]:
    """The direct_launch triggers for the legs' games, in kickoff order and at most max_games to a session, and the
    games left out: those whose decision window has closed, or with include_late (which trades those off_protocol)
    those with too little of their entry window left."""
    s = slate.set_index("game_id")
    rows = legs.assign(kickoff=lambda d: d["game_id"].map(s["kickoff"]), p_home=lambda d: d["game_id"].map(s["p_home"]))
    closed = rows["kickoff"] - DECIDE + MAX_LATE < now
    if include_late:
        out_of_time = rows["kickoff"] - ENTRY_CLOSE - MIN_ENTRY < now
        drops = [(int(g), "kalshi", "entry window nearly closed") for g in rows.loc[out_of_time, "game_id"]]
        rows = rows[~out_of_time].assign(off_protocol=closed[~out_of_time])
    else:
        drops = [(int(g), "kalshi", "decision window closed") for g in rows.loc[closed, "game_id"]]
        rows = rows[~closed].assign(off_protocol=False)
    rows = rows.sort_values(["kickoff", "game_id"])
    every = [{"game_id": int(r.game_id), "listing_home": int(r.listing_home), "listing_away": int(r.listing_away),
              "kickoff": pd.Timestamp(r.kickoff).isoformat(), "p_home": float(r.p_home),
              "off_protocol": bool(r.off_protocol)} for r in rows.itertuples()]
    out = []
    for start in range(0, len(every), max_games):
        games = every[start:start + max_games]
        deadline = (max(pd.Timestamp(g["kickoff"]) for g in games) - ENTRY_CLOSE).isoformat()
        out.append({"venue": "kalshi", "variant": "early_week", "deadline": deadline, "data": {
            "strategy_id": int(strategy_id), "strategy_type": "python", "strategy_class": STRATEGY_CLASS,
            "mode": "paper", "research_commit": research_commit, "instance_type": launch.INSTANCE,
            "listings": [lid for g in games for lid in (g["listing_home"], g["listing_away"])],
            "strategy_args": {"games": games, "model": model, **args},
        }})
    return out, drops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slate", required=True, help="the weekly model file (game_id plus one column per model)")
    ap.add_argument("--games", required=True, help="the week's C2 predictions parquet (kickoffs, CFBD team names)")
    ap.add_argument("--model", default="e11", help="the slate column to trade on (e11, or e11+e15 for the overlay)")
    ap.add_argument("--strategy", type=int, required=True, help="the registry strategy id")
    ap.add_argument("--research-commit", required=True)
    ap.add_argument("--start", help="only games kicking off at or after this time")
    ap.add_argument("--end", help="only games kicking off before this time")
    ap.add_argument("--contracts", type=float, default=1.0)
    ap.add_argument("--min-gap", type=float, default=0.05)
    ap.add_argument("--max-games", type=int, default=MAX_GAMES)
    ap.add_argument("--include-late", action="store_true",
                    help="trade games past their decision window too, flagged off_protocol")
    ap.add_argument("--send", action="store_true", help="queue the triggers on the launcher's launch queue")
    a = ap.parse_args()
    now = pd.Timestamp(datetime.now(timezone.utc))
    slate = load_slate(Path(a.slate).expanduser(), Path(a.games).expanduser(), a.model)
    if a.start:
        slate = slate[slate["kickoff"] >= pd.Timestamp(a.start)]
    if a.end:
        slate = slate[slate["kickoff"] < pd.Timestamp(a.end)]
    legs = kalshi_legs(slate[["game_id", "kickoff", "home_team", "away_team"]], launch.kalshi_markets())
    unmatched = [(int(g), "kalshi", "no Kalshi market matched") for g in slate["game_id"]
                 if not (legs["game_id"] == g).any()]
    resolved, missing = launch.resolve(legs, RegistryClient())
    checked, flagged = launch.cross_check(resolved, launch.leg_mids(resolved), slate)
    out, closed = triggers(checked, slate, strategy_id=a.strategy, research_commit=a.research_commit, model=a.model,
                           now=now, max_games=a.max_games, args={"contracts": a.contracts, "min_gap": a.min_gap},
                           include_late=a.include_late)

    names = slate.set_index("game_id")
    print(f"{len(slate)} games priced by {a.model}, {sum(len(t['data']['strategy_args']['games']) for t in out)} "
          f"in {len(out)} sessions")
    print("left out:")
    for game_id, venue, reason in sorted(unmatched + missing + flagged + closed):
        g = names.loc[game_id]
        print(f"  {game_id} {g.away_team} at {g.home_team}: {reason}")
    late = [g for t in out for g in t["data"]["strategy_args"]["games"] if g["off_protocol"]]
    if late:
        print(f"OFF PROTOCOL: {len(late)} games decided after their 72-hour window (reported apart):")
        for g in late:
            n = names.loc[g["game_id"]]
            print(f"  {g['game_id']} {n.away_team} at {n.home_team}, kickoff {g['kickoff']}")
    for t in out:
        print(f"trigger: {len(t['data']['strategy_args']['games'])} games, deadline {t['deadline']}")
    print(json.dumps([{"rule_type": launch.RULE_TYPE, "action": "launch", "deadline": t["deadline"], "data": t["data"]}
                      for t in out], indent=1))
    if a.send:
        sqs = boto3.client("sqs", region_name="us-east-1")
        print("queued:", launch.send(out, sqs, launch.queue_url(sqs), now))


if __name__ == "__main__":
    main()
