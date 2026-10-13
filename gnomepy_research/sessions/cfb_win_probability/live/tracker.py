"""Live in-game tracker: Polymarket moneyline prices alongside ESPN game state.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.live.tracker --interval 3

Every interval, for each college football game in progress: Polymarket best bid/ask/mid for the
first outcome's token, and ESPN's score, period, clock, possession and field position. Rows are
appended to results/live/<date>.csv, which is the replay dataset for the in-game model.

Latency (measured 2026-10-10 on Texas-Oklahoma): ESPN plays appear ~40 s (32-45 s) after
ESPN's own play wallclock. Polymarket books are real time.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com/events"
BOOK = "https://clob.polymarket.com/book"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
CFB_SERIES = 12756
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "live"
FIELDS = ["ts", "slug", "espn_id", "team0", "team1", "bid0", "ask0", "mid0", "bid_size0", "ask_size0",
          "home", "away", "home_score", "away_score", "period", "clock", "possession", "yard_line", "down_distance",
          "status", "last_play"]


def session() -> requests.Session:
    s = requests.Session()
    # Gamma rejects default library user agents.
    s.headers["User-Agent"] = "Mozilla/5.0"
    return s


def live_espn_games(s: requests.Session) -> dict[str, dict]:
    """ESPN event id -> {home, away} for every FBS game currently in progress."""
    d = s.get(ESPN_SCOREBOARD, params={"groups": 80, "limit": 300}, timeout=15).json()
    out = {}
    for e in d.get("events", []):
        comp = e["competitions"][0]
        if comp["status"]["type"]["state"] != "in":
            continue
        teams = {c["homeAway"]: c["team"]["location"] for c in comp["competitors"]}
        out[e["id"]] = {"home": teams.get("home"), "away": teams.get("away")}
    return out


def polymarket_moneylines(s: requests.Session, day: pd.Timestamp) -> list[dict]:
    lo, hi = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (day + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    events = s.get(GAMMA, params={"series_id": CFB_SERIES, "closed": "false", "limit": 500,
                                  "end_date_min": f"{lo}T00:00:00Z", "end_date_max": f"{hi}T00:00:00Z"}, timeout=30).json()
    rows = []
    for e in events:
        for m in e.get("markets", []):
            if m.get("sportsMarketType") == "moneyline":
                rows.append({"slug": m["slug"], "team0": json.loads(m["outcomes"])[0],
                             "team1": json.loads(m["outcomes"])[1], "token0": json.loads(m["clobTokenIds"])[0],
                             "kickoff": pd.Timestamp(m["gameStartTime"])})
    return rows


def _norm(x: str | None) -> str:
    return (x or "").lower().replace("state", "st").replace(".", "").replace("é", "e").strip()


def pair(markets: list[dict], games: dict[str, dict]) -> list[dict]:
    """Attach each live ESPN game to its Polymarket moneyline by team names."""
    out = []
    for gid, g in games.items():
        names = {_norm(g["home"]), _norm(g["away"])}
        for m in markets:
            if {_norm(m["team0"]), _norm(m["team1"])} == names:
                out.append({**m, "espn_id": gid})
                break
    return out


def book_top(s: requests.Session, token: str) -> dict:
    b = s.get(BOOK, params={"token_id": token}, timeout=10).json()
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
    asks = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
    bid, bsz = bids[0] if bids else (None, None)
    ask, asz = asks[0] if asks else (None, None)
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    return {"bid0": bid, "ask0": ask, "mid0": mid, "bid_size0": bsz, "ask_size0": asz}


def espn_state(s: requests.Session, gid: str) -> dict:
    d = s.get(ESPN, params={"event": gid}, timeout=10).json()
    comp = d["header"]["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    sit = d.get("situation") or {}
    plays = (d.get("drives", {}).get("current") or {}).get("plays") or []
    return {
        "home": teams["home"]["team"]["location"], "away": teams["away"]["team"]["location"],
        "home_score": teams["home"].get("score"), "away_score": teams["away"].get("score"),
        "period": comp["status"].get("period"), "clock": comp["status"].get("displayClock"),
        "possession": sit.get("possession"), "yard_line": sit.get("yardLine"),
        "down_distance": sit.get("shortDownDistanceText"), "status": comp["status"]["type"].get("detail"),
        "last_play": (plays[-1].get("text") or "")[:120] if plays else "",
    }


def run(interval: float) -> None:
    s = session()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        markets, tracked, refreshed = [], [], 0.0
        while True:
            if time.time() - refreshed > 300:
                markets = polymarket_moneylines(s, pd.Timestamp.now(tz="UTC"))
                tracked = pair(markets, live_espn_games(s))
                refreshed = time.time()
                print(f"{pd.Timestamp.now(tz='UTC'):%H:%M:%S} tracking {len(tracked)} live games", flush=True)
            for m in tracked:
                try:
                    row = {"ts": pd.Timestamp.now(tz="UTC").isoformat(), "slug": m["slug"], "espn_id": m["espn_id"],
                           "team0": m["team0"], "team1": m["team1"], **book_top(s, m["token0"]), **espn_state(s, m["espn_id"])}
                    w.writerow(row)
                except (requests.RequestException, KeyError, ValueError) as e:
                    print(f"{m['slug']}: {e}", flush=True)
            f.flush()
            time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    run(ap.parse_args().interval)


if __name__ == "__main__":
    main()
