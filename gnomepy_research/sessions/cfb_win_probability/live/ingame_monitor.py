"""Live in-game monitor: CFBD live state -> win-probability model vs Polymarket, per game.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.live.ingame_monitor --interval 20

CFBD live calls count against the monthly quota (75k on tier 3): ~17 games at a 20 s interval
for a 4 h window is ~12k calls. The prior is the consensus closing line (no-vig moneyline, else
spread), the same prior the model was trained on. Timeouts aren't in the live feed and enter
the model as missing, which it saw in ~2% of training rows.

Rows go to results/live/ingame_<date>.csv. A row is flagged `fresh_score` for 60 s after the
score changes: Polymarket finishes repricing a score ~50 s before our feeds show it, so the
model must never trade against a price that knows something our feed doesn't.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from scipy.stats import norm

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.polymarket_week import match_games
from gnomepy_research.sessions.cfb_win_probability.live.tracker import book_top, polymarket_moneylines
from gnomepy_research.sessions.cfb_win_probability.live.tracker import session as pm_session
from gnomepy_research.sessions.cfb_win_probability.live.wp_data import (
    FEATURES,
    SPREAD_SD,
    game_seconds_left,
    half_seconds_left,
)

CFBD = "https://api.collegefootballdata.com"
MODEL_DIR = Path(__file__).resolve().parent.parent / "results" / "ingame"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "live"
FRESH_SCORE_S = 60
FIELDS = ["ts", "slug", "game_id", "home", "away", "home_score", "away_score", "period", "clock", "possession",
          "down", "distance", "yards_to_goal", "prior_home", "model_home", "pm_mid_home", "pm_bid_home",
          "pm_ask_home", "edge", "fresh_score"]


class LiveWinProb:
    def __init__(self, name: str = "wp_xgb_2018_2024") -> None:
        meta = json.loads((MODEL_DIR / f"{name}.meta.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(MODEL_DIR / f"{name}.json")
        self.best_iteration = meta["best_iteration"]
        self.a, self.b = meta["calib_a"], meta["calib_b"]

    def predict(self, x: pd.DataFrame) -> float:
        if int(x["is_ot"].iloc[0]) == 1:
            return float(x["prior"].iloc[0])
        raw = float(np.clip(self.booster.predict(xgb.DMatrix(x[FEATURES]), iteration_range=(0, self.best_iteration + 1))[0],
                            1e-6, 1 - 1e-6))
        return float(1 / (1 + np.exp(-(self.a + self.b * np.log(raw / (1 - raw))))))


def _clock_s(clock: str | None) -> float:
    if not clock or ":" not in clock:
        return 0.0
    m, s = clock.split(":")[:2]
    return float(m) * 60 + float(s)


def live_features(d: dict, prior_home: float) -> tuple[pd.DataFrame, dict]:
    """Model inputs for the next snap from a CFBD /live/plays response."""
    teams = {t["homeAway"]: t for t in d["teams"]}
    home, away = teams["home"]["team"], teams["away"]["team"]
    home_score, away_score = float(teams["home"]["points"]), float(teams["away"]["points"])
    period = int(d.get("period") or 1)
    clock = _clock_s(d.get("clock"))
    plays = [p for dr in d.get("drives", []) for p in dr.get("plays", [])]
    if plays and int(plays[-1]["period"]) == period:
        # The top-level clock can lag the latest play by a minute or more.
        clock = min(clock, _clock_s(plays[-1]["clock"]))
    first_offense = d["drives"][0]["offense"] if d.get("drives") else None
    prior_home = float(np.clip(prior_home, 0.005, 0.995))
    sec_game = float(game_seconds_left(pd.Series([period]), pd.Series([clock]))[0])
    row = {
        "prior": prior_home, "prior_logit": np.log(prior_home / (1 - prior_home)),
        "score_diff": home_score - away_score, "sec_game": sec_game,
        "sec_half": float(half_seconds_left(pd.Series([period]), pd.Series([clock]))[0]),
        "period": period, "is_ot": int(period >= 5), "home_has_ball": int(d.get("possession") == home),
        "yards_to_goal": d.get("yardsToGoal"), "down": d.get("down") or np.nan, "distance": d.get("distance"),
        "home_timeouts": np.nan, "away_timeouts": np.nan,
        "home_receives_2h": int(first_offense is not None and first_offense != home),
    }
    row["prior_x_remaining"] = row["prior_logit"] * sec_game / 3600.0
    row["diff_per_sqrt_time"] = row["score_diff"] / np.sqrt((sec_game + 60) / 60)
    state = {"home": home, "away": away, "home_score": home_score, "away_score": away_score, "period": period,
             "clock": d.get("clock"), "possession": d.get("possession"), "down": d.get("down"),
             "distance": d.get("distance"), "yards_to_goal": d.get("yardsToGoal"), "status": d.get("status")}
    return pd.DataFrame([row]), state


def closing_prior(cfbd: requests.Session, game_id: int) -> float | None:
    lines = cfbd.get(f"{CFBD}/lines", params={"gameId": game_id}, timeout=15).json()
    if not lines:
        return None
    probs, spreads = [], []
    for ln in lines[0].get("lines", []):
        h, a = ln.get("homeMoneyline"), ln.get("awayMoneyline")
        if h and a:
            ph = 100 / (h + 100) if h > 0 else -h / (-h + 100)
            pa = 100 / (a + 100) if a > 0 else -a / (-a + 100)
            probs.append(ph / (ph + pa))
        if ln.get("spread") is not None:
            spreads.append(float(ln["spread"]))
    if probs:
        return float(np.median(probs))
    return float(norm.cdf(-np.median(spreads) / SPREAD_SD)) if spreads else None


def run(interval: float) -> None:
    pm = pm_session()
    cfbd = requests.Session()
    cfbd.headers["Authorization"] = f"Bearer {os.environ['CFBD_API_KEY']}"
    model = LiveWinProb()
    games = DatasetStore().load("cfb_games")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"ingame_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.csv"
    new = not path.exists()
    priors: dict[int, float | None] = {}
    last_score: dict[int, tuple[float, float, float]] = {}
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        tracked, refreshed = [], 0.0
        while True:
            if time.time() - refreshed > 300:
                live = {int(g["id"]) for g in cfbd.get(f"{CFBD}/scoreboard", params={"classification": "fbs"}, timeout=20).json()
                        if g.get("status") == "in_progress"}
                # Scoreboard names carry mascots ("Troy Trojans"); match markets to games by kickoff and
                # school name instead, which also says which outcome is the home team.
                ml = match_games(pd.DataFrame(polymarket_moneylines(pm, pd.Timestamp.now(tz="UTC"))), games)
                tracked = ml[ml["game_id"].isin(live)].to_dict("records")
                refreshed = time.time()
                print(f"{pd.Timestamp.now(tz='UTC'):%H:%M:%S} monitoring {len(tracked)} games", flush=True)
            for m in tracked:
                gid = int(m["game_id"])
                try:
                    if gid not in priors:
                        priors[gid] = closing_prior(cfbd, gid)
                    if priors[gid] is None:
                        continue
                    r = cfbd.get(f"{CFBD}/live/plays", params={"gameId": gid}, timeout=15)
                    if r.status_code != 200:
                        continue
                    x, st = live_features(r.json(), priors[gid])
                    p_home = model.predict(x)
                    top = book_top(pm, m["token0"])
                    if m["team0_is_home"]:
                        mid, bid, ask = top["mid0"], top["bid0"], top["ask0"]
                    else:
                        mid = None if top["mid0"] is None else 1 - top["mid0"]
                        bid = None if top["ask0"] is None else 1 - top["ask0"]
                        ask = None if top["bid0"] is None else 1 - top["bid0"]
                    now = time.time()
                    score = (st["home_score"], st["away_score"])
                    prev = last_score.get(gid)
                    if prev is None or prev[:2] != score:
                        last_score[gid] = (*score, now if prev is not None else 0.0)
                    fresh = now - last_score[gid][2] < FRESH_SCORE_S
                    w.writerow({"ts": pd.Timestamp.now(tz="UTC").isoformat(), "slug": m["slug"], "game_id": gid,
                                **{k: st[k] for k in ("home", "away", "home_score", "away_score", "period", "clock",
                                                      "possession", "down", "distance", "yards_to_goal")},
                                "prior_home": round(priors[gid], 4), "model_home": round(p_home, 4),
                                "pm_mid_home": mid, "pm_bid_home": bid, "pm_ask_home": ask,
                                "edge": None if mid is None else round(p_home - mid, 4), "fresh_score": fresh})
                except (requests.RequestException, KeyError, ValueError, IndexError) as e:
                    print(f"{m['slug']}: {type(e).__name__} {e}", flush=True)
            f.flush()
            time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=20.0)
    run(ap.parse_args().interval)


if __name__ == "__main__":
    main()
