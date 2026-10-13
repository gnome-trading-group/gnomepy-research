"""Compare model probabilities with Polymarket (international) college football moneylines for one week.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.polymarket_week 2026-10-06 2026-10-13

Polymarket prices come from public endpoints (Gamma events, CLOB prices-history, hourly bars):
- open: first hourly bar off the 0.50 listing placeholder (the first real price)
- t72h / t24h: last bar at least 72h / 24h before kickoff
- pre_kickoff: last bar before kickoff
Model probabilities come from the latest posterior draws for the game's season and the outcome
model fitted on every completed season before it.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import N_FIT_DRAWS, fit, game_draws

GAMMA = "https://gamma-api.polymarket.com/events"
PRICES = "https://clob.polymarket.com/prices-history"
CFB_SERIES = 12756
FIRST_OUTCOME_SEASON = 2020
RESULTS = Path(__file__).resolve().parent.parent / "results" / "polymarket"
# Polymarket and CFBD name a few programs differently.
ALIASES = {"miami fl": "miami", "miami (fl)": "miami", "ole miss": "ole miss", "uconn": "connecticut",
           "southern miss": "southern mississippi", "fiu": "florida international", "utsa": "ut san antonio",
           "umass": "massachusetts", "app state": "appalachian state", "ul monroe": "louisiana monroe",
           "hawaii": "hawai'i", "sjsu": "san jose state", "nc state": "nc state", "byu": "byu"}


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    s = re.sub(r"[^a-z0-9' ()&-]", "", s)
    return ALIASES.get(s, s)


def session() -> requests.Session:
    s = requests.Session()
    # Gamma rejects default library user agents.
    s.headers["User-Agent"] = "Mozilla/5.0"
    return s


def fetch_moneylines(s: requests.Session, start: str, end: str) -> pd.DataFrame:
    events = []
    for closed in ("true", "false"):
        off = 0
        while True:
            page = s.get(GAMMA, params={"series_id": CFB_SERIES, "closed": closed, "limit": 500, "offset": off,
                                        "end_date_min": f"{start}T00:00:00Z", "end_date_max": f"{end}T00:00:00Z"},
                         timeout=60).json()
            events += page
            if len(page) < 500:
                break
            off += 500
    rows = []
    for e in events:
        for m in e.get("markets", []):
            if m.get("sportsMarketType") != "moneyline":
                continue
            outcomes = json.loads(m["outcomes"])
            tokens = json.loads(m["clobTokenIds"])
            prices = json.loads(m.get("outcomePrices") or "[]")
            rows.append({
                "event": e["title"], "slug": m["slug"], "kickoff": pd.Timestamp(m["gameStartTime"]),
                "listed": pd.Timestamp(m.get("startDate") or e["startDate"]), "team0": outcomes[0], "team1": outcomes[1],
                "token0": tokens[0], "closed": bool(m.get("closed")),
                "final0": float(prices[0]) if prices else np.nan, "volume": float(m.get("volumeNum") or 0),
            })
    return pd.DataFrame(rows).drop_duplicates("slug")


def price_path(s: requests.Session, token: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    r = s.get(PRICES, params={"market": token, "startTs": int(start.timestamp()), "endTs": int(end.timestamp()),
                              "fidelity": 60}, timeout=60).json()
    h = r.get("history", [])
    return pd.Series([x["p"] for x in h], index=pd.to_datetime([x["t"] for x in h], unit="s", utc=True), dtype=float)


def market_snapshots(s: requests.Session, ml: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in ml.itertuples():
        path = price_path(s, r.token0, r.listed, r.kickoff + pd.Timedelta(minutes=5))
        before = path[path.index < r.kickoff]
        priced = before[(before - 0.5).abs() > 0.005]
        t72 = before[before.index <= r.kickoff - pd.Timedelta(hours=72)]
        t24 = before[before.index <= r.kickoff - pd.Timedelta(hours=24)]
        out.append({"slug": r.slug, "p0_open": priced.iloc[0] if len(priced) else np.nan,
                    "open_at": priced.index[0] if len(priced) else pd.NaT,
                    "p0_t72h": t72.iloc[-1] if len(t72) else np.nan,
                    "p0_t24h": t24.iloc[-1] if len(t24) else np.nan,
                    "p0_pre_kickoff": before.iloc[-1] if len(before) else np.nan, "n_bars": len(before)})
    return ml.merge(pd.DataFrame(out), on="slug")


def match_games(ml: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games.assign(h=games["home_team"].map(norm_name), a=games["away_team"].map(norm_name))
    rows = []
    for r in ml.itertuples():
        t0, t1 = norm_name(r.team0), norm_name(r.team1)
        near = g[(g["kickoff"] - r.kickoff).abs() <= pd.Timedelta(hours=6)]
        hit = near[((near["h"] == t0) & (near["a"] == t1)) | ((near["h"] == t1) & (near["a"] == t0))]
        if len(hit) != 1:
            rows.append({"slug": r.slug, "game_id": np.nan, "team0_is_home": np.nan})
            continue
        rows.append({"slug": r.slug, "game_id": int(hit["game_id"].iloc[0]),
                     "team0_is_home": bool(hit["h"].iloc[0] == t0)})
    return ml.merge(pd.DataFrame(rows), on="slug")


def model_probabilities(games: pd.DataFrame, season: int) -> pd.DataFrame:
    model = fit(game_draws(games, list(range(FIRST_OUTCOME_SEASON, season)), n_draws=N_FIT_DRAWS))
    gd = game_draws(games, [season])
    pred = model.predict(gd.x)
    return gd.games[["game_id"]].assign(p_home_model=pred["p_home"].to_numpy(), p_lo=pred["p_lo"].to_numpy(),
                                        p_hi=pred["p_hi"].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    args = ap.parse_args()
    s = session()
    ml = fetch_moneylines(s, args.start, args.end)
    ml = market_snapshots(s, ml)
    games = DatasetStore().load("cfb_games")
    ml = match_games(ml, games)
    season = int(pd.Timestamp(args.start).year)
    m = ml.dropna(subset=["game_id"]).merge(model_probabilities(games, season), on="game_id", how="left")
    for col in ("p0_open", "p0_t72h", "p0_t24h", "p0_pre_kickoff"):
        m[col.replace("p0", "p_home")] = np.where(m["team0_is_home"], m[col], 1 - m[col])
    m["home_won"] = np.where(m["closed"] & m["final0"].isin([0.0, 1.0]),
                             np.where(m["team0_is_home"], m["final0"], 1 - m["final0"]), np.nan)
    m = m.merge(games[["game_id", "home_team", "away_team", "week"]], on="game_id")
    RESULTS.mkdir(parents=True, exist_ok=True)
    m.to_parquet(RESULTS / f"week_{args.start}.parquet", index=False)

    print(f"Polymarket moneylines: {len(ml)}; matched to CFBD: {ml['game_id'].notna().sum()}; "
          f"with a model probability: {m['p_home_model'].notna().sum()}")
    unmatched = ml[ml["game_id"].isna()]
    if len(unmatched):
        fbs = set(games["home_team"].map(norm_name)[games["home_classification"] == "fbs"])
        involves_fbs = unmatched[unmatched["team0"].map(norm_name).isin(fbs) | unmatched["team1"].map(norm_name).isin(fbs)]
        print(f"unmatched: {len(unmatched)} ({len(involves_fbs)} name an FBS team):",
              [f"{a} / {b}" for a, b in zip(involves_fbs["team0"], involves_fbs["team1"])])
    show = m.dropna(subset=["p_home_model"]).sort_values("kickoff")
    show = show.assign(edge=show["p_home_model"] - show["p_home_pre_kickoff"])
    show["open_h_before"] = (show["kickoff"] - show["open_at"]).dt.total_seconds() / 3600
    cols = ["kickoff", "away_team", "home_team", "p_home_model", "p_home_open", "open_h_before", "p_home_t24h",
            "p_home_pre_kickoff", "edge", "volume", "home_won"]
    print(show[cols].round(3).to_string(index=False))
    done = show.dropna(subset=["home_won", "p_home_pre_kickoff"])
    if len(done):
        y = done["home_won"].to_numpy()
        print(f"\ncompleted games: {len(done)}")
        for c in ("p_home_model", "p_home_open", "p_home_t72h", "p_home_t24h", "p_home_pre_kickoff"):
            ok = done[c].notna()
            print(f"  {c:20s} log loss {log_loss(y[ok], done.loc[ok, c].to_numpy()).mean():.4f} (n={ok.sum()})")
    print(f"\nmean |model - pre_kickoff| {show['edge'].abs().mean():.3f}; "
          f"corr(model, pre_kickoff) {show[['p_home_model', 'p_home_pre_kickoff']].corr().iloc[0, 1]:.3f}")


if __name__ == "__main__":
    main()
