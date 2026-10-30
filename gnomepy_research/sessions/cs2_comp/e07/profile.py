"""
FACEIT stat profile -> HLTV-scale rating.

HLTV's per-map rating is the individual-performance signal the Bayes model uses next to round results. FACEIT has no
rating, but 5,000 series sit in both datasets, so the same player-maps carry an HLTV rating and a FACEIT stat line.
A linear map from per-round FACEIT stats to the HLTV rating, fitted on series before 2026-01-01 and frozen in
profile_coefs.py, puts FACEIT performance on the scale the rating channel already understands. Older FACEIT rows lack
ADR, entry and utility stats, so a second map uses only the basics every row has.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load

FIT_BEFORE = pd.Timestamp("2026-01-01")

STAT_COLUMNS = ["match_id", "map_name", "faceit_id", "hltv_id", "kills", "deaths", "assists", "headshots", "mvps",
                "double_kills", "triple_kills", "quadro_kills", "penta_kills", "adr", "first_kills", "entry_wins",
                "1v1wins", "1v2wins", "utility_damage"]
BASIC = ["kpr", "dpr", "apr", "hs_rate", "mvp_pr", "multi_pr"]
FULL = BASIC + ["double_pr", "adr", "fk_pr", "entry_win_pr", "clutch_pr", "util_dmg_pr"]


def features(stats: pd.DataFrame, rounds: pd.Series) -> pd.DataFrame:
    """Per-round stat features; `rounds` is aligned with `stats` and holds the rounds of each player's map."""
    n = rounds.to_numpy(float)
    k = stats.kills.to_numpy(float)
    f = pd.DataFrame(index=stats.index)
    f["kpr"] = k / n
    f["dpr"] = stats.deaths.to_numpy(float) / n
    f["apr"] = stats.assists.to_numpy(float) / n
    f["hs_rate"] = np.where(k > 0, stats.headshots.to_numpy(float) / np.maximum(k, 1), 0.0)
    f["mvp_pr"] = stats.mvps.to_numpy(float) / n
    f["multi_pr"] = (stats.triple_kills + 2 * stats.quadro_kills + 3 * stats.penta_kills).to_numpy(float) / n
    f["double_pr"] = stats.double_kills.to_numpy(float) / n
    f["adr"] = stats.adr.to_numpy(float)
    f["fk_pr"] = stats.first_kills.to_numpy(float) / n
    f["entry_win_pr"] = stats.entry_wins.to_numpy(float) / n
    f["clutch_pr"] = (stats["1v1wins"] + stats["1v2wins"]).to_numpy(float) / n
    f["util_dmg_pr"] = stats.utility_damage.to_numpy(float) / n
    return f


def apply(stats: pd.DataFrame, rounds: pd.Series, coefs: dict) -> np.ndarray:
    f = features(stats, rounds)
    out = np.full(len(f), np.nan)
    valid_rounds = np.isfinite(rounds.to_numpy(float)) & (rounds.to_numpy(float) > 0)
    for name, cols in (("basic", BASIC), ("full", FULL)):
        x = f[cols].to_numpy(float)
        ok = np.isfinite(x).all(axis=1) & valid_rounds
        c = coefs[name]
        pred = c["intercept"] + x @ np.array([c["weights"][col] for col in cols])
        out[ok] = pred[ok]
    return out


def fit(data_dir: Path) -> dict:
    stats = load(data_dir, "faceit/faceit_player_map_stats", FIT_BEFORE, STAT_COLUMNS)
    stats = stats[stats.hltv_id.notna() & stats.kills.notna()]
    hist = load(data_dir, "faceit/faceit_match_history", FIT_BEFORE,
                ["match_id", "match_date", "map_name", "team_a_score", "team_b_score"])
    hist["rounds"] = hist.team_a_score + hist.team_b_score
    stats = stats.merge(hist[["match_id", "map_name", "rounds"]], on=["match_id", "map_name"])
    link = load(data_dir, "faceit/faceit_hltv_series_link", FIT_BEFORE)
    link = link[link.quality == "exact"][["faceit_match_id", "hltv_match_id"]]
    stats = stats.merge(link, left_on="match_id", right_on="faceit_match_id")
    hs = load(data_dir, "hltv/cs2_player_map_stats", FIT_BEFORE, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
    hs = hs[(hs.side == "all") & hs.rating.notna()].rename(columns={"match_id": "hltv_match_id", "player_id": "hltv_id"})
    d = stats.merge(hs[["hltv_match_id", "map_name", "hltv_id", "rating"]], on=["hltv_match_id", "map_name", "hltv_id"])
    d = d.reset_index(drop=True)
    f = features(d, d.rounds)
    out = {}
    for name, cols in (("basic", BASIC), ("full", FULL)):
        ok = np.isfinite(f[cols].to_numpy(float)).all(axis=1)
        x = np.column_stack([np.ones(ok.sum()), f.loc[ok, cols].to_numpy(float)])
        y = d.loc[ok, "rating"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        out[name] = {"intercept": float(beta[0]), "weights": {c: float(w) for c, w in zip(cols, beta[1:])},
                     "n": int(ok.sum()), "r2": float(1 - resid.var() / y.var()), "resid_sd": float(resid.std())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.data), indent=1))


if __name__ == "__main__":
    main()
