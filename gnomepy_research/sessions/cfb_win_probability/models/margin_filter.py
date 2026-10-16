"""Scoreboard channel (Phase 23): a closed-form Kalman filter on game margins, run beside the efficiency model.

    theta_t,0 ~ N(a * s_t + b * P4_t + c, tau_tier^2)      s_t: the efficiency model's preseason strength
    y_g = theta_h - theta_a + HFA * venue + eps,  eps ~ N(0, SIGMA^2),  y capped at +/- CAP
    theta_t,w = theta_t,w-1 + eta,  eta ~ N(0, Q) per week elapsed

Margins carry what per-play efficiency misses (turnovers, special teams, non-offensive scores, finishing), and the two
channels' errors are only partly correlated, so the outcome layer weights both. The prior map (a, b, c, tau) comes from
regressing end-of-season SRS on s over earlier seasons only; the tiered map adds a Power-4 offset b and per-tier tau
(Group-of-5 preseason guesses are rougher). Each game's prediction uses only games that kicked off before it. The
fixed constants were measured on 2018-24 game data: one-game margin noise about 13 points, home edge about 2.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.features.ratings import FCS_TEAM

SIGMA = 13.0
HFA = 2.5
CAP = 28.0
Q = 1.0
FCS_MEAN, FCS_VAR = -26.0, 36.0
P4_STABLE = {"SEC", "Big Ten", "Big 12", "ACC"}
PAC12_LAST_P4_SEASON = 2023


def team_tiers(games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """(season, team_id, p4) for FBS teams; a team's conference is its most frequent one that season."""
    g = games[games["season"].isin(seasons)]
    rows = pd.concat([
        g[["season", "home_id", "home_conference", "home_team", "home_classification"]].set_axis(["season", "team_id", "conf", "team", "cls"], axis=1),
        g[["season", "away_id", "away_conference", "away_team", "away_classification"]].set_axis(["season", "team_id", "conf", "team", "cls"], axis=1),
    ])
    rows = rows[rows["cls"] == "fbs"].dropna(subset=["conf"])
    t = rows.groupby(["season", "team_id"]).agg(conf=("conf", lambda s: s.mode().iloc[0]), team=("team", "first")).reset_index()
    t["p4"] = t["conf"].isin(P4_STABLE) | ((t["conf"] == "Pac-12") & (t["season"] <= PAC12_LAST_P4_SEASON)) | (t["team"] == "Notre Dame")
    return t[["season", "team_id", "p4"]]


def preseason_strength(draws: dict) -> pd.Series:
    """Team id -> mean over rating draws of summed offense + defense (all units) at the season's first period."""
    first = min((k.split("__")[0] for k in draws if k.startswith("regular_")), key=lambda k: int(k.split("_")[1]))
    s = pd.Series((draws[f"{first}__off"] + draws[f"{first}__def"]).mean(0), index=draws[f"{first}__team_ids"])
    return s.groupby(level=0).sum().drop(FCS_TEAM, errors="ignore")


def fit_prior_map(x: pd.DataFrame, tiered: bool) -> dict:
    """x: strength s, p4 flag and end-of-season srs. OLS srs ~ a*s (+ b*p4) + c; tau = residual sd (per tier if tiered)."""
    A = np.column_stack([x["s"], *([x["p4"].astype(float)] if tiered else []), np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, x["srs"].to_numpy(float), rcond=None)
    res = x["srs"].to_numpy(float) - A @ coef
    p4 = x["p4"].to_numpy(bool)
    tau = (float(res[p4].std()), float(res[~p4].std())) if tiered else (float(res.std()),) * 2
    return {"a": float(coef[0]), "b": float(coef[1]) if tiered else 0.0, "c": float(coef[-1]), "tau_p4": tau[0], "tau_g5": tau[1]}


def season_priors(strength: pd.DataFrame, srs: pd.Series, season: int, tiered: bool) -> tuple[dict, dict]:
    """Prior (mean, var) per FBS team for `season`, the map fitted on earlier seasons (the season itself if none),
    means centred on the FBS average."""
    have = strength.join(srs.rename("srs"), on=["season", "team_id"]).dropna(subset=["srs"])
    train = have[have["season"] < season]
    m = fit_prior_map(train if len(train) else have[have["season"] == season], tiered)
    cur = strength[strength["season"] == season]
    mu = m["a"] * cur["s"] + m["b"] * cur["p4"].astype(float) + m["c"]
    mu = mu - mu.mean()
    var = np.where(cur["p4"], m["tau_p4"] ** 2, m["tau_g5"] ** 2)
    return {int(t): (float(u), float(v)) for t, u, v in zip(cur["team_id"], mu, var)}, m


def run_filter(games: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Pre-game mean and variance of theta_home - theta_away for every game in `games` (one season), updating on each
    completed game's capped margin. Teams without a prior (FCS) sit at FCS_MEAN with FCS_VAR and are not updated."""
    state = {t: list(mv) for t, mv in priors.items()}
    last = {}
    rows = []
    g = games.sort_values(["kickoff", "game_id"])
    for r in g.itertuples(index=False):
        step = 1 if r.season_type == "postseason" else int(r.week)
        for t in (r.home_id, r.away_id):
            if t in state and t in last:
                state[t][1] += Q * max(step - last[t], 1 if r.season_type == "postseason" else 0)
        mh, vh = state.get(r.home_id, (FCS_MEAN, FCS_VAR))
        ma, va = state.get(r.away_id, (FCS_MEAN, FCS_VAR))
        rows.append((r.game_id, mh - ma, vh + va))
        if not (bool(r.completed) and pd.notna(r.home_points) and pd.notna(r.away_points)):
            continue
        venue = 0.0 if r.neutral_site else 1.0
        innov = float(np.clip(r.home_points - r.away_points, -CAP, CAP)) - (mh - ma + HFA * venue)
        total = vh + va + SIGMA ** 2
        if r.home_id in state:
            state[r.home_id][0] += vh / total * innov
            state[r.home_id][1] -= vh ** 2 / total
            last[r.home_id] = step
        if r.away_id in state:
            state[r.away_id][0] -= va / total * innov
            state[r.away_id][1] -= va ** 2 / total
            last[r.away_id] = step
    return pd.DataFrame(rows, columns=["game_id", "kf_diff", "kf_var"])


def kf_table(games: pd.DataFrame, srs: pd.Series, strengths: dict[int, pd.Series], tiered: bool) -> pd.DataFrame:
    """game_id -> kf_diff (points, home minus away, no home edge) and kf_var for every season in `strengths`
    (season -> preseason_strength). srs: end-of-season SRS indexed by (season, team_id)."""
    seasons = sorted(strengths)
    tiers = team_tiers(games, seasons)
    strength = pd.concat([s.rename("s").rename_axis("team_id").reset_index().assign(season=y) for y, s in strengths.items()])
    strength = strength.merge(tiers, on=["season", "team_id"], how="inner")
    out = []
    for season in seasons:
        priors, _ = season_priors(strength, srs, season, tiered)
        out.append(run_filter(games[games["season"] == season], priors))
    return pd.concat(out, ignore_index=True)
