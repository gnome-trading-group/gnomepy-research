"""A FACEIT performance rating on the HLTV rating scale, learned from series both sites cover.

For the linked series the same player-map has an HLTV rating and a FACEIT stat row. Within a team the stats are
reduced to deviations from the teammates' mean (the team's result is common to all five and cancels), compressed to a
few principal components, and ridge-regressed onto the HLTV rating deviation. The result maps any FACEIT player-map to a
rating deviation that the skill filter reads like an HLTV one.

The mapping is fitted once on series before FIT_END and frozen, so it only uses data every evaluated cutoff has.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e10.data import faceit_stats, hltv_rating_dev

FIT_END = pd.Timestamp("2025-10-01")
N_COMPONENTS = 8
RIDGE = 30.0
DETAIL_KEYS = ["adr", "first_kills", "entry_count", "clutch_kills", "utility_damage", "enemies_flashed", "pistol_kills"]


def stat_features(s: pd.DataFrame, rounds: pd.Series) -> pd.DataFrame:
    """Per-round and per-kill features of a FACEIT player-map stat frame, aligned to s's index."""
    r = rounds.to_numpy(np.float32)
    kills = s.kills.to_numpy(np.float32)
    f = pd.DataFrame(index=s.index)
    f["kpr"] = kills / r
    f["dpr"] = s.deaths.to_numpy(np.float32) / r
    f["apr"] = s.assists.to_numpy(np.float32) / r
    f["hs"] = s.headshots.to_numpy(np.float32) / 100.0
    f["mvp"] = s.mvps.to_numpy(np.float32) / r
    f["multi"] = (s.double_kills.fillna(0) + 2 * s.triple_kills.fillna(0) + 3 * s.quadro_kills.fillna(0)
                  + 4 * s.penta_kills.fillna(0)).to_numpy(np.float32) / r
    f["kd"] = np.clip(s.k_d_ratio.to_numpy(np.float32), 0, 5)
    f["adr"] = s.adr.to_numpy(np.float32) / 100.0
    f["fk"] = s.first_kills.to_numpy(np.float32) / r
    f["entry"] = s.entry_count.to_numpy(np.float32) / r
    f["entry_win"] = s.entry_wins.to_numpy(np.float32) / np.maximum(s.entry_count.to_numpy(np.float32), 1)
    f["clutch"] = (s["1v1wins"] + s["1v2wins"]).to_numpy(np.float32) / r
    f["pistol"] = s.pistol_kills.to_numpy(np.float32) / r
    f["sniper"] = s.sniper_kills.to_numpy(np.float32) / np.maximum(kills, 1)
    f["udmg"] = s.utility_damage.to_numpy(np.float32) / r / 10.0
    f["flash"] = s.enemies_flashed.to_numpy(np.float32) / r
    f["util"] = s.utility_successes.to_numpy(np.float32) / r
    return f


def team_deviation(f: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Each feature minus its mean over the five players of the same team and map; missing detail stays missing."""
    g = f.groupby([keys.match_id.to_numpy(), keys.map_name.to_numpy(), keys.faction.to_numpy()])
    return f - g.transform("mean")


@dataclass
class RatingMap:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    coef: np.ndarray
    intercept: float
    mse: float

    def predict(self, dev: pd.DataFrame) -> np.ndarray:
        x = (dev[self.columns].fillna(0.0).to_numpy(np.float64) - self.mean) / self.scale
        return (x @ self.components.T) @ self.coef + self.intercept


@dataclass
class RatingMaps:
    detail: RatingMap
    basic: RatingMap

    def rating_dev(self, s: pd.DataFrame, rounds: pd.Series) -> pd.Series:
        """Predicted HLTV-scale rating deviation for every FACEIT player-map row of s, with team mean zero."""
        f = stat_features(s, rounds)
        dev = team_deviation(f, s)
        has_detail = s.adr.notna().groupby([s.match_id.to_numpy(), s.map_name.to_numpy(), s.faction.to_numpy()]).transform("all")
        out = np.zeros(len(s))
        for model, sel in ((self.detail, has_detail.to_numpy()), (self.basic, ~has_detail.to_numpy())):
            if sel.any():
                out[sel] = model.predict(dev[sel])
        pred = pd.Series(out, index=s.index)
        return pred - pred.groupby([s.match_id.to_numpy(), s.map_name.to_numpy(), s.faction.to_numpy()]).transform("mean")


def _fit(dev: pd.DataFrame, y: np.ndarray, columns: list[str], k: int) -> tuple[RatingMap, dict]:
    x = dev[columns].to_numpy(np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-9
    z = (x - mean) / scale
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    comps = vt[:k]
    pc = z @ comps.T
    y0 = y - y.mean()
    coef = np.linalg.solve(pc.T @ pc + RIDGE * np.eye(k), pc.T @ y0)
    resid = y0 - pc @ coef
    full_coef = np.linalg.solve(z.T @ z + RIDGE * np.eye(z.shape[1]), z.T @ y0)
    info = {"n": len(y), "r2_pcr": 1 - resid.var() / y0.var(), "r2_full": 1 - (y0 - z @ full_coef).var() / y0.var(),
            "explained": float((sv[:k] ** 2).sum() / (sv ** 2).sum())}
    return RatingMap(columns, mean, scale, comps, coef, float(y.mean()), float(resid.var())), info


def fit_rating_maps(data_dir: Path, k: int = N_COMPONENTS) -> tuple[RatingMaps, dict]:
    link = load(data_dir, "faceit/faceit_hltv_series_link", FIT_END)
    fmh = load(data_dir, "faceit/faceit_match_history", FIT_END, ["match_id", "match_date", "map_name", "team_a_score", "team_b_score"])
    fmh = fmh[fmh.match_id.isin(link.faceit_match_id)]
    rounds = (fmh.team_a_score + fmh.team_b_score).rename("rounds")
    fmh = fmh.assign(rounds=rounds).drop_duplicates(["match_id", "map_name"])
    s = faceit_stats(data_dir, FIT_END, set(link.faceit_match_id))
    s = s.merge(fmh[["match_id", "map_name", "rounds"]], on=["match_id", "map_name"])
    s = s[s.rounds > 0].reset_index(drop=True)
    f = stat_features(s, s.rounds)
    dev = team_deviation(f, s)
    has_detail = s.adr.notna().groupby([s.match_id.to_numpy(), s.map_name.to_numpy(), s.faction.to_numpy()]).transform("all").to_numpy()
    hs = hltv_rating_dev(data_dir, FIT_END)
    pair_ix = s.merge(link[["faceit_match_id", "hltv_match_id"]], left_on="match_id", right_on="faceit_match_id", how="left")
    assert len(pair_ix) == len(s)
    pair_ix["row"] = np.arange(len(pair_ix))
    pair_ix["hltv_id"] = pair_ix.hltv_id.astype("float64")
    j = pair_ix.dropna(subset=["hltv_id", "hltv_match_id"]).assign(hltv_id=lambda d: d.hltv_id.astype(int)).merge(
        hs, left_on=["hltv_match_id", "map_name", "hltv_id"], right_on=["match_id", "map_name", "player_id"], suffixes=("", "_h"))
    rows = j.row.to_numpy()
    y = j.dev.to_numpy(np.float64)
    out, info = {}, {}
    maps = {}
    for name, columns, sel in (("detail", list(dev.columns), has_detail), ("basic", [c for c in dev.columns if c not in
                               ("adr", "fk", "entry", "entry_win", "clutch", "pistol", "udmg", "flash", "util")], ~has_detail)):
        mask = sel[rows]
        d = dev.iloc[rows[mask]].fillna(0.0)
        maps[name], info[name] = _fit(d, y[mask], columns, min(k, len(columns)))
    return RatingMaps(maps["detail"], maps["basic"]), info
