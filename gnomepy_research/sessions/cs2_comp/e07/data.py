"""
Event streams for the player-skill filter: every competitive map from HLTV and (optionally) FACEIT, in time order.

A player is one id whichever source shows him: his HLTV id when hltv_faceit_map links him, otherwise his FACEIT id.
FACEIT-only players are players too, so FACEIT opponents connect the rating graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.e07 import profile
from gnomepy_research.sessions.cs2_comp.e07.profile_coefs import COEFS

HLTV, FACEIT = 0, 1
NS_PER_DAY = 86_400 * 10 ** 9
LINEUP = 5
OVERTIME_ROUNDS = 24


@dataclass
class Streams:
    t: np.ndarray          # (M,) days since epoch
    src: np.ndarray        # (M,) HLTV or FACEIT
    a: np.ndarray          # (M, 5) player indices
    b: np.ndarray
    rounds_a: np.ndarray   # (M,) regulation rounds won (overtime maps count 12.5 / 11.5 of 24)
    rounds_b: np.ndarray
    dev_a: np.ndarray      # (M, 5) rating minus the team's mean on the map; NaN where unknown
    dev_b: np.ndarray
    team_a: np.ndarray     # (M,) HLTV team id or -1
    team_b: np.ndarray
    level_a: np.ndarray    # (M, 5) FACEIT skill level of each player on the map; NaN where unknown
    level_b: np.ndarray
    n_players: int
    index: dict            # player key -> index
    series: pd.DataFrame   # HLTV series: match_id, team_a_id, team_b_id, t, bo_type, y


def _days(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    return ts.to_numpy("datetime64[ns]").astype(np.int64) / NS_PER_DAY


def _regulation(score_a: np.ndarray, score_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = score_a + score_b
    overtime = total > OVERTIME_ROUNDS
    ra = np.where(overtime, 12.0 + np.where(score_a > score_b, 0.5, -0.5), score_a)
    rb = np.where(overtime, OVERTIME_ROUNDS - ra, score_b)
    return ra, rb


def _hltv_maps(data_dir, before) -> pd.DataFrame:
    cols = ["match_id", "match_date", "match_time", "event_name", "format", "bo_type", "is_lan", "team_a_id", "team_b_id",
            "map_name", "map_position_in_series", "team_a_score", "team_b_score", "team_a_player_ids", "team_b_player_ids"]
    h = bundle.load(data_dir, "hltv/cs2_match_history", before, cols)
    h = h[~showmatch_mask(h) & (h.team_a_score != h.team_b_score)]
    five = h.team_a_player_ids.map(lambda x: x is not None and len(x) == LINEUP) \
        & h.team_b_player_ids.map(lambda x: x is not None and len(x) == LINEUP)
    h = h[five].copy()
    h["t"] = _days(h.match_time.fillna(h.match_date))
    return h.sort_values(["t", "match_id", "map_position_in_series"], kind="stable").reset_index(drop=True)


def _faceit_maps(data_dir, before) -> pd.DataFrame:
    cols = ["match_id", "match_date", "match_time", "map_name", "map_position_in_series", "team_a_id", "team_b_id",
            "team_a_score", "team_b_score", "team_a_faceit_ids", "team_b_faceit_ids"]
    f = bundle.load(data_dir, "faceit/faceit_match_history", before, cols)
    ex = bundle.load(data_dir, "faceit/faceit_exclusions", before)
    whole = ex[ex.map_name.isna()].match_id
    f = f[~f.match_id.isin(whole)]
    part = ex[ex.map_name.notna()].drop_duplicates(["match_id", "map_name"])
    f = f.merge(part.assign(drop=True)[["match_id", "map_name", "drop"]], on=["match_id", "map_name"], how="left")
    f = f[f["drop"].isna()].drop(columns="drop")
    five = f.team_a_faceit_ids.map(lambda x: len(x) == LINEUP) & f.team_b_faceit_ids.map(lambda x: len(x) == LINEUP)
    f = f[five & (f.team_a_score != f.team_b_score)].copy()
    f["t"] = _days(f.match_time.fillna(f.match_date))
    return f.sort_values(["t", "match_id", "map_position_in_series"], kind="stable").reset_index(drop=True)


def _faceit_to_hltv(data_dir) -> dict:
    m = bundle.load(data_dir, "faceit/hltv_faceit_map")
    return dict(zip(m.faceit_id, m.hltv_id.astype(int)))


def _lineup_rows(lineups: list, n: int) -> np.ndarray:
    return np.array(lineups, dtype=object).reshape(n, LINEUP)


def _devs(rows: pd.DataFrame, n: int) -> np.ndarray:
    """rows: one per (map, side, slot) in order with a 'rating' column; deviations from the team mean, NaN unless all five known."""
    r = rows.rating.to_numpy(float).reshape(n * 2, LINEUP)
    mean = r.mean(axis=1, keepdims=True)
    dev = np.where(np.isfinite(r).all(axis=1, keepdims=True), r - mean, np.nan)
    return dev.reshape(n, 2, LINEUP)


def _hltv_ratings(data_dir, before, maps: pd.DataFrame) -> np.ndarray:
    s = bundle.load(data_dir, "hltv/cs2_player_map_stats", before, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
    s = s[(s.side == "all") & s.rating.notna()][["match_id", "map_name", "player_id", "rating"]]
    s = s.drop_duplicates(["match_id", "map_name", "player_id"])
    n = len(maps)
    rows = pd.DataFrame({
        "match_id": np.repeat(maps.match_id.to_numpy(), 2 * LINEUP),
        "map_name": np.repeat(maps.map_name.to_numpy(), 2 * LINEUP),
        "player_id": np.concatenate([np.concatenate([np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)])
                                     for a, b in zip(maps.team_a_player_ids, maps.team_b_player_ids)]),
    })
    rows = rows.merge(s, on=["match_id", "map_name", "player_id"], how="left", sort=False)
    return _devs(rows, n)


def _faceit_ratings(data_dir, before, maps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = bundle.load(data_dir, "faceit/faceit_player_map_stats", before, profile.STAT_COLUMNS)
    rounds = (maps.team_a_score + maps.team_b_score).to_numpy(float)
    rmap = pd.DataFrame({"match_id": maps.match_id.to_numpy(), "map_name": maps.map_name.to_numpy(), "rounds": rounds})
    s = s.merge(rmap, on=["match_id", "map_name"])
    s = s.assign(rating=profile.apply(s, s.rounds, COEFS))
    level = load_levels(data_dir, before)
    s = s[["match_id", "map_name", "faceit_id", "rating"]].drop_duplicates(["match_id", "map_name", "faceit_id"])
    n = len(maps)
    ids = np.concatenate([np.concatenate([np.asarray(a, dtype=object), np.asarray(b, dtype=object)])
                          for a, b in zip(maps.team_a_faceit_ids, maps.team_b_faceit_ids)])
    rows = pd.DataFrame({"match_id": np.repeat(maps.match_id.to_numpy(), 2 * LINEUP),
                         "map_name": np.repeat(maps.map_name.to_numpy(), 2 * LINEUP), "faceit_id": ids})
    rows = rows.merge(s, on=["match_id", "map_name", "faceit_id"], how="left", sort=False)
    lv = rows[["match_id", "faceit_id"]].merge(level, on=["match_id", "faceit_id"], how="left", sort=False)
    return _devs(rows, n), lv.skill_level.to_numpy(float).reshape(n, 2, LINEUP)


def load_levels(data_dir, before) -> pd.DataFrame:
    r = bundle.load(data_dir, "faceit/faceit_rosters", before, ["match_id", "faceit_id", "skill_level"])
    return r.drop_duplicates(["match_id", "faceit_id"])


def build(data_dir, before: pd.Timestamp, use_faceit: bool) -> Streams:
    h = _hltv_maps(data_dir, before)
    nh = len(h)
    pieces = {"t": [h.t.to_numpy()], "src": [np.full(nh, HLTV)], "team_a": [h.team_a_id.fillna(-1).to_numpy(np.int64)],
              "team_b": [h.team_b_id.fillna(-1).to_numpy(np.int64)]}
    ra, rb = _regulation(h.team_a_score.to_numpy(float), h.team_b_score.to_numpy(float))
    pieces["ra"], pieces["rb"] = [ra], [rb]
    keys_a = [list(map(int, x)) for x in h.team_a_player_ids]
    keys_b = [list(map(int, x)) for x in h.team_b_player_ids]
    dev = _hltv_ratings(data_dir, before, h)
    devs = [(dev[:, 0], dev[:, 1])]
    levels = [(np.full((nh, LINEUP), np.nan), np.full((nh, LINEUP), np.nan))]
    won = (h.team_a_score > h.team_b_score).astype(int)
    series = h.assign(won=won).groupby("match_id", sort=False).agg(
        team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"), t=("t", "min"),
        bo_type=("bo_type", "first"), is_lan=("is_lan", "first"), maps_a=("won", "sum"), maps=("won", "size")).reset_index()
    series["y"] = (series.maps_a * 2 > series.maps).astype(int)
    first = h.drop_duplicates("match_id").set_index("match_id")
    series["a_players"] = [tuple(first.at[m, "team_a_player_ids"]) for m in series.match_id]
    series["b_players"] = [tuple(first.at[m, "team_b_player_ids"]) for m in series.match_id]
    if use_faceit:
        f = _faceit_maps(data_dir, before)
        nf = len(f)
        link = _faceit_to_hltv(data_dir)
        key = lambda ids: [link.get(i, i) for i in ids]
        keys_a += [key(x) for x in f.team_a_faceit_ids]
        keys_b += [key(x) for x in f.team_b_faceit_ids]
        fra, frb = _regulation(f.team_a_score.to_numpy(float), f.team_b_score.to_numpy(float))
        pieces["t"].append(f.t.to_numpy())
        pieces["src"].append(np.full(nf, FACEIT))
        pieces["team_a"].append(f.team_a_id.fillna(-1).to_numpy(np.int64))
        pieces["team_b"].append(f.team_b_id.fillna(-1).to_numpy(np.int64))
        pieces["ra"].append(fra)
        pieces["rb"].append(frb)
        fdev, flevel = _faceit_ratings(data_dir, before, f)
        devs.append((fdev[:, 0], fdev[:, 1]))
        levels.append((flevel[:, 0], flevel[:, 1]))
    t = np.concatenate(pieces["t"])
    order = np.argsort(t, kind="stable")
    index: dict = {}
    for k in (k for ks in (keys_a, keys_b) for five in ks for k in five):
        index.setdefault(k, len(index))
    a = np.array([[index[k] for k in five] for five in keys_a], dtype=np.int64)[order]
    b = np.array([[index[k] for k in five] for five in keys_b], dtype=np.int64)[order]
    return Streams(
        t=t[order], src=np.concatenate(pieces["src"])[order], a=a, b=b,
        rounds_a=np.concatenate(pieces["ra"])[order], rounds_b=np.concatenate(pieces["rb"])[order],
        dev_a=np.concatenate([d[0] for d in devs])[order], dev_b=np.concatenate([d[1] for d in devs])[order],
        team_a=np.concatenate(pieces["team_a"])[order], team_b=np.concatenate(pieces["team_b"])[order],
        level_a=np.concatenate([x[0] for x in levels])[order], level_b=np.concatenate([x[1] for x in levels])[order],
        n_players=len(index), index=index, series=series)


class RankPoints:
    """HLTV ranking points of a team on or before a day (days since epoch); 0 when unranked or the ranking is stale."""

    MAX_AGE_DAYS = 120.0

    def __init__(self, rankings: pd.DataFrame):
        r = rankings.assign(day=_days(rankings["date"])).sort_values("day")
        self._hist = {int(k): (g.day.to_numpy(), g.points.to_numpy(float)) for k, g in r.groupby("team_id")}

    def __call__(self, team: int, day: float) -> float:
        h = self._hist.get(int(team))
        if h is None:
            return 0.0
        i = int(np.searchsorted(h[0], day, side="right")) - 1
        if i < 0 or day - h[0][i] > self.MAX_AGE_DAYS:
            return 0.0
        return float(h[1][i])
