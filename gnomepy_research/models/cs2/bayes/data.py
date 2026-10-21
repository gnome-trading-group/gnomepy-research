"""
Model inputs for one fit window, point in time: only maps dated before the window's end are used.

Each map contributes its regulation rounds (an overtime map counts as 12-12 of 24) between the two announced
lineups, and each player's map rating relative to his teammates' mean on that map. Skills live on a monthly grid:
a map indexes the month it was played in.

A debutant is a player whose first map anywhere in `maps` falls inside the window. Players appearing in the data's
first DEBUT_GRACE_DAYS are never debutants: the data starting is not the player starting.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import LINEUP_SIZE, RankStrength, home_regions

DEBUT_GRACE_DAYS = 60


@dataclass
class FitData:
    months: list[pd.Timestamp]
    player_ids: np.ndarray       # (P,)
    strength: np.ndarray         # (P,) standardized strength of the team each player debuted for in the window
    strength_mu: float
    strength_sd: float
    debutant: np.ndarray         # (P,) first map anywhere in the data falls inside the window
    career_ids: np.ndarray       # every player with a map before the window's end
    home: dict[int, int]
    tm_team_ids: np.ndarray      # (K,)
    tm_maps: np.ndarray          # (K,)
    plays: pd.DataFrame          # team_id, map_name, maps
    n_series: int
    lineup_a: np.ndarray         # (N, 5) player indices
    lineup_b: np.ndarray
    month: np.ndarray            # (N,)
    tm_a: np.ndarray             # (N,)
    tm_b: np.ndarray
    region_a: np.ndarray         # (N,)
    region_b: np.ndarray
    is_lan: np.ndarray           # (N,) 1.0 for a map played on LAN
    series: np.ndarray           # (N,)
    rounds: np.ndarray           # (N,) regulation rounds played
    rounds_a: np.ndarray         # (N,) won by A
    rating_player: np.ndarray    # (R,)
    rating_month: np.ndarray     # (R,)
    rating_lineup: np.ndarray    # (R, 5) the player's own lineup
    rating_dev: np.ndarray       # (R,) rating minus his team's mean on that map

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


def prepare_maps(history: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Decisive maps with both announced fives, regulation rounds and the month each was played in."""
    h = history[history.team_a_score != history.team_b_score].copy()
    h["day"] = pd.to_datetime(h.match_date).dt.normalize()
    fives = {k: g.sort_values("slot").player_id.astype(int).tolist()[:LINEUP_SIZE]
             for k, g in lineups.groupby(["match_id", "team_id"])}
    h["lineup_a"] = [fives.get((m, t)) for m, t in zip(h.match_id, h.team_a_id)]
    h["lineup_b"] = [fives.get((m, t)) for m, t in zip(h.match_id, h.team_b_id)]
    full = h.lineup_a.map(lambda x: x is not None and len(x) == LINEUP_SIZE) \
        & h.lineup_b.map(lambda x: x is not None and len(x) == LINEUP_SIZE)
    h = h[full].copy()
    overtime = h.team_a_score + h.team_b_score > 24
    h["rounds_a"] = np.where(overtime, 12, h.team_a_score)
    h["rounds"] = np.where(overtime, 24, h.team_a_score + h.team_b_score)
    h["month"] = h.day.dt.to_period("M").dt.to_timestamp()
    return h.reset_index(drop=True)


def build_fit_data(maps: pd.DataFrame, player_stats: pd.DataFrame, rank_strength: RankStrength,
                   start: pd.Timestamp, end: pd.Timestamp) -> FitData:
    """Inputs from the maps played in [start, end); `maps` comes from prepare_maps."""
    tr = maps[(maps.day >= start) & (maps.day < end)].sort_values(["day", "match_id"]).reset_index(drop=True)
    months = sorted(tr.month.unique())
    midx = {m: i for i, m in enumerate(months)}

    players = sorted({p for col in ("lineup_a", "lineup_b") for five in tr[col] for p in five})
    pidx = {p: i for i, p in enumerate(players)}
    debut: dict[int, tuple[int, pd.Timestamp]] = {}
    for r in tr.itertuples():
        for five, team in ((r.lineup_a, r.team_a_id), (r.lineup_b, r.team_b_id)):
            for p in five:
                debut.setdefault(p, (team, r.day))
    raw = np.array([rank_strength(*debut[p]) for p in players])
    # With every debut team unranked there is no spread to standardize by; the prior then rests on b alone.
    mu, sd = float(raw.mean()), float(raw.std()) or 1.0

    before_end = maps[maps.day < end]
    first_map = pd.concat([before_end[["day", "lineup_a"]].rename(columns={"lineup_a": "five"}),
                           before_end[["day", "lineup_b"]].rename(columns={"lineup_b": "five"})]).explode("five") \
        .groupby("five").day.min()
    grace = maps.day.min() + pd.Timedelta(days=DEBUT_GRACE_DAYS)

    keys = sorted(set(zip(tr.team_a_id.astype(int), tr.map_name)) | set(zip(tr.team_b_id.astype(int), tr.map_name)))
    tmidx = {k: i for i, k in enumerate(keys)}
    plays = collections.Counter(list(zip(tr.team_a_id.astype(int), tr.map_name)) + list(zip(tr.team_b_id.astype(int), tr.map_name)))
    home = home_regions(tr.drop_duplicates("match_id"))
    sidx = {m: i for i, m in enumerate(tr.match_id.unique())}

    ratings = _rating_rows(tr, player_stats, pidx)
    return FitData(
        months=months,
        player_ids=np.array(players, dtype=np.int64),
        strength=(raw - mu) / sd,
        strength_mu=mu,
        strength_sd=sd,
        debutant=np.array([first_map[p] >= max(start, grace) for p in players]),
        career_ids=first_map.index.to_numpy(np.int64),
        home=home,
        tm_team_ids=np.array([k[0] for k in keys], dtype=np.int64),
        tm_maps=np.array([k[1] for k in keys]),
        plays=pd.DataFrame([(t, m, c) for (t, m), c in plays.items()], columns=["team_id", "map_name", "maps"]),
        n_series=len(sidx),
        lineup_a=np.array([[pidx[p] for p in five] for five in tr.lineup_a]),
        lineup_b=np.array([[pidx[p] for p in five] for five in tr.lineup_b]),
        month=tr.month.map(midx).to_numpy(),
        tm_a=np.array([tmidx[(int(t), m)] for t, m in zip(tr.team_a_id, tr.map_name)]),
        tm_b=np.array([tmidx[(int(t), m)] for t, m in zip(tr.team_b_id, tr.map_name)]),
        region_a=np.array([home.get(int(t), 0) for t in tr.team_a_id]),
        region_b=np.array([home.get(int(t), 0) for t in tr.team_b_id]),
        is_lan=(pd.to_numeric(tr.is_lan, errors="coerce").fillna(0).to_numpy() == 1).astype(float),
        series=tr.match_id.map(sidx).to_numpy(),
        rounds=tr.rounds.to_numpy(),
        rounds_a=tr.rounds_a.to_numpy(),
        rating_player=ratings.player_id.map(pidx).to_numpy(),
        rating_month=ratings.month.map(midx).to_numpy(),
        rating_lineup=np.array([[pidx[p] for p in five] for five in ratings.own_lineup], dtype=np.int64).reshape(-1, LINEUP_SIZE),
        rating_dev=ratings.dev.to_numpy(float),
    )


def _rating_rows(tr: pd.DataFrame, player_stats: pd.DataFrame, pidx: dict[int, int]) -> pd.DataFrame:
    """One row per lineup player per map with an all-sides rating: the rating minus his team's mean on the map."""
    stats = player_stats[(player_stats.side == "all") & player_stats.rating.notna()]
    r = stats[["match_id", "map_name", "player_id", "team_id", "rating"]].merge(
        tr[["match_id", "map_name", "team_a_id", "team_b_id", "lineup_a", "lineup_b", "month"]], on=["match_id", "map_name"])
    r = r[(r.team_id == r.team_a_id) | (r.team_id == r.team_b_id)].copy()
    r["own_lineup"] = [a if t == ta else b for t, ta, a, b in zip(r.team_id, r.team_a_id, r.lineup_a, r.lineup_b)]
    r = r[[int(p) in five for p, five in zip(r.player_id, r.own_lineup)]]
    r = r[r.player_id.astype(int).isin(pidx)].copy()
    r["player_id"] = r.player_id.astype(int)
    r["dev"] = r.rating - r.groupby(["match_id", "map_name", "team_id"]).rating.transform("mean")
    return r.reset_index(drop=True)
