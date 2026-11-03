"""Sequential team and player Elo over HLTV maps with weekly snapshots.

State is advanced map by map in time order; at every Monday boundary the features of the coming week's series are read
from the state, so a series' features only ever see maps finished before its week_cutoff."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask

BASE = 1500.0
SCALE = 400.0 / np.log(10.0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def prepare_maps(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist[~showmatch_mask(hist)].copy()
    h["t"] = pd.to_datetime(h.match_time.fillna(h.match_date), utc=True).dt.tz_convert(None)
    h = h.dropna(subset=["team_a_won", "team_a_score", "team_b_score"])
    h = h[h.team_a_player_ids.map(len).eq(5) & h.team_b_player_ids.map(len).eq(5)]
    return h.sort_values(["t", "match_id", "map_position_in_series"], kind="stable").reset_index(drop=True)


def series_table(maps: pd.DataFrame) -> pd.DataFrame:
    g = maps.groupby("match_id", sort=False)
    s = g.agg(t=("t", "min"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
              wins_a=("team_a_won", "sum"), n=("team_a_won", "size"), bo=("bo_type", "first"),
              lan=("is_lan", "first"), tier=("event_tier", "first"),
              pa=("team_a_player_ids", "first"), pb=("team_b_player_ids", "first")).reset_index()
    s["y"] = (s.wins_a * 2 > s.n).astype(float)
    s.loc[s.wins_a * 2 == s.n, "y"] = np.nan
    return s


class Elo:
    def __init__(self, k_team=18.0, k_player=10.0, k_new=2.0, n_new=15.0, margin=0.6, halflife_days=0.0, reset_overlap=3):
        self.k_team, self.k_player, self.k_new, self.n_new = k_team, k_player, k_new, n_new
        self.margin, self.reset_overlap = margin, reset_overlap
        self.team = defaultdict(lambda: BASE)
        self.team_n = defaultdict(float)
        self.player = defaultdict(lambda: BASE)
        self.player_n = defaultdict(float)
        self.last_lineup: dict[int, frozenset] = {}
        self.last_seen: dict[int, pd.Timestamp] = {}

    def lineup_strength(self, ids) -> tuple[float, float]:
        r = np.fromiter((self.player[p] for p in ids), float, count=len(ids))
        n = np.fromiter((self.player_n[p] for p in ids), float, count=len(ids))
        return float(r.mean()), float(n.mean())

    def features(self, ta, tb, pa, pb) -> list[float]:
        ra, na = self.lineup_strength(pa)
        rb, nb = self.lineup_strength(pb)
        ova = len(self.last_lineup.get(ta, frozenset()) & frozenset(pa))
        ovb = len(self.last_lineup.get(tb, frozenset()) & frozenset(pb))
        return [(self.team[ta] - self.team[tb]) / SCALE, (ra - rb) / SCALE,
                np.log1p(self.team_n[ta]) - np.log1p(self.team_n[tb]),
                np.log1p(min(na, nb)), (ova - ovb) / 5.0, min(ova, ovb) / 5.0]

    def update(self, row) -> None:
        ta, tb = row.team_a_id, row.team_b_id
        pa, pb = list(row.team_a_player_ids), list(row.team_b_player_ids)
        ra, _ = self.lineup_strength(pa)
        rb, _ = self.lineup_strength(pb)
        for team, pl in ((ta, pa), (tb, pb)):
            prev = self.last_lineup.get(team)
            if prev is not None and len(prev & frozenset(pl)) < self.reset_overlap:
                mean_r, _ = self.lineup_strength(pl)
                self.team[team] = 0.5 * self.team[team] + 0.5 * mean_r
            self.last_lineup[team] = frozenset(pl)
        total = row.team_a_score + row.team_b_score
        s_margin = row.team_a_score / total if total > 0 else 0.5
        s = (1 - self.margin) * float(row.team_a_won) + self.margin * s_margin
        e_team = _sigmoid((self.team[ta] - self.team[tb]) / SCALE)
        e_play = _sigmoid((ra - rb) / SCALE)
        for team, sign in ((ta, 1.0), (tb, -1.0)):
            k = self.k_team * (1 + 3.0 / (1 + self.team_n[team] / 10.0))
            self.team[team] += sign * k * (s - e_team)
            self.team_n[team] += 1
        for pl, sign in ((pa, 1.0), (pb, -1.0)):
            for p in pl:
                k = self.k_player * (1 + 3.0 / (1 + self.player_n[p] / self.n_new))
                self.player[p] += sign * k * (s - e_play)
                self.player_n[p] += 1


def monday(ts: pd.Series) -> pd.Series:
    d = ts.dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def snapshot_features(maps: pd.DataFrame, queries: pd.DataFrame, params: dict | None = None) -> np.ndarray:
    """Features for each query row (columns: week, team_a_id, team_b_id, pa, pb) read from the state at its week start."""
    elo = Elo(**(params or {}))
    out = np.full((len(queries), 6), np.nan)
    order = np.argsort(queries.week.values, kind="stable")
    mt = maps.t.values
    i = 0
    n = len(maps)
    rows = maps.itertuples(index=False)
    cur = next(rows, None)
    for qi in order:
        w = np.datetime64(queries.week.iloc[qi])
        while cur is not None and cur.t.to_datetime64() < w:
            elo.update(cur)
            cur = next(rows, None)
        q = queries.iloc[qi]
        out[qi] = elo.features(q.team_a_id, q.team_b_id, list(q.pa), list(q.pb))
    return out
