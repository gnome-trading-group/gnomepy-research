"""Sequential player ratings on HLTV maps, snapshotted at every Monday 00:00 UTC.

Each map moves the ten players' ratings by the round-share surprise: a five's strength is the mean of its players'
ratings in round-logit units, so sigmoid(R_A - R_B) is the expected share of rounds for A. A player's step size shrinks
with the maps the player has played, so newcomers find their level fast and veterans stay put. Because every update
uses only maps already played, a snapshot is point-in-time for everything after it, which gives out-of-sample skill
prices for every past series in a single pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask

FIVE = 5


@dataclass(frozen=True)
class EloConfig:
    k0: float = 0.6
    n0: float = 8.0
    k_min: float = 0.06
    newcomer: float = -0.15
    # Ratings drift back toward the newcomer level while a player is idle: a long break makes a rating less certain and
    # rusty players tend to come back weaker.
    idle_half_life_days: float = 720.0


def _ids(v) -> tuple[int, ...]:
    if v is None or isinstance(v, float):
        return ()
    return tuple(int(x) for x in v)


def map_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Rated HLTV maps in play order: both fives complete and a round score."""
    h = history[~showmatch_mask(history)].copy()
    h["time"] = pd.to_datetime(h.match_time.fillna(h.match_date))
    if h.time.dt.tz is not None:
        h["time"] = h.time.dt.tz_convert(None)
    h["fa"] = h.team_a_player_ids.map(_ids)
    h["fb"] = h.team_b_player_ids.map(_ids)
    ok = (h.fa.map(len) == FIVE) & (h.fb.map(len) == FIVE) & h.team_a_score.notna() & h.team_b_score.notna()
    ok &= (h.team_a_score + h.team_b_score) > 0
    h = h[ok]
    return h.sort_values(["time", "match_id", "map_position_in_series"], kind="stable").reset_index(drop=True)


class Ratings:
    """Runs the ratings over maps and keeps a snapshot at each requested cutoff."""

    def __init__(self, maps: pd.DataFrame, cutoffs: list[pd.Timestamp], cfg: EloConfig):
        self.cfg = cfg
        fa = np.array(maps.fa.tolist(), dtype=np.int64).reshape(-1, FIVE)
        fb = np.array(maps.fb.tolist(), dtype=np.int64).reshape(-1, FIVE)
        self.players = np.unique(np.concatenate([fa.ravel(), fb.ravel()]))
        ia = np.searchsorted(self.players, fa)
        ib = np.searchsorted(self.players, fb)
        P = len(self.players)
        r = np.full(P, cfg.newcomer)
        n = np.zeros(P)
        last = np.full(P, np.nan)
        times = maps.time.to_numpy().astype("datetime64[s]").astype(np.int64) / 86400.0
        share = (maps.team_a_score / (maps.team_a_score + maps.team_b_score)).to_numpy(float)
        cutoffs = sorted(pd.Timestamp(c) for c in cutoffs)
        cut_days = [c.to_datetime64().astype("datetime64[s]").astype(np.int64) / 86400.0 for c in cutoffs]
        self.snap: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        decay = np.log(2) / cfg.idle_half_life_days
        j = 0
        for i in range(len(maps)):
            while j < len(cut_days) and cut_days[j] <= times[i]:
                self.snap[cutoffs[j]] = (r.copy(), n.copy(), last.copy())
                j += 1
            pa, pb = ia[i], ib[i]
            both = np.concatenate([pa, pb])
            gap = np.where(np.isnan(last[both]), 0.0, times[i] - last[both])
            keep = np.exp(-decay * gap)
            r[both] = cfg.newcomer + (r[both] - cfg.newcomer) * keep
            d = r[pa].mean() - r[pb].mean()
            surprise = share[i] - 1.0 / (1.0 + np.exp(-d))
            k = np.maximum(cfg.k0 / (1.0 + n[both] / cfg.n0), cfg.k_min)
            r[pa] += k[:FIVE] * surprise
            r[pb] -= k[FIVE:] * surprise
            n[both] += 1
            last[both] = times[i]
        while j < len(cut_days):
            self.snap[cutoffs[j]] = (r.copy(), n.copy(), last.copy())
            j += 1
        self.cut_days = dict(zip(cutoffs, cut_days))

    def five(self, cutoff: pd.Timestamp, five: tuple[int, ...]) -> tuple[float, int, float]:
        """Mean rating of a five at a cutoff, how many of its players have fewer than 10 maps, and mean idle days."""
        r, n, last = self.snap[pd.Timestamp(cutoff)]
        now = self.cut_days[pd.Timestamp(cutoff)]
        decay = np.log(2) / self.cfg.idle_half_life_days
        vals, thin, idle = [], 0, []
        for p in five:
            i = np.searchsorted(self.players, p)
            if i < len(self.players) and self.players[i] == p and n[i] > 0:
                gap = now - last[i]
                vals.append(self.cfg.newcomer + (r[i] - self.cfg.newcomer) * np.exp(-decay * gap))
                thin += int(n[i] < 10)
                idle.append(gap)
            else:
                vals.append(self.cfg.newcomer)
                thin += 1
                idle.append(365.0)
        return float(np.mean(vals)), thin, float(np.mean(idle))
