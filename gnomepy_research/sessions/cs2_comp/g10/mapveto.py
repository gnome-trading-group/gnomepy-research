"""Map-pool pricing: per (team, map) strength offsets on top of the player-skill gap, and a bo3 veto that lets each
side ban its worst maps and pick its best.

The offsets are one shrunken Newton step from zero: each recent HLTV map a team played contributes its round residual
against the skill fit's expectation, weighted by recency, and a prior precision pulls thinly played maps to zero. The
active pool is the most played maps over the recent window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.bundle import load

POOL_SIZE = 7


@dataclass(frozen=True)
class MapConfig:
    half_life_days: float = 120.0
    window_days: int = 365
    prior_rounds: float = 60.0
    pool_days: int = 60
    form_half_life_days: float = 21.0


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def load_maps(data_dir, until: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna() & (h.team_a_score != h.team_b_score)]
    h = h.assign(time=_naive(h.match_time.fillna(h.match_date)))
    sa, sb = h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int)
    ot = sa + sb > 24
    return pd.DataFrame({"time": h.time.to_numpy(), "team_a": h.team_a_id.astype(np.int64).to_numpy(),
                         "team_b": h.team_b_id.astype(np.int64).to_numpy(), "map_name": h.map_name.to_numpy(),
                         "rounds": np.where(ot, 24, sa + sb), "rounds_a": np.where(ot, 12, sa),
                         "five_a": h.team_a_player_ids.to_numpy(), "five_b": h.team_b_player_ids.to_numpy(),
                         "is_lan": h.is_lan.fillna(0).astype(int).to_numpy()})


def _five_skill(fit, fives, ranks=None) -> np.ndarray:
    out = np.full(len(fives), np.nan)
    for i, five in enumerate(fives):
        if five is None or len(five) != 5:
            continue
        vals = []
        for p in five:
            k = fit.player(int(p))
            vals.append(k[0] if k is not None else fit.unknown_prior(300.0))
        out[i] = np.mean(vals)
    return out


class MapModel:
    def __init__(self, maps: pd.DataFrame, fit, cutoff: pd.Timestamp, cfg: MapConfig):
        m = maps[(maps.time < cutoff) & (maps.time >= cutoff - pd.Timedelta(days=cfg.window_days))]
        recent = m[m.time >= cutoff - pd.Timedelta(days=cfg.pool_days)].map_name.value_counts()
        self.pool = list(recent.index[:POOL_SIZE])
        d = _five_skill(fit, m.five_a) - _five_skill(fit, m.five_b)
        ok = ~np.isnan(d)
        m, d = m[ok], d[ok]
        p = expit(d)
        age = (cutoff - m.time).dt.total_seconds().to_numpy() / 86400.0
        w = 0.5 ** (age / cfg.half_life_days)
        n = m.rounds.to_numpy(float)
        r = m.rounds_a.to_numpy(float) - n * p
        info = w * n * p * (1 - p)
        long = pd.DataFrame({"team": np.r_[m.team_a, m.team_b], "map": np.r_[m.map_name, m.map_name],
                             "lan": np.r_[m.is_lan, m.is_lan], "g": np.r_[w * r, -w * r], "h": np.r_[info, info]})
        prior = cfg.prior_rounds * 0.25
        agg = long.groupby(["team", "map"])[["g", "h"]].sum()
        self.offset = (agg.g / (agg.h + prior)).to_dict()
        agg = long.groupby(["team", "lan"])[["g", "h"]].sum()
        self.lan_offset = (agg.g / (agg.h + prior)).to_dict()
        agg = long.groupby("team")[["g", "h"]].sum()
        self.team_offset = (agg.g / (agg.h + prior)).to_dict()
        recent = age <= 3 * cfg.form_half_life_days
        wf = 0.5 ** (age[recent] / cfg.form_half_life_days)
        rr, inf = r[recent] , (n * p * (1 - p))[recent]
        form = pd.DataFrame({"team": np.r_[m.team_a.to_numpy()[recent], m.team_b.to_numpy()[recent]],
                             "g": np.r_[wf * rr, -wf * rr], "h": np.r_[wf * inf, wf * inf]}).groupby("team")[["g", "h"]].sum()
        self.form_offset = (form.g / (form.h + prior)).to_dict()
        lo, hi = np.minimum(m.team_a, m.team_b).to_numpy(), np.maximum(m.team_a, m.team_b).to_numpy()
        sign = np.where(m.team_a.to_numpy() == lo, 1.0, -1.0)
        pair = pd.DataFrame({"lo": lo, "hi": hi, "g": sign * w * r, "h": info}).groupby(["lo", "hi"])[["g", "h"]].sum()
        self.pair_offset = (pair.g / (pair.h + prior)).to_dict()

    def pair_gap(self, gap: float, team_a: int, team_b: int) -> float:
        """The skill gap plus the residual of past meetings between these two teams."""
        lo, hi = min(team_a, team_b), max(team_a, team_b)
        o = self.pair_offset.get((lo, hi), 0.0)
        return gap + (o if team_a == lo else -o)

    def offset_gap(self, gap: float, team_a: int, team_b: int, which: str) -> float:
        o = getattr(self, f"{which}_offset")
        return gap + o.get(team_a, 0.0) - o.get(team_b, 0.0)

    def lan_gap(self, gap: float, team_a: int, team_b: int, is_lan: int) -> float:
        """The skill gap plus each team's residual in this setting (LAN or online)."""
        return gap + self.lan_offset.get((team_a, is_lan), 0.0) - self.lan_offset.get((team_b, is_lan), 0.0)

    def series_price(self, gap: float, team_a: int, team_b: int) -> float:
        """Average of the two veto orders (either side banning first)."""
        maps = self.pool
        if len(maps) < 3:
            p = regulation_map_win(np.array([gap]))[0]
            return float(p * p * (3 - 2 * p))
        adv = np.array([gap + self.offset.get((team_a, mp), 0.0) - self.offset.get((team_b, mp), 0.0) for mp in maps])
        pw = regulation_map_win(adv)
        return 0.5 * (_veto(pw, True) + _veto(pw, False))


def _veto(pw: np.ndarray, a_first: bool) -> float:
    """bo3 veto ban, ban, pick, pick, ban, ban, decider; A maximises pw and B minimises it."""
    left = list(range(len(pw)))
    picks = []
    order = ["ban", "ban", "pick", "pick", "ban", "ban"]
    who = [a_first, not a_first] * 3
    for act, is_a in zip(order, who):
        if len(left) <= 1:
            break
        vals = pw[left]
        if act == "ban":
            j = int(np.argmin(vals)) if is_a else int(np.argmax(vals))
            left.pop(j)
        else:
            j = int(np.argmax(vals)) if is_a else int(np.argmin(vals))
            picks.append(left.pop(j))
    while len(picks) < 2 and left:
        picks.append(left.pop(0))
    decider = pw[left[0]] if left else pw[picks[-1]]
    p1, p2 = pw[picks[0]], pw[picks[1]]
    return float(p1 * p2 + (p1 * (1 - p2) + (1 - p1) * p2) * decider)
