"""The time-ordered map stream, the player priors, and the driver that runs the filter between week cutoffs."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e05 import data as D
from gnomepy_research.sessions.cs2_comp.e05 import kernel as K

EPOCH = np.datetime64("1970-01-01")
FIRST_TIERS = ("entry", "open", "inter", "main", "adv", "prem", "org")


def per_source(default: float, **overrides: float) -> np.ndarray:
    return np.array([overrides.get(s, default) for s in D.SOURCES], dtype=float)


@dataclass(frozen=True)
class Config:
    # Fitted by one-step-ahead map log loss on the existing HLTV player-skill filter (models/cs2/hltv/player_skill.py).
    tau: float = 0.009319191322975635
    lam_best: float = 0.0761331162651603
    hltv_a: float = -0.1081114476105491
    hltv_b: float = 0.14696262712175345
    hltv_sigma0: float = 0.2536091079975324
    faceit_sigma0: float = 0.25
    faceit_a: dict = field(default_factory=lambda: {t: -0.1 for t in FIRST_TIERS})
    faceit_level_c: float = 0.1
    faceit_level_missing: float = 0.0
    lam: dict = field(default_factory=dict)
    weight: dict = field(default_factory=dict)
    rho: dict = field(default_factory=dict)
    rho_default: float = 0.017
    sigma_map: dict = field(default_factory=dict)
    sigma_map_default: float = 0.37
    perf_g: dict = field(default_factory=dict)
    perf_noise: dict = field(default_factory=dict)
    perf_noise_default: float = 0.06
    faceit: bool = True


@dataclass
class Stream:
    keys: np.ndarray           # (P,) sorted player keys
    day: np.ndarray            # (N,) int32 days since epoch
    source: np.ndarray
    rounds_a: np.ndarray
    rounds: np.ndarray
    win_a: np.ndarray
    side_a: np.ndarray         # (N, 5) compact player index
    side_b: np.ndarray
    perf_a: np.ndarray
    perf_b: np.ndarray
    link: np.ndarray           # (P,) compact index of the FACEIT record of an HLTV player, -1 when there is none
    first: pd.DataFrame        # per player: first_day, first_source, first_team, level
    hltv_points: pd.Series | None = None


def build_stream(maps: D.Maps, faceit: bool, links: dict[int, int]) -> Stream:
    order = np.argsort(maps.day, kind="stable")
    if not faceit:
        order = order[maps.source[order] == 0]
    take = lambda a: a[order]
    keys, inv = np.unique(np.concatenate([take(maps.side_a).ravel(), take(maps.side_b).ravel()]), return_inverse=True)
    n = len(order)
    side_a, side_b = inv[:n * D.TEAM_SIZE].reshape(n, D.TEAM_SIZE), inv[n * D.TEAM_SIZE:].reshape(n, D.TEAM_SIZE)
    day = (take(maps.day) - EPOCH).astype(np.int32)
    both = np.concatenate([side_a, side_b], axis=1)
    _, first_pos = np.unique(both.ravel(), return_index=True)
    row, col = first_pos // both.shape[1], first_pos % both.shape[1]
    team = np.where(col < D.TEAM_SIZE, take(maps.team_a)[row], take(maps.team_b)[row])
    first = pd.DataFrame({"first_day": day[row], "first_source": take(maps.source)[row], "first_team": team,
                          "level": np.nan}, index=keys)
    return Stream(keys=keys, day=day, source=take(maps.source), rounds_a=take(maps.rounds_a).astype(np.float64),
                  rounds=take(maps.rounds).astype(np.float64), win_a=take(maps.win_a).astype(np.int64), side_a=side_a.astype(np.int64), side_b=side_b.astype(np.int64),
                  perf_a=take(maps.perf_a), perf_b=take(maps.perf_b), link=_link_index(keys, links if faceit else {}), first=first)


def _link_index(keys: np.ndarray, links: dict[int, int]) -> np.ndarray:
    out = np.full(len(keys), -1, np.int64)
    hltv = np.array(sorted(links), dtype=np.int64)
    face = np.array([links[h] for h in hltv], dtype=np.int64)
    ph = np.searchsorted(keys, hltv)
    pf = np.searchsorted(keys, face)
    ok = (ph < len(keys)) & (pf < len(keys))
    ok[ok] = (keys[ph[ok]] == hltv[ok]) & (keys[pf[ok]] == face[ok])
    out[ph[ok]] = pf[ok]
    return out


def ranking_points(rankings: pd.DataFrame, teams: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Ranking points of each team in the latest ranking snapshot dated strictly before the day (0 when absent)."""
    date = pd.to_datetime(rankings["date"])
    r = rankings.assign(date=date.dt.tz_convert(None) if date.dt.tz is not None else date)
    dates = np.sort(r.date.unique())
    when = (EPOCH + days.astype("timedelta64[D]")).astype("datetime64[ns]")
    snap = np.searchsorted(dates, when, side="left") - 1
    wanted = pd.DataFrame({"team_id": teams, "date": np.where(snap >= 0, dates[np.maximum(snap, 0)], np.datetime64("NaT"))})
    pts = wanted.merge(r[["team_id", "date", "points"]].drop_duplicates(["team_id", "date"]), on=["team_id", "date"], how="left")
    return pts.points.fillna(0.0).to_numpy(float)


def player_priors(stream: Stream, cfg: Config, rankings: pd.DataFrame, levels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    f = stream.first
    mean = np.zeros(len(f))
    var = np.full(len(f), cfg.hltv_sigma0 ** 2)
    hltv = (f.first_source.to_numpy() == 0)
    points = ranking_points(rankings, f.first_team.to_numpy(), f.first_day.to_numpy())
    mean[hltv] = cfg.hltv_a + cfg.hltv_b * np.log1p(points[hltv])
    fa = ~hltv
    tiers = np.array([D.SOURCES[s] for s in f.first_source.to_numpy()[fa]])
    group = np.where(np.isin(tiers, FIRST_TIERS[:-1]), tiers, "org")
    lv = levels.reindex(f.index[fa])
    centered = np.where(lv.notna(), (lv.fillna(8.0).to_numpy() - 8.0) / 2.0, cfg.faceit_level_missing)
    mean[fa] = np.array([cfg.faceit_a[g] for g in group]) + cfg.faceit_level_c * centered
    var[fa] = cfg.faceit_sigma0 ** 2
    return mean, var


_INNER_X, _INNER_W = np.polynomial.hermite_e.hermegauss(21)
_INNER_W = _INNER_W / _INNER_W.sum()


@lru_cache(maxsize=256)
def _map_table(sigma: float) -> np.ndarray:
    grid = K.GRID_LO + K.GRID_STEP * np.arange(int(round(-2 * K.GRID_LO / K.GRID_STEP)) + 2)
    return regulation_map_win(grid[:, None] + sigma * _INNER_X[None, :]) @ _INNER_W


def map_table(sigma: float) -> np.ndarray:
    """P(win a regulation map) on the kernel's grid of round-logit gaps, with per-map noise of sd sigma integrated out."""
    return _map_table(round(float(sigma), 4))


class Filter:
    """Filter state plus the stream it runs over; advance(cutoff) processes every map from before that day."""

    def __init__(self, stream: Stream, cfg: Config, rankings: pd.DataFrame, levels: pd.Series):
        self.stream, self.cfg = stream, cfg
        p = len(stream.keys)
        self.mu, self.var = np.zeros(p), np.ones(p)
        self.seen = np.full(p, -1, np.int32)
        self.prior_mu, self.prior_var = player_priors(stream, cfg, rankings, levels)
        self.cov = K.new_cov()
        self.pos = 0
        self.ll = np.zeros(len(stream.day))
        self.src_lam = per_source(1.0, **cfg.lam)
        self.src_weight = per_source(1.0, **cfg.weight)
        self.src_rho = per_source(cfg.rho_default, **cfg.rho)
        self.src_g = per_source(0.0, **cfg.perf_g)
        self.src_noise = per_source(cfg.perf_noise_default, **cfg.perf_noise)
        self.tables = np.array([map_table(sd) for sd in per_source(cfg.sigma_map_default, **cfg.sigma_map)])

    def advance(self, cutoff: pd.Timestamp) -> None:
        stop = int(np.searchsorted(self.stream.day, (np.datetime64(cutoff.date()) - EPOCH).astype(np.int32), side="left"))
        s = self.stream
        K.run(self.mu, self.var, self.seen, self.prior_mu, self.prior_var, self.cov, s.day, s.source, s.rounds_a, s.rounds,
              s.win_a, s.side_a, s.side_b, s.perf_a, s.perf_b, self.pos, stop, self.cfg.tau ** 2, self.cfg.lam_best, self.src_lam,
              self.src_weight, self.src_rho, self.src_g, self.src_noise, self.tables, self.ll)
        self.pos = max(self.pos, stop)
