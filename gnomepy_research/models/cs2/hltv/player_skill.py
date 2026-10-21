"""
Player skill over time, aggregated to the announced lineup.

Team Elo learns a roster only from that roster's own results, so a rebuilt or
unsettled five starts nearly blind; player Elo, form EWMAs and five-together
counts were each a hand-tuned patch for that. This replaces them with one
state-space model whose settings are fitted by likelihood (an extended Kalman
filter, as in arXiv 2308.02414; squad effects as in TrueSkill 2).

Skill is in round-logit units: a lineup gap D gives each round to team A with
probability sigmoid(D). Each player is Gaussian; variance grows tau^2 per idle
day, and a newcomer starts at N(a + b*log1p(debut team's ranking points),
sigma0^2). A lineup is the mean of its players plus lam on the best one, plus
chemistry kappa_max*(1 - exp(-c/c0)) for c maps this exact five played in the
last chem_days, minus delta per announced stand-in.

Covariances are kept between players who have shared a map. A fixed five only
ever reveals its sum, so each player's own variance barely shrinks while
teammates' errors turn negatively correlated; without those terms the lineup's
variance never falls and the filter degenerates to fixed-K Elo. Each map
updates the 10x10 block of the players in it and leaves covariance with
everyone else alone (a local approximation).

The rounds of a map are not independent trials, so the binomial gradient and
curvature are divided by 1 + (n - 1)*rho, the beta-binomial variance inflation.

Predictions read the announced lineup; updates use the players who actually
played. A missing announced lineup gives NaN rather than falling back to the
players who played, which are only known afterwards.
"""
from __future__ import annotations

import bisect
import collections
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom

from gnomepy_research.models.cs2.hltv.pit import SeriesBatch, pit_scan

logger = logging.getLogger(__name__)

LINEUPS_AUX = "lineups"

_PAIRED = ("strength", "chem", "sd", "new_players", "best", "worst", "five_maps")
PLAYER_SKILL_FEATURES = (
    ["ps_p_map", "ps_p_series", "ps_strength_diff", "ps_gap_sd"]
    + [f"team_{side}_ps_{k}" for side in "ab" for k in _PAIRED]
)

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(7)
_GH_W = _GH_W / _GH_W.sum()
_ROUNDS = np.arange(25)


@dataclass(frozen=True)
class PlayerSkillConfig:
    """Fitted by one-step-ahead map log loss on Sep-Dec 2025 after a Jul-Aug warm-up (0.6496)."""

    tau: float = 0.006305591805969653
    sigma0: float = 0.3032541994005592
    a: float = -0.10680695217179037
    b: float = 0.058062780790403466
    lam: float = 0.10507536756845584
    kappa_max: float = 0.05302967232099904
    c0: float = 16.862178105713124
    delta: float = 0.03663038889879238
    rho: float = 0.013077404265598343
    chem_days: int = 180
    new_player_maps: int = 20


def regulation_map_win(d: np.ndarray) -> np.ndarray:
    """P(win a regulation-24 map) for each round-logit gap in d (any shape); 12-12 counts as a coin flip."""
    d = np.asarray(d, dtype=float)
    pi = 1 / (1 + np.exp(-d))
    pmf = binom.pmf(_ROUNDS.reshape((-1,) + (1,) * d.ndim), 24, pi[None])
    return pmf[13:].sum(axis=0) + 0.5 * pmf[12]


def map_win_prob(d_mean: float, d_var: float) -> float:
    """P(win a regulation-24 map) for a round-logit gap ~ N(d_mean, d_var)."""
    return float(regulation_map_win(d_mean + math.sqrt(max(d_var, 0.0)) * _GH_X) @ _GH_W)


def _bo3(p: float) -> float:
    return p * p * (3 - 2 * p)


def _ids(value) -> tuple[int, ...]:
    if value is None or isinstance(value, float):
        return ()
    return tuple(int(p) for p in value)


@dataclass
class _Lineup:
    players: list[int]
    mus: list[float]
    variances: list[float]
    weights: list[float]
    strength: float
    kappa: float
    five_maps: float
    total: float


class PlayerSkillFilter:
    """Per-player Gaussian skill. Read with lineup/gap, update with observe_map; never interleave within a day."""

    def __init__(self, cfg: PlayerSkillConfig | None = None, rankings: pd.DataFrame | None = None):
        self.cfg = cfg or PlayerSkillConfig()
        self._mu: dict[int, float] = {}
        self._var: dict[int, float] = {}
        self._seen: dict[int, pd.Timestamp] = {}
        self._cov: dict[tuple[int, int], float] = {}
        self.games: collections.Counter = collections.Counter()
        self._five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self._rank_dates: list[pd.Timestamp] = []
        self._rank_points: list[dict[int, float]] = []
        if rankings is not None and len(rankings):
            r = rankings.assign(date=pd.to_datetime(rankings["date"]))
            for day, g in r.groupby("date", sort=True):
                self._rank_dates.append(pd.Timestamp(day))
                self._rank_points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def _points(self, team_id: int, day: pd.Timestamp) -> float:
        i = bisect.bisect_left(self._rank_dates, day) - 1
        return self._rank_points[i].get(team_id, 0.0) if i >= 0 else 0.0

    def _state(self, player: int, team_id: int, day: pd.Timestamp) -> tuple[float, float]:
        if player not in self._mu:
            return self.cfg.a + self.cfg.b * math.log1p(self._points(team_id, day)), self.cfg.sigma0 ** 2
        idle = max((day - self._seen[player]).days, 0)
        return self._mu[player], self._var[player] + self.cfg.tau ** 2 * idle

    def _five_maps(self, five: tuple[int, ...], day: pd.Timestamp) -> float:
        q = self._five_days.get(frozenset(five))
        if not q:
            return 0.0
        cutoff = day - pd.Timedelta(days=self.cfg.chem_days)
        return float(sum(1 for d in q if d >= cutoff))

    def _weights(self, mus: list[float]) -> list[float]:
        n = len(mus)
        best = int(np.argmax(mus))
        return [1 / n + self.cfg.lam * ((1.0 if i == best else 0.0) - 1 / n) for i in range(n)]

    def _block(self, players: list[int], variances: list[float]) -> np.ndarray:
        cov = np.diag(variances)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p, q = sorted((players[i], players[j]))
                cov[i, j] = cov[j, i] = self._cov.get((p, q), 0.0)
        return cov

    def lineup(self, players: tuple[int, ...], team_id: int, standins: float, day: pd.Timestamp) -> _Lineup:
        states = [self._state(p, team_id, day) for p in players]
        mus = [s[0] for s in states]
        weights = self._weights(mus)
        five_maps = self._five_maps(players, day) if len(players) == 5 else 0.0
        kappa = self.cfg.kappa_max * (1 - math.exp(-five_maps / self.cfg.c0))
        strength = sum(w * m for w, m in zip(weights, mus))
        return _Lineup(list(players), mus, [s[1] for s in states], weights, strength, kappa, five_maps,
                       strength + kappa - self.cfg.delta * standins)

    def gap(self, a: _Lineup, b: _Lineup) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Mean and variance of S_a - S_b, the ten players' joint covariance, and the weight vector."""
        w = np.array(a.weights + [-x for x in b.weights])
        cov = self._block(a.players + b.players, a.variances + b.variances)
        return a.total - b.total, float(w @ cov @ w), cov, w

    def lineup_sd(self, lineup: _Lineup) -> float:
        w = np.array(lineup.weights)
        return math.sqrt(max(float(w @ self._block(lineup.players, lineup.variances) @ w), 0.0))

    def observe_map(self, day: pd.Timestamp, team_a: int, team_b: int, players_a: tuple[int, ...],
                    players_b: tuple[int, ...], rounds_a: int, rounds_b: int) -> None:
        if not players_a or not players_b:
            return
        a = self.lineup(players_a, team_a, 0.0, day)
        b = self.lineup(players_b, team_b, 0.0, day)
        players = a.players + b.players
        if len(set(players)) != len(players):
            return
        d, s2, cov, w = self.gap(a, b)
        n = rounds_a + rounds_b
        pi = 1 / (1 + math.exp(-d))
        inflation = 1 + (n - 1) * self.cfg.rho
        grad, curv = (rounds_a - n * pi) / inflation, n * pi * (1 - pi) / inflation
        cw = cov @ w
        denom = 1 + curv * s2
        mus = np.array(a.mus + b.mus) + cw * grad / denom
        cov = cov - np.outer(cw, cw) * curv / denom
        for i, p in enumerate(players):
            self._mu[p] = float(mus[i])
            self._var[p] = max(1e-6, float(cov[i, i]))
            self._seen[p] = day
            self.games[p] += 1
            for j in range(i + 1, len(players)):
                self._cov[tuple(sorted((p, players[j])))] = float(cov[i, j])
        cutoff = day - pd.Timedelta(days=self.cfg.chem_days)
        for five in (players_a, players_b):
            if len(five) == 5:
                q = self._five_days[frozenset(five)]
                q.append(day)
                while q and q[0] < cutoff:
                    q.popleft()


_NAN_ROW = {name: np.nan for name in PLAYER_SKILL_FEATURES}


class _PlayerSkillAccumulator:
    def __init__(self, cfg: PlayerSkillConfig, rankings: pd.DataFrame | None):
        self.f = PlayerSkillFilter(cfg, rankings)

    def snapshot(self, batch: SeriesBatch) -> dict:
        first = batch.records[0]
        lineup_a, lineup_b = _ids(first.team_a_lineup_ids), _ids(first.team_b_lineup_ids)
        if not lineup_a or not lineup_b:
            return dict(_NAN_ROW)
        team_a, team_b = int(first.team_a_id), int(first.team_b_id)
        standins = collections.Counter()
        for r in batch.aux_rows(LINEUPS_AUX):
            standins[int(r.team_id)] += int(r.is_standin == 1)
        f, day = self.f, batch.match_date
        a = f.lineup(lineup_a, team_a, float(standins[team_a]), day)
        b = f.lineup(lineup_b, team_b, float(standins[team_b]), day)
        d, s2, _, _ = f.gap(a, b)
        p_map = map_win_prob(d, s2)
        row = {"ps_p_map": p_map, "ps_p_series": _bo3(p_map), "ps_strength_diff": a.strength - b.strength,
               "ps_gap_sd": math.sqrt(max(s2, 0.0))}
        for side, lu in (("a", a), ("b", b)):
            row.update({
                f"team_{side}_ps_strength": lu.strength,
                f"team_{side}_ps_chem": lu.kappa,
                f"team_{side}_ps_sd": f.lineup_sd(lu),
                f"team_{side}_ps_new_players": float(sum(f.games[p] < f.cfg.new_player_maps for p in lu.players)),
                f"team_{side}_ps_best": max(lu.mus),
                f"team_{side}_ps_worst": min(lu.mus),
                f"team_{side}_ps_five_maps": lu.five_maps,
            })
        return row

    def update(self, batch: SeriesBatch) -> None:
        for r in batch.records:
            rounds_a, rounds_b = r.team_a_score, r.team_b_score
            # Upcoming maps are priced alongside history and have no result yet.
            if pd.isna(rounds_a) or pd.isna(rounds_b) or rounds_a + rounds_b <= 0:
                continue
            self.f.observe_map(batch.match_date, int(r.team_a_id), int(r.team_b_id), _ids(r.team_a_player_ids),
                               _ids(r.team_b_player_ids), int(rounds_a), int(rounds_b))


def compute_player_skill_features(
    history: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    cfg: PlayerSkillConfig | None = None,
) -> pd.DataFrame:
    """Pre-series player-skill features for every map row, keyed (match_id, map_name)."""
    if "team_a_lineup_ids" not in history.columns or "team_a_player_ids" not in history.columns:
        logger.warning("lineup or player ids absent — player skill unavailable, emitting no rows")
        return pd.DataFrame(columns=["match_id", "map_name"] + PLAYER_SKILL_FEATURES)
    aux = {LINEUPS_AUX: lineups} if lineups is not None else None
    return pit_scan(history, _PlayerSkillAccumulator(cfg or PlayerSkillConfig(), rankings), aux=aux)
