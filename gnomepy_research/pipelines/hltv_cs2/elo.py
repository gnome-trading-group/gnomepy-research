"""
Causal Elo ladders for CS2 teams.

Three design choices that are load-bearing rather than cosmetic.

Ratings freeze per series. If Elo updated after map 1, a map-2 training row
would carry both elo_diff (already containing map 1's result) and
team_a_series_score=1 — collinear. Worse, the series DP prices hypothetical
future maps at nodes like (1,0) and cannot know the Elo update without
simulating it, so per-map updates would build a train/serve skew on purpose.
team_a_series_score already carries the within-series information.

Updates batch by day. match_date is date-granular and match_id is not monotone
in date — 237 of 243 days have a match_id range overlapping the next day's — so
ordering within a day would be arbitrary. Every series on a date is scored
against the ratings that stood at the start of that date.

Updates use round share, not the binary result. A map is 24+ Bernoulli trials
and 13-2 says more than 13-11; this also puts the round margin to use, which is
otherwise only a scrape sanity check.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnomepy_research.pipelines.hltv_cs2.pit import SeriesBatch, pit_scan
from gnomepy_research.pipelines.hltv_cs2.sides import regulation_side_splits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EloConfig:
    base: float = 1500.0
    scale: float = 400.0
    k_global: float = 24.0
    k_map: float = 16.0
    k_side: float = 12.0
    decay_tau_days: float = 365.0
    provisional_games: int = 30
    provisional_k_mult: float = 1.75
    seed_alpha: float = 120.0
    score_clip: float = 0.05


@dataclass
class _Rating:
    value: float
    last_seen: pd.Timestamp
    games: int = 0


def _expected(rating_a: float, rating_b: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


class EloEngine:
    """
    Ladders keyed on team_id. Call snapshot_* to read, observe_series to update,
    and never interleave the two within a day.
    """

    def __init__(self, cfg: EloConfig | None = None, rankings: pd.DataFrame | None = None):
        self.cfg = cfg or EloConfig()
        self._global: dict[int, _Rating] = {}
        self._map: dict[tuple[int, str], _Rating] = {}
        self._side: dict[tuple[int, int], _Rating] = {}
        self._seed_points = self._build_seed_table(rankings)

    def _build_seed_table(self, rankings: pd.DataFrame | None):
        if rankings is None or len(rankings) == 0:
            return None
        r = rankings.sort_values("date")
        median_by_date = r.groupby("date")["points"].median()
        return r, median_by_date

    def _seed(self, team_id: int, as_of: pd.Timestamp) -> float:
        """
        Start a new team from its published ranking points rather than the base.

        Cold-starting everyone at 1500 wastes the opening months of a nine-month
        dataset; ranking points are published before the match, so seeding is causal.
        """
        if self._seed_points is None:
            return self.cfg.base
        r, median_by_date = self._seed_points
        sub = r[(r["team_id"].values == team_id) & (r["date"].values < as_of)]
        if len(sub) == 0:
            return self.cfg.base
        latest = sub.iloc[-1]
        med = float(median_by_date.loc[latest["date"]])
        if med <= 0:
            return self.cfg.base
        return self.cfg.base + self.cfg.seed_alpha * (
            math.log1p(float(latest["points"])) - math.log1p(med)
        )

    def _decayed(self, rating: _Rating, as_of: pd.Timestamp) -> float:
        """Pull an unseen team back toward the base so stale elite ratings do not persist."""
        days = max(0.0, float((as_of - rating.last_seen).days))
        w = math.exp(-days / self.cfg.decay_tau_days)
        return self.cfg.base + w * (rating.value - self.cfg.base)

    def _get(self, store: dict, key, as_of: pd.Timestamp, seed_id: int | None = None) -> _Rating:
        if key not in store:
            seed = self._seed(seed_id, as_of) if seed_id is not None else self.cfg.base
            store[key] = _Rating(value=seed, last_seen=as_of, games=0)
        return store[key]

    def rating_global(self, team_id: int, as_of: pd.Timestamp) -> tuple[float, int]:
        r = self._get(self._global, team_id, as_of, seed_id=team_id)
        return self._decayed(r, as_of), r.games

    def rating_map(self, team_id: int, map_name: str, as_of: pd.Timestamp) -> float:
        r = self._get(self._map, (team_id, map_name), as_of, seed_id=team_id)
        return self._decayed(r, as_of)

    def rating_side(self, team_id: int, side: int, as_of: pd.Timestamp) -> float:
        r = self._get(self._side, (team_id, side), as_of, seed_id=team_id)
        return self._decayed(r, as_of)

    def _k(self, rating: _Rating, base_k: float) -> float:
        if rating.games < self.cfg.provisional_games:
            return base_k * self.cfg.provisional_k_mult
        return base_k

    def _update_pair(self, store, key_a, key_b, score_a: float, base_k: float,
                     as_of: pd.Timestamp, seed_a: int, seed_b: int) -> None:
        ra = self._get(store, key_a, as_of, seed_id=seed_a)
        rb = self._get(store, key_b, as_of, seed_id=seed_b)
        va, vb = self._decayed(ra, as_of), self._decayed(rb, as_of)
        exp_a = _expected(va, vb, self.cfg.scale)
        ka, kb = self._k(ra, base_k), self._k(rb, base_k)
        ra.value = va + ka * (score_a - exp_a)
        rb.value = vb + kb * ((1.0 - score_a) - (1.0 - exp_a))
        ra.last_seen = rb.last_seen = as_of
        ra.games += 1
        rb.games += 1

    def observe_series(self, series_rows: pd.DataFrame) -> None:
        """Apply every ladder update implied by one finished series."""
        cfg = self.cfg
        first = series_rows.iloc[0]
        id_a, id_b = int(first["team_a_id"]), int(first["team_b_id"])
        as_of = pd.Timestamp(first["match_date"])

        rounds_a = float(series_rows["team_a_score"].sum())
        rounds_total = float((series_rows["team_a_score"] + series_rows["team_b_score"]).sum())
        if rounds_total > 0:
            share = np.clip(rounds_a / rounds_total, cfg.score_clip, 1.0 - cfg.score_clip)
            self._update_pair(self._global, id_a, id_b, float(share), cfg.k_global, as_of, id_a, id_b)

        for row in series_rows.itertuples():
            total = float(row.team_a_score + row.team_b_score)
            if total <= 0:
                continue
            share = float(np.clip(row.team_a_score / total, cfg.score_clip, 1.0 - cfg.score_clip))
            self._update_pair(self._map, (id_a, row.map_name), (id_b, row.map_name),
                              share, cfg.k_map, as_of, id_a, id_b)

            if any(pd.isna(getattr(row, c, np.nan)) for c in
                   ("team_a_h1_score", "team_b_h1_score", "team_a_started_ct")):
                continue
            split = regulation_side_splits(
                int(row.team_a_score), int(row.team_b_score),
                int(row.team_a_h1_score), int(row.team_b_h1_score),
                bool(row.team_a_started_ct),
            )
            if not split.valid:
                continue
            ct_share = float(np.clip(split.team_a_ct_won / split.team_a_ct_played,
                                     cfg.score_clip, 1.0 - cfg.score_clip))
            t_share = float(np.clip(split.team_a_t_won / split.team_a_t_played,
                                    cfg.score_clip, 1.0 - cfg.score_clip))
            self._update_pair(self._side, (id_a, 1), (id_b, -1), ct_share, cfg.k_side, as_of, id_a, id_b)
            self._update_pair(self._side, (id_a, -1), (id_b, 1), t_share, cfg.k_side, as_of, id_a, id_b)

    def snapshot_series(self, series_rows: pd.DataFrame) -> dict:
        """Pre-series ratings, identical for every map of the series."""
        first = series_rows.iloc[0]
        id_a, id_b = int(first["team_a_id"]), int(first["team_b_id"])
        as_of = pd.Timestamp(first["match_date"])

        g_a, n_a = self.rating_global(id_a, as_of)
        g_b, n_b = self.rating_global(id_b, as_of)
        ct_a, t_a = self.rating_side(id_a, 1, as_of), self.rating_side(id_a, -1, as_of)
        ct_b, t_b = self.rating_side(id_b, 1, as_of), self.rating_side(id_b, -1, as_of)

        per_map = {
            m: self.rating_map(id_a, m, as_of) - self.rating_map(id_b, m, as_of)
            for m in series_rows["map_name"].unique()
        }
        return {
            "elo_a": g_a, "elo_b": g_b, "elo_diff": g_a - g_b,
            "elo_games_min": float(min(n_a, n_b)),
            "elo_side_asym": (ct_a - t_a) - (ct_b - t_b),
            "map_elo_diff": per_map,
        }


class _EloAccumulator:
    """Adapts EloEngine to the shared PIT driver, which enforces the freeze."""

    def __init__(self, cfg: EloConfig, rankings: pd.DataFrame | None):
        self._engine = EloEngine(cfg, rankings)

    def snapshot(self, batch: SeriesBatch) -> list[dict]:
        snap = self._engine.snapshot_series(batch.rows)
        out = []
        for row in batch.rows.itertuples():
            map_elo = snap["map_elo_diff"].get(row.map_name, 0.0)
            out.append({
                "match_id": batch.match_id,
                "map_name": row.map_name,
                "elo_a": snap["elo_a"],
                "elo_b": snap["elo_b"],
                "elo_diff": snap["elo_diff"],
                "elo_map_diff": map_elo,
                "elo_map_resid_diff": map_elo - snap["elo_diff"],
                "elo_side_asym": snap["elo_side_asym"],
                "elo_games_min": snap["elo_games_min"],
            })
        return out

    def update(self, batch: SeriesBatch) -> None:
        self._engine.observe_series(batch.rows)


def compute_elo_features(
    history: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
    cfg: EloConfig | None = None,
) -> pd.DataFrame:
    """Pre-series Elo features for every map row, keyed (match_id, map_name)."""
    cfg = cfg or EloConfig()
    return pit_scan(history, _EloAccumulator(cfg, rankings))
