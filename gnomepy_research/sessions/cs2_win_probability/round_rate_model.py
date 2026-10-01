"""
Per-round side rates: the CT/T split the map DP needs.

Inverting the Layer-1 map probability is one equation in two unknowns — it fixes
the level of team strength but says nothing about how that strength divides
between CT and T. This module supplies the split.

It is fitted on half scores rather than demos. Every map's h1 total is 12 (a map
cannot be won inside the first half), so h2 follows from the final score, and the
starting side attributes both halves — yielding 807,696 side-labelled rounds
across 16,827 maps, against 16,504 rounds on 762 maps from the demo corpus.

  logit(p_CT) = theta + delta        logit(p_T) = theta - delta
  delta       = logit(ct_rate[map]) + gamma * elo_side_asym

theta comes from the Layer-1 inversion; delta is what is fitted here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnomepy_research.pipelines.hltv_cs2.sides import build_side_round_dataset
from gnomepy_research.sessions.cs2_win_probability.map_dp import logit

logger = logging.getLogger(__name__)

_PRIOR_ROUNDS = 2000.0


@dataclass
class MapSideBias:
    """Per-map CT round-win rate, shrunk toward the pooled rate by sample size."""
    ct_rate: dict[str, float] = field(default_factory=dict)
    pooled: float = 0.5
    gamma: float = 0.0

    def delta(self, map_name: str, elo_side_asym: float = 0.0) -> float:
        rate = self.ct_rate.get(map_name, self.pooled)
        d = logit(rate)
        if self.gamma and np.isfinite(elo_side_asym):
            d += self.gamma * elo_side_asym
        return d

    def side_rate(self, map_name: str) -> float:
        return self.ct_rate.get(map_name, self.pooled)


def fit_map_side_bias(history: pd.DataFrame, prior_rounds: float = _PRIOR_ROUNDS) -> MapSideBias:
    """
    Estimate each map's CT round-win rate from half scores.

    Shrinkage is toward the pooled rate rather than 0.5, so a thinly-played map
    inherits the meta's overall CT lean instead of an artificial neutrality.
    """
    rounds = build_side_round_dataset(history)
    if rounds.empty:
        logger.warning("no side-attributed rounds available — defaulting to a neutral bias")
        return MapSideBias()

    ct = rounds[rounds.side == 1]
    pooled = float(ct.successes.sum() / ct.trials.sum())

    rates = {}
    for map_name, g in ct.groupby("map_name"):
        wins, trials = float(g.successes.sum()), float(g.trials.sum())
        rates[map_name] = (wins + prior_rounds * pooled) / (trials + prior_rounds)

    logger.info(
        "map side bias from %d side-labelled rounds across %d maps (pooled CT rate %.4f)",
        int(rounds.trials.sum()), rounds.groupby(["match_id", "map_name"]).ngroups, pooled,
    )
    return MapSideBias(ct_rate=rates, pooled=pooled)


def fit_side_asymmetry_gamma(
    history: pd.DataFrame,
    priors: pd.DataFrame,
    bias: MapSideBias,
) -> float:
    """
    Regress the residual CT lean on each matchup's Elo side asymmetry.

    Returns the coefficient; 0.0 when the signal is absent, which collapses delta
    back to the per-map rate alone.
    """
    rounds = build_side_round_dataset(history)
    if rounds.empty:
        return 0.0
    ct = rounds[(rounds.side == 1) & rounds.team_is_a].merge(
        priors[["match_id", "map_name", "elo_side_asym"]], on=["match_id", "map_name"], how="inner"
    )
    ct = ct[np.isfinite(ct.elo_side_asym) & (ct.trials > 0)]
    if len(ct) < 500:
        return 0.0

    observed = np.clip(ct.successes / ct.trials, 0.02, 0.98)
    residual = np.array([logit(v) for v in observed]) - np.array(
        [bias.delta(m) for m in ct.map_name]
    )
    x = ct.elo_side_asym.to_numpy(dtype=float)
    gamma = float(np.dot(x, residual) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0
    logger.info("side-asymmetry gamma = %.6f (n=%d)", gamma, len(ct))
    return gamma
