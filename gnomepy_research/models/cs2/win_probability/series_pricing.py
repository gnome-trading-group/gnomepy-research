"""
Pre-veto prices for upcoming BO3 series: series winner and map 1 winner.

Live strategies price cs2_upcoming_features rows with this, so it is production code.

Before kickoff the veto is not public, so every map of the series is priced as an
unknown map: the pre-veto model is asked for P(team A wins the next map) at each
series score, and the series DP composes those into the series price. Map 1 is
the lattice's (0, 0) node. This is the construction the walk-forward backtest
scored (holding/preveto_checks.py).

price_blended is the production series price: this pre-map price blended with the Bayesian
series model's (models/cs2/bayes), which prices only once both lineups are known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import LINEUP_SIZE, RankStrength, SeriesBlend, SeriesPosterior
from gnomepy_research.models.cs2.win_probability.pre_map_model import CS2PreMapModel, _matrix
from gnomepy_research.models.cs2.win_probability.series_dp import series_nodes, series_win_prob

MAPS_TO_WIN = 2
PROB_CLIP = (0.01, 0.99)


def node_frame(priors_row: pd.Series) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """One feature row per series score, carrying the series-level features of the match."""
    rows, nodes = [], []
    for a, b in series_nodes(MAPS_TO_WIN):
        row = priors_row.to_dict()
        position = a + b + 1
        row.update({"team_a_series_score": float(a), "team_b_series_score": float(b),
                    "map_position_in_series": float(position), "map_name": "",
                    "is_decider": float(position == 2 * MAPS_TO_WIN - 1)})
        rows.append(row)
        nodes.append((a, b))
    return pd.DataFrame(rows), nodes


def price(model: CS2PreMapModel, priors: pd.DataFrame) -> pd.DataFrame:
    """
    priors: one row per upcoming series (build_priors_for output).
    Returns match_id, series, game1 - each P(HLTV team A wins), clipped.
    """
    if priors.empty:
        return pd.DataFrame(columns=["match_id", "series", "game1"])
    frames, nodes = zip(*(node_frame(r) for _, r in priors.iterrows()))
    p = model.predict_batch(_matrix(pd.concat(frames, ignore_index=True), model.feature_names))
    out, cur = [], 0
    for (_, r), f, nd in zip(priors.iterrows(), frames, nodes):
        lattice = dict(zip(nd, p[cur:cur + len(f)]))
        cur += len(f)
        out.append({"match_id": int(r.match_id),
                    "series": series_win_prob(lattice, MAPS_TO_WIN, n_veto_maps=2 * MAPS_TO_WIN - 1),
                    "game1": lattice[(0, 0)]})
    df = pd.DataFrame(out)
    df[["series", "game1"]] = np.clip(df[["series", "game1"]], *PROB_CLIP)
    return df


def _five(value) -> list[int] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or len(value) != LINEUP_SIZE:
        return None
    return [int(p) for p in value]


def price_blended(model: CS2PreMapModel, posterior: SeriesPosterior, blend: SeriesBlend, ranks: RankStrength,
                  priors: pd.DataFrame) -> pd.DataFrame:
    """
    priors: upcoming series rows with team ids, match_date and both lineups (team_a/b_lineup_ids).
    Returns match_id, series (the blend), xgb and bayes - each P(HLTV team A wins the series). A row
    without both full lineups has no Bayesian price, so its series price is NaN.
    """
    xgb = price(model, priors)["series"].to_numpy()
    bayes, blended = np.full(len(priors), np.nan), np.full(len(priors), np.nan)
    for i, r in enumerate(priors.itertuples()):
        five_a, five_b = _five(r.team_a_lineup_ids), _five(r.team_b_lineup_ids)
        if five_a is None or five_b is None:
            continue
        day = pd.Timestamp(r.match_date)
        bayes[i] = posterior.price(r.team_a_id, five_a, ranks(r.team_a_id, day), r.team_b_id, five_b, ranks(r.team_b_id, day))
        blended[i] = blend.predict(np.array([bayes[i]]), np.array([xgb[i]]))[0]
    return pd.DataFrame({"match_id": priors["match_id"].astype(int).to_numpy(), "series": np.clip(blended, *PROB_CLIP),
                         "xgb": xgb, "bayes": bayes})
