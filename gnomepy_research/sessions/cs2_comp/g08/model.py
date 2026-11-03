"""g08's series price: the recurrent skill state read out at each weekly cutoff, a logistic layer refitted every week on
past HLTV bo3 series, and an even logit average with the frozen e12 price.

The layer sees one feature, logit of the veto-simulated series price from the state, and is fitted on both orientations
of every past series (no intercept), so swapping the teams flips the price. It is refitted each week on the series
known before that week's cutoff from the previous LAYER_WINDOW_DAYS, which keeps the price-to-outcome scale current as
the state's own scale drifts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp.g08.data import Inputs
from gnomepy_research.sessions.cs2_comp.g08.ratings import RatingConfig, run

CONFIG = RatingConfig(market_var=0.3, map_var=0.003)
HISTORY_START = pd.Timestamp("2024-06-01")
LAYER_WINDOW_DAYS = 540
E12_WEIGHT = 0.5


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _queries(inputs: Inputs, targets: pd.DataFrame) -> pd.DataFrame:
    s = inputs.series
    s = s[(s.bo_type == 3) & (s.time >= HISTORY_START)]
    hist = pd.DataFrame({"qid": "h" + s.match_id.astype(str), "match_id": s.match_id.to_numpy(), "cutoff": s.cutoff.to_numpy(),
                         "known": s.known.to_numpy(), "y": s.y.to_numpy(float), "team_a": s.team_a.to_numpy(),
                         "team_b": s.team_b.to_numpy(), "five_a": s.five_a.to_numpy(), "five_b": s.five_b.to_numpy(),
                         "target": False})
    tg = pd.DataFrame({"qid": "t" + targets.match_id.astype(str), "match_id": targets.match_id.to_numpy(),
                       "cutoff": targets.week_cutoff.to_numpy(), "known": targets.week_cutoff.to_numpy(), "y": np.nan,
                       "team_a": targets.team_a_id.to_numpy(np.int64), "team_b": targets.team_b_id.to_numpy(np.int64),
                       "five_a": targets.team_a_lineup_ids.to_numpy(), "five_b": targets.team_b_lineup_ids.to_numpy(),
                       "target": True})
    return pd.concat([hist, tg], ignore_index=True)


def state_prices(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """match_id, week_cutoff, q_state for every target: the layer over the recurrent state, fitted per week."""
    inputs = Inputs(data_dir, targets.week_cutoff.max(), faceit=True, market=True)
    Q = _queries(inputs, targets)
    feats = run(inputs.maps, Q, CONFIG)
    F = Q.merge(feats[["qid", "series_veto"]], on="qid")
    x = logit(F.series_veto.to_numpy())
    out = []
    for week in sorted(targets.week_cutoff.unique()):
        week = pd.Timestamp(week)
        tr = (~F.target & (F.known < week) & (F.cutoff >= week - pd.Timedelta(days=LAYER_WINDOW_DAYS))).to_numpy()
        te = (F.target & (F.cutoff == week)).to_numpy()
        X = np.concatenate([x[tr], -x[tr]])[:, None]
        y = np.concatenate([F.y.to_numpy()[tr], 1 - F.y.to_numpy()[tr]])
        layer = LogisticRegression(C=1.0, fit_intercept=False).fit(X, y)
        out.append(pd.DataFrame({"match_id": F.match_id.to_numpy()[te], "week_cutoff": week,
                                 "q_state": layer.predict_proba(x[te][:, None])[:, 1]}))
        if verbose:
            print(f"{week.date()}: layer on {tr.sum()} series, slope {layer.coef_[0, 0]:.3f}, {te.sum()} targets", flush=True)
    return pd.concat(out, ignore_index=True)


def combine(q_state: np.ndarray, q_e12: np.ndarray) -> np.ndarray:
    z = (1 - E12_WEIGHT) * logit(q_state) + E12_WEIGHT * logit(q_e12)
    return 1 / (1 + np.exp(-z))
