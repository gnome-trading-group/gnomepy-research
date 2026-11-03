"""g08's series price: the recurrent skill state read out at each weekly cutoff and a logistic layer refitted every week
on past HLTV bo3 series.

The layer sees logit of the veto-simulated series price from the state and the difference in the two teams' recent form
(how far their last few maps beat the state's own expectation, a short memory the slow state smooths away). It is
fitted on both orientations of every past series (no intercept), so swapping the teams flips the price. It is refitted each week on the series
known before that week's cutoff from the previous LAYER_WINDOW_DAYS, which keeps the price-to-outcome scale current as
the state's own scale drifts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp.g08.data import Inputs
from gnomepy_research.sessions.cs2_comp.g08.ratings import RatingConfig, run

CONFIG = RatingConfig(prior_var=1.5, drift_per_day=0.00002, new_offset=-0.7, market_var=0.3, map_var=0.003,
                      rating_gamma=0.4, rating_sd=1.0, faceit_rating_sd=2.0)
HISTORY_START = pd.Timestamp("2024-06-01")
LAYER_WINDOW_DAYS = 540
LAYER_COLUMNS = ("sv", "form")


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def layer_features(F: pd.DataFrame, cols: list[str]) -> np.ndarray:
    feats = {"sv": logit(F.series_veto), "sf": logit(F.series_flat), "su": logit(F.series_unc),
             "unk": (F.unknown_a - F.unknown_b).to_numpy(float), "form": (F.form_a - F.form_b).to_numpy(float),
             "coh": (F.cohesion_a - F.cohesion_b).to_numpy(float)}
    feats["sv_lan"] = feats["sv"] * F.lan.to_numpy(float)
    for t in sorted(F.tier.unique()):
        feats[f"sv_t{t}"] = feats["sv"] * (F.tier == t).to_numpy(float)
    return np.column_stack([feats[c] for c in cols])


def _queries(inputs: Inputs, targets: pd.DataFrame) -> pd.DataFrame:
    s = inputs.series
    s = s[(s.bo_type == 3) & (s.time >= HISTORY_START)]
    hist = pd.DataFrame({"qid": "h" + s.match_id.astype(str), "match_id": s.match_id.to_numpy(), "cutoff": s.cutoff.to_numpy(),
                         "known": s.known.to_numpy(), "y": s.y.to_numpy(float), "team_a": s.team_a.to_numpy(),
                         "team_b": s.team_b.to_numpy(), "five_a": s.five_a.to_numpy(), "five_b": s.five_b.to_numpy(),
                         "lan": s.is_lan.fillna(0).to_numpy(int), "tier": s.event_tier.fillna(-1).to_numpy(int), "target": False})
    tg = pd.DataFrame({"qid": "t" + targets.match_id.astype(str), "match_id": targets.match_id.to_numpy(),
                       "cutoff": targets.week_cutoff.to_numpy(), "known": targets.week_cutoff.to_numpy(), "y": np.nan,
                       "team_a": targets.team_a_id.to_numpy(np.int64), "team_b": targets.team_b_id.to_numpy(np.int64),
                       "five_a": targets.team_a_lineup_ids.to_numpy(), "five_b": targets.team_b_lineup_ids.to_numpy(),
                       "lan": targets.is_lan.fillna(0).to_numpy(int), "tier": targets.event_tier.fillna(-1).to_numpy(int),
                       "target": True})
    return pd.concat([hist, tg], ignore_index=True)


def state_prices(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """match_id, week_cutoff, q_state for every target: the layer over the recurrent state, fitted per week."""
    inputs = Inputs(data_dir, targets.week_cutoff.max(), faceit=True, market=True)
    Q = _queries(inputs, targets)
    feats = run(inputs.maps, Q, CONFIG)
    F = Q.merge(feats, on="qid")
    x = layer_features(F, list(LAYER_COLUMNS))
    out = []
    for week in sorted(targets.week_cutoff.unique()):
        week = pd.Timestamp(week)
        tr = (~F.target & (F.known < week) & (F.cutoff >= week - pd.Timedelta(days=LAYER_WINDOW_DAYS))).to_numpy()
        te = (F.target & (F.cutoff == week)).to_numpy()
        X = np.concatenate([x[tr], -x[tr]])
        y = np.concatenate([F.y.to_numpy()[tr], 1 - F.y.to_numpy()[tr]])
        layer = LogisticRegression(C=1.0, fit_intercept=False).fit(X, y)
        out.append(pd.DataFrame({"match_id": F.match_id.to_numpy()[te], "week_cutoff": week,
                                 "q_state": layer.predict_proba(x[te])[:, 1]}))
        if verbose:
            print(f"{week.date()}: layer on {tr.sum()} series, weights {layer.coef_[0].round(3)}, {te.sum()} targets", flush=True)
    return pd.concat(out, ignore_index=True)
