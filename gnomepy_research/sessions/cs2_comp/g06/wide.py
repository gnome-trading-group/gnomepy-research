"""
e12 with a wider player-skill prior: both skill fits (HLTV, and HLTV+FACEIT) use prior sd PRIOR_SD around the learned
prior mean instead of e12's 0.5; everything else in e12, including its fixed logistic layer, is unchanged.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g06.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g06.e12.features import feature_rows
from gnomepy_research.sessions.cs2_comp.g06.e12.predict import CONFIG_FACEIT, CONFIG_HLTV, LAYER_FACEIT, design
from gnomepy_research.sessions.cs2_comp.g06.predict import targets_frame


def e12_wide(data_dir: str | Path, series: pd.DataFrame, prior_sd: float) -> pd.DataFrame:
    """e12's HLTV+FACEIT price for each series from the fits at its own week_cutoff, with prior sd `prior_sd`."""
    return e12_wide_internals(data_dir, series, prior_sd)[0]


def e12_wide_internals(data_dir: str | Path, series: pd.DataFrame, prior_sd: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """e12_wide's prices, and per series the inputs they were built from: each side's features from both skill fits
    (h_*, f_*) and the layer columns (x_*)."""
    t = targets_frame(series)
    inputs = Inputs(data_dir, t.week_cutoff.max(), True)
    h = feature_rows(inputs, t, dataclasses.replace(CONFIG_HLTV, sigma=prior_sd), False)
    f = feature_rows(inputs, t, dataclasses.replace(CONFIG_FACEIT, sigma=prior_sd), True)
    X = design(h, f)[list(LAYER_FACEIT)]
    prices = pd.DataFrame({"match_id": h.match_id.to_numpy(), "week_cutoff": h.week_cutoff.to_numpy(),
                           "e12": expit(X.to_numpy() @ np.array(list(LAYER_FACEIT.values())))})
    side = [c for c in h.columns if c.endswith(("_a", "_b")) or c in ("gap", "gap_last")]
    internals = pd.concat([prices[["match_id", "week_cutoff"]].reset_index(drop=True),
                           h[side].add_prefix("h_").reset_index(drop=True),
                           f[[c for c in f.columns if c in side or c.startswith("linked_")]].add_prefix("f_").reset_index(drop=True),
                           X.add_prefix("x_").reset_index(drop=True)], axis=1)
    return prices, internals
