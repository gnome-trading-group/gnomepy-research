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
from gnomepy_research.sessions.cs2_comp.g06.newcomer import feature_rows as newcomer_rows
from gnomepy_research.sessions.cs2_comp.g06.predict import targets_frame


def e12_wide(data_dir: str | Path, series: pd.DataFrame, prior_sd: float, newcomer: bool = False) -> pd.DataFrame:
    """e12's HLTV+FACEIT price for each series from the fits at its own week_cutoff, with prior sd `prior_sd`, and
    with `newcomer` the FACEIT-level prior for players the HLTV+FACEIT fit has not seen."""
    t = targets_frame(series)
    inputs = Inputs(data_dir, t.week_cutoff.max(), True)
    h = feature_rows(inputs, t, dataclasses.replace(CONFIG_HLTV, sigma=prior_sd), False)
    cfg = dataclasses.replace(CONFIG_FACEIT, sigma=prior_sd)
    f = newcomer_rows(inputs, t, cfg) if newcomer else feature_rows(inputs, t, cfg, True)
    X = design(h, f)[list(LAYER_FACEIT)]
    return pd.DataFrame({"match_id": h.match_id.to_numpy(), "week_cutoff": h.week_cutoff.to_numpy(),
                         "e12": expit(X.to_numpy() @ np.array(list(LAYER_FACEIT.values())))})
