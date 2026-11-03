"""The kickoff price of each series from its Polymarket path.

m0 is the last print within 30 minutes before kickoff (the price the competition scores against); without one, the
last print before kickoff stands in and has_m0 is 0. Earlier points of the path (prices one to 24 hours out, the
six-hour mean, how often the price moved) were tried as layer inputs and added nothing, so only m0 is kept."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g07.data import naive

M0_WINDOW = pd.Timedelta(minutes=30)


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """paths: match_id, kickoff, t, p_a. One row per match_id with a print at or before kickoff."""
    if paths.empty:
        return pd.DataFrame(columns=["match_id", "m0", "has_m0", "nprint"])
    d = paths.assign(kickoff=naive(paths.kickoff), t=naive(paths.t))
    d = d[d.t <= d.kickoff].sort_values(["match_id", "t"], kind="stable")
    g = d.groupby("match_id")
    last = g.tail(1).set_index("match_id")
    return pd.DataFrame({"match_id": last.index.to_numpy(), "m0": last.p_a.to_numpy(float),
                         "has_m0": ((last.kickoff - last.t) <= M0_WINDOW).astype(float).to_numpy(),
                         "nprint": g.size().reindex(last.index).to_numpy()})
