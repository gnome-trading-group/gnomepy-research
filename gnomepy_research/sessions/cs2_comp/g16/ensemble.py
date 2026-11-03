"""
e12's price as one more column of the layer: the frozen baseline's batch fit over a year of maps keeps information
the filter's local updates lose, and the two skill estimators, priced from the same fives, disagree enough to be
worth a weight each. The copied e12 (`g16/e12`) prices every market-joined series at its own week's cutoff, as the
filter does, so the layer learns the weights walk-forward.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g16.e12.predict import predict as e12_predict


def e12_prices(data_dir: str | Path, rows: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """rows: match_id, week_cutoff, team_a, team_b, lineup_a, lineup_b, standins_a, standins_b. Returns match_id, e12."""
    t = pd.DataFrame({"match_id": rows.match_id.to_numpy(), "team_a_id": rows.team_a.to_numpy(), "team_b_id": rows.team_b.to_numpy(),
                      "week_cutoff": rows.week_cutoff.to_numpy(),
                      "team_a_lineup_ids": [list(v) if v is not None else None for v in rows.lineup_a],
                      "team_b_lineup_ids": [list(v) if v is not None else None for v in rows.lineup_b],
                      "team_a_standin_ids": [[0] * int(n) for n in rows.standins_a],
                      "team_b_standin_ids": [[0] * int(n) for n in rows.standins_b]})
    out = e12_predict(data_dir, t, faceit=True, verbose=verbose)
    return out.rename(columns={"q": "e12"})[["match_id", "e12"]]
