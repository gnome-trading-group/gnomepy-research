"""
g06 variant: the widened e12 of predict_wide plus the announced fives' FACEIT matchmaking skill (mm_skill):

    logit(q) = logit(e12 wide) + GAMMA * (mm_A - mm_B)

GAMMA is mm's weight relative to logit(e12 wide) in a logistic regression on the same pre-April 2025-01-06 ..
2026-03-01 HLTV bo3 outcomes PRIOR_SD was chosen on.

    python -m gnomepy_research.sessions.cs2_comp.g06.predict_wide_mm --data <bundle> --targets <parquet> --out <parquet>
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g06.history import logit
from gnomepy_research.sessions.cs2_comp.g06.mm_skill import MatchmakingGames, five_diff
from gnomepy_research.sessions.cs2_comp.g06.predict_wide import PRIOR_SD, arguments, read_targets, write
from gnomepy_research.sessions.cs2_comp.g06.wide import e12_wide

GAMMA = 0.406


def mm_diff(data_dir: str, targets: pd.DataFrame) -> pd.Series:
    games = MatchmakingGames(data_dir, targets.week_cutoff.max())
    skills = {c: games.fit(pd.Timestamp(c)) for c in targets.week_cutoff.unique()}
    return pd.Series([five_diff(skills[c], a, b) for a, b, c in
                      zip(targets.team_a_lineup_ids, targets.team_b_lineup_ids, targets.week_cutoff)], index=targets.match_id)


def main() -> None:
    a = arguments(__doc__)
    targets = read_targets(a.targets, a.weeks)
    with threadpool_limits(1):
        e = e12_wide(a.data, targets, PRIOR_SD)
        mm = mm_diff(a.data, targets)
    q = expit(logit(e.e12) + GAMMA * mm.reindex(e.match_id).to_numpy(float))
    write(pd.DataFrame({"match_id": e.match_id, "q": np.asarray(q), "fit_cutoff": e.week_cutoff}), a.out)


if __name__ == "__main__":
    main()
