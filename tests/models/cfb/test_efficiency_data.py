from __future__ import annotations

import pandas as pd

from gnomepy_research.models.cfb.efficiency.data import RETURNING_FEATURES, unknown_stale_returning


def test_returning_production_from_a_stale_roster_is_unknown():
    # CFBD says 90% of passing PPA returns, yet last season's primary passer is not on this season's roster (team 1).
    pre = pd.DataFrame({"team_id": [1, 2, 3], "qb_returning_starter": [False, True, False],
                        "returning_pct_ppa": [0.8, 0.8, 0.4], "returning_pct_passing_ppa": [0.9, 0.9, 0.5]})
    out = unknown_stale_returning(pre).set_index("team_id")
    assert out.loc[1, RETURNING_FEATURES].isna().all()
    assert out.loc[2, "returning_pct_passing_ppa"] == 0.9 and out.loc[3, "returning_pct_passing_ppa"] == 0.5


def test_input_not_mutated():
    pre = pd.DataFrame({"team_id": [1], "qb_returning_starter": [False], "returning_pct_ppa": [0.8],
                        "returning_pct_passing_ppa": [0.9]})
    unknown_stale_returning(pre)
    assert pre.loc[0, "returning_pct_passing_ppa"] == 0.9
