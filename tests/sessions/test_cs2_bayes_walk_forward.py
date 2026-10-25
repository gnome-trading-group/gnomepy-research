import pandas as pd
import pytest

from gnomepy_research.sessions.cs2_win_probability.bayes.walk_forward import STEPS, series_in


@pytest.mark.parametrize("step", sorted(STEPS))
def test_each_bo3_series_is_priced_by_exactly_one_window(step):
    days = pd.to_datetime(["2026-04-30", "2026-05-01", "2026-05-01", "2026-05-03", "2026-05-04", "2026-05-31"])
    maps = pd.DataFrame({"match_id": [1, 2, 2, 3, 4, 5], "day": days, "bo_type": [3, 3, 3, 3, 1, 3]})
    freq, length = STEPS[step]
    starts = pd.date_range("2026-04-01", "2026-06-01", freq=freq)
    priced = pd.concat([series_in(maps, s, length) for s in starts]).match_id
    assert sorted(priced) == [1, 2, 3, 5]
