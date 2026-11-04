"""Weekly e12-wide-mm q (series known from start + buffer, pit.py) for every decided HLTV series Mar-Oct 2026, any
format, priced at its own Monday cutoff: the expectation each in-event result is measured against. One part of the
weeks.

    python -m gnomepy_research.sessions.cs2_comp.g06.within_event.all_series --data <bundle> --part i --parts n
Writes $OUTPUT_DIR/all_series_<part>.parquet."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g06.within_event import pit

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--parts", type=int, default=1)
a = ap.parse_args()
pit.install()

from gnomepy_research.models.cs2.hltv.config import showmatch_mask  # noqa: E402
from gnomepy_research.sessions.cs2_comp.g06.history import logit, naive, standins, week_start  # noqa: E402
from gnomepy_research.sessions.cs2_comp.g06.predict_wide import PRIOR_SD  # noqa: E402
from gnomepy_research.sessions.cs2_comp.g06.predict_wide_mm import GAMMA, mm_diff  # noqa: E402
from gnomepy_research.sessions.cs2_comp.g06.wide import e12_wide  # noqa: E402

h = pd.read_parquet(f"{a.data}/hltv/cs2_match_history.parquet")
h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna()]
s = h.groupby("match_id").agg(
    match_time=("match_time", "first"), match_date=("match_date", "first"), bo_type=("bo_type", "first"),
    event_name=("event_name", "first"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
    wins=("team_a_won", "sum"), maps=("team_a_won", "size"),
    team_a_lineup_ids=("team_a_lineup_ids", "first"), team_b_lineup_ids=("team_b_lineup_ids", "first")).reset_index()
s = s[s.wins * 2 != s.maps]
s["y"] = (s.wins * 2 > s.maps).astype(int)
t0 = naive(s.match_time)
s = s[(t0 >= "2026-03-01") & (t0 < "2026-10-27")]
five = lambda v: isinstance(v, (list, np.ndarray)) and len(v) == 5
s = s[[five(x) and five(y) for x, y in zip(s.team_a_lineup_ids, s.team_b_lineup_ids)]].copy()
s["week_cutoff"] = week_start(naive(s.match_time))
weeks = sorted(s.week_cutoff.unique())
s = s[s.week_cutoff.isin(weeks[a.part::a.parts])]
s = standins(a.data, s, None).astype({"team_a_id": np.int64, "team_b_id": np.int64}).reset_index(drop=True)
start = time.time()
with threadpool_limits(1):
    e = e12_wide(a.data, s, PRIOR_SD)
    mm = mm_diff(a.data, s)
q = expit(logit(e.e12) + GAMMA * mm.reindex(e.match_id).to_numpy(float))
out = s[["match_id", "match_time", "bo_type", "event_name", "team_a_id", "team_b_id", "y"]].merge(
    pd.DataFrame({"match_id": e.match_id, "q": np.asarray(q)}), on="match_id")
out.to_parquet(Path(os.environ.get("OUTPUT_DIR", ".")) / f"all_series_{a.part}.parquet", index=False)
print(f"part {a.part}/{a.parts}: {len(out)} series priced of {len(s)}, {s.week_cutoff.nunique()} weeks, {time.time() - start:.0f}s")
