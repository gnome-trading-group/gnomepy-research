"""e12-wide-mm (this branch's g06 at 59e8d4c) q for every target under one fit schedule; one part of the cutoffs.

    python -m gnomepy_research.sessions.cs2_comp.g06.within_event.fits --data <bundle> --targets <parquet> \
        --variant W|Wb|D --part i --parts n

S   a fit at 08:30 UTC each day on what hltv_cs2 (06:00) and faceit_daily (07:30) had loaded
W   weekly Monday 00:00 fits, the loader's own rule (series known from their start): what prod does
Wb  weekly fits, series known from start + buffer (pit.py)
D   a fit at 00:00 UTC of each target's match day, series known from start + buffer
Writes $OUTPUT_DIR/q_<variant>_<part>.parquet."""
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
ap.add_argument("--targets", required=True)
ap.add_argument("--variant", choices=("W", "Wb", "D", "S"), required=True)
ap.add_argument("--noise", choices=("none", "tight", "jitter1", "jitter2"), default="none",
                help="fit-noise check: tight = gtol 1e-8 and 1,000 iterations; jitterN = a seeded random start")
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--parts", type=int, default=1)
a = ap.parse_args()
if a.variant in ("Wb", "D"):
    pit.install()
if a.variant == "S":
    pit.install_schedule()
if a.noise != "none":
    # The skill fit is convex, so the only fit noise is where its optimizer stops: perturb that and nothing else.
    from gnomepy_research.sessions.cs2_comp.g06.e12 import model as skill_model
    _minimize = skill_model.minimize

    def minimize(fun, x0, **kw):
        if a.noise == "tight":
            kw["options"] = {**kw.get("options", {}), "gtol": 1e-8, "maxiter": 1000}
        else:
            x0 = np.random.default_rng(int(a.noise[-1])).normal(0.0, 0.1, len(x0))
        return _minimize(fun, x0, **kw)

    skill_model.minimize = minimize

from gnomepy_research.sessions.cs2_comp.g06.history import logit, naive  # noqa: E402
from gnomepy_research.sessions.cs2_comp.g06.predict_wide import PRIOR_SD  # noqa: E402
from gnomepy_research.sessions.cs2_comp.g06.predict_wide_mm import GAMMA, mm_diff  # noqa: E402
from gnomepy_research.sessions.cs2_comp.g06.wide import e12_wide  # noqa: E402

t = pd.read_parquet(a.targets).drop_duplicates("match_id").reset_index(drop=True)
t["week_cutoff"] = naive(t.week_cutoff)
if a.variant == "D":
    t["week_cutoff"] = naive(t.match_time).dt.normalize()
if a.variant == "S":
    # The day's fit runs at 08:30 and is ready by 09:30; a target uses the latest one ready 30 min before kickoff.
    usable = naive(t.match_time) - pd.Timedelta(minutes=30) - pd.Timedelta(hours=9.5)
    t["week_cutoff"] = usable.dt.normalize() + pd.Timedelta(hours=8.5)
cutoffs = sorted(t.week_cutoff.unique())
t = t[t.week_cutoff.isin(cutoffs[a.part::a.parts])].reset_index(drop=True)
assert (t.week_cutoff <= naive(t.match_time)).all(), "a fit cutoff after kickoff"
start = time.time()
with threadpool_limits(1):
    e = e12_wide(a.data, t, PRIOR_SD)
    mm = mm_diff(a.data, t)
q = expit(logit(e.e12) + GAMMA * mm.reindex(e.match_id).to_numpy(float))
out = Path(os.environ.get("OUTPUT_DIR", "."))
pd.DataFrame({"match_id": e.match_id, "q": np.asarray(q), "fit_cutoff": e.week_cutoff}).to_parquet(
    out / f"q_{a.variant}{'' if a.noise == 'none' else '_' + a.noise}_{a.part}.parquet", index=False)
print(f"{a.variant} part {a.part}/{a.parts}: {len(e)} predictions, {t.week_cutoff.nunique()} cutoffs, {time.time() - start:.0f}s")
