"""Selection runs for the refit variant, on Batch: one task per JSON list of overrides of the scored model's constants.

    python -m gnomepy_research.sessions.cs2_comp.g08.refit_tune '<json list of overrides>'

Same criterion the scored model's constants were chosen by (log loss of the weekly-refitted layer over past HLTV bo3
series), on a longer window: every week from 2025-09-01 up to the 2026-08-01 cut, with every table read through
`bundle.load(before=2026-08-01)`. Overrides may name RatingConfig fields and "window" (the layer's training window)."""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp.g08.data import Inputs
from gnomepy_research.sessions.cs2_comp.g08.model import CONFIG, LAYER_COLUMNS, LAYER_WINDOW_DAYS, _queries, layer_features
from gnomepy_research.sessions.cs2_comp.g08.ratings import run

DATA = Path("/work/in/comp")
CUT = pd.Timestamp("2026-08-01")
START = pd.Timestamp("2025-09-01")


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def selection_ll(inputs: Inputs, overrides: dict) -> float:
    ov = dict(overrides)
    window = ov.pop("window", LAYER_WINDOW_DAYS)
    cfg = dataclasses.replace(CONFIG, **ov)
    no_targets = pd.read_parquet(DATA / "targets_tune.parquet").iloc[:0]
    Q = _queries(inputs, no_targets)
    F = Q.merge(run(inputs.maps, Q, cfg), on="qid")
    X = layer_features(F, list(LAYER_COLUMNS))
    p = np.full(len(F), np.nan)
    for w in sorted(F.cutoff[(F.cutoff >= START) & (F.cutoff < CUT)].unique()):
        w = pd.Timestamp(w)
        tr = ((F.known < w) & (F.cutoff >= w - pd.Timedelta(days=window))).to_numpy()
        te = ((F.cutoff == w) & (F.known < CUT)).to_numpy()
        m = LogisticRegression(fit_intercept=False).fit(np.vstack([X[tr], -X[tr]]), np.r_[F.y.values[tr], 1 - F.y.values[tr]])
        p[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(p)
    return float(ll(p[ok], F.y.to_numpy()[ok]).mean())


def main() -> None:
    inputs = Inputs(DATA, CUT, faceit=True, market=True)
    results = []
    for ov in json.loads(sys.argv[1]):
        r = {"overrides": ov, "ll": selection_ll(inputs, ov)}
        print(json.dumps(r), flush=True)
        results.append(r)
    out = Path(os.environ.get("OUTPUT_DIR", "."))
    (out / f"refit_{abs(hash(sys.argv[1])) % 10**8}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
