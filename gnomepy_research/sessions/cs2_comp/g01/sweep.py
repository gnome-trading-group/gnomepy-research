"""Hyper-parameter sweep for the skill fit: stage-1 log loss on April-July series, stage 1 trained before April."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.g01 import ratings, skill

FIRST_WEEK = "2025-10-06"
LAST_WEEK = "2026-07-27"
SPLIT = "2026-04-01"


def log_loss(p, y):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--configs", required=True, help="JSON list of SkillFit keyword dicts")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    a = ap.parse_args()
    hist = bundle.load(a.data, "hltv/cs2_match_history")
    maps = ratings.prepare_maps(hist)
    stats = skill.prepare_stats(bundle.load(a.data, "hltv/cs2_player_map_stats", columns=skill.STATS_COLUMNS))
    fm = skill.prepare_faceit(bundle.load(a.data, "faceit/faceit_match_history"), bundle.load(a.data, "faceit/faceit_exclusions"))
    s = ratings.series_table(maps)
    s = s[(s.bo == 3) & s.y.notna() & (s.t >= FIRST_WEEK)].copy()
    s["week"] = ratings.monday(s.t)
    results = []
    for cfg in json.loads(a.configs):
        t0 = time.time()
        kw = dict(cfg)
        faceit = fm if kw.pop("use_faceit", False) else None
        rows = []
        for w in pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D"):
            sub = s[s.week == w]
            if sub.empty:
                continue
            fit = skill.SkillFit(maps, w, stats, faceit=faceit, **kw)
            rows += [(r.t, r.y, *fit.diff(list(r.pa), list(r.pb))) for r in sub.itertuples()]
        d = pd.DataFrame(rows, columns=["t", "y"] + skill.FEATURES)
        tr = d.t < SPLIT
        clf = LogisticRegression(C=1, max_iter=1000).fit(d[skill.FEATURES][tr], d.y[tr])
        res = {"config": cfg, "loss": float(log_loss(clf.predict_proba(d[skill.FEATURES][~tr])[:, 1], d.y[~tr])),
               "seconds": time.time() - t0}
        print(res, flush=True)
        results.append(res)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sweep_{abs(hash(a.configs)) % 10**8}.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
