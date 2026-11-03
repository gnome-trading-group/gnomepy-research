from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04.pm import Fit

CONFIGS = [dict(faceit_weight=0.5), dict(use_faceit=False)]


def bo3(p):
    return p * p * (3 - 2 * p)


def _logit(p):
    return np.log(p / (1 - p))


def predict(data_dir, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff, grp in targets.groupby("week_cutoff"):
        fits = [Fit(data_dir, cutoff, **c) for c in CONFIGS]
        for r in grp.itertuples():
            zs = [f.map_logit(r.team_a_id, list(r.team_a_lineup_ids), r.team_b_id, list(r.team_b_lineup_ids)) for f in fits]
            lq = np.mean([_logit(bo3(1 / (1 + np.exp(-z)))) for z in zs])
            rows.append((r.match_id, 1 / (1 + np.exp(-lq)), pd.Timestamp(cutoff)))
    out = pd.DataFrame(rows, columns=["match_id", "q", "fit_cutoff"])
    out["q"] = out.q.clip(0.02, 0.98)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    predict(a.data, pd.read_parquet(a.targets)).to_parquet(a.out)


if __name__ == "__main__":
    main()
