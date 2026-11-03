from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04.players import PlayerFit

PARAMS = dict(lam_player=16.0, lam_team=0.5, half_life=1200.0, soft=1.0, rating_prior=20.0)


def bo3(p):
    return p * p * (3 - 2 * p)


def predict(data_dir, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff, grp in targets.groupby("week_cutoff"):
        fit = PlayerFit(data_dir, cutoff, **PARAMS)
        for r in grp.itertuples():
            z = fit.map_logit(r.team_a_id, list(r.team_a_lineup_ids), r.team_b_id, list(r.team_b_lineup_ids))
            rows.append((r.match_id, bo3(1 / (1 + np.exp(-z))), pd.Timestamp(cutoff)))
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
