"""
Development diagnostics on series before the tune period, scored against HLTV outcomes only (no market).

Every HLTV bo3 series from --first up to the tune period becomes a target with its own week cutoff, and is priced
exactly as predict.py prices the tune targets. Reports raw log loss and log loss after a month-held-out logistic
recalibration (the scale the organizer's pool would apply), overall and by FACEIT coverage of the two fives.

    python -m gnomepy_research.sessions.cs2_comp.e11.evaluate --data <bundle> --out <dir> [--no-faceit]
        [--w-faceit 0.5] [--a-faceit 0] [--s-faceit 0] [--first 2025-10-06]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.e11.predict import predict
from gnomepy_research.sessions.cs2_comp.e11.skill import FilterConfig
from gnomepy_research.sessions.cs2_comp.e11.stream import FACEIT, build_stream

TUNE_START = pd.Timestamp("2026-03-30")


def dev_targets(data: Path, first: pd.Timestamp) -> pd.DataFrame:
    h = bundle.load(data, "hltv/cs2_match_history", TUNE_START)
    h = h[~showmatch_mask(h) & (h.bo_type == 3)]
    h = h[pd.to_datetime(h.match_date) >= first]
    won = h.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() * 2 > len(w)))
    s = h.drop_duplicates("match_id")[["match_id", "match_date", "team_a_id", "team_b_id", "is_lan", "team_a_rank",
                                       "team_b_rank"]].copy()
    d = pd.to_datetime(s.match_date)
    s["week_cutoff"] = d - pd.to_timedelta(d.dt.weekday, unit="D")
    s["y"] = s.match_id.map(won).to_numpy()
    return s[s.week_cutoff < TUNE_START].reset_index(drop=True)


def _ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def recalibrated(q, y, months) -> np.ndarray:
    x = _logit(q)
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        w = 1.0
        for _ in range(50):
            p = 1 / (1 + np.exp(-w * x[tr]))
            w += float(np.sum(x[tr] * (y[tr] - p)) / (np.sum(x[tr] ** 2 * p * (1 - p)) + 1e-9))
        out[~tr] = 1 / (1 + np.exp(-w * x[~tr]))
    return out


def coverage(data: Path, targets: pd.DataFrame, preds: pd.DataFrame) -> np.ndarray:
    """Fewest players in either five with a FACEIT map before the cutoff (the harness's split)."""
    s = build_stream(data, targets.week_cutoff.max(), True)
    fa = s.source == FACEIT
    first: dict[int, np.datetime64] = {}
    for t, pa, pb in zip(s.avail[fa], np.array(s.players_a, dtype=object)[fa], np.array(s.players_b, dtype=object)[fa]):
        for p in pa + pb:
            if p > 0 and p not in first:
                first[p] = t
    out = []
    for r in preds.itertuples():
        c = np.datetime64(pd.Timestamp(r.fit_cutoff).to_datetime64(), "ns")
        out.append(min(sum(1 for p in five if p in first and first[p] < c) for five in (r.five_a, r.five_b)))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--w-faceit", type=float, default=0.5)
    ap.add_argument("--a-faceit", type=float, default=0.0)
    ap.add_argument("--s-faceit", type=float, default=0.0)
    ap.add_argument("--sigma-series", type=float, default=0.0)
    ap.add_argument("--faceit-lineups", type=int, default=1)
    ap.add_argument("--first", default="2025-10-06")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    t0 = time.time()
    targets = dev_targets(a.data, pd.Timestamp(a.first))
    cfg = FilterConfig(w_faceit=a.w_faceit, a_faceit=a.a_faceit, s_faceit=a.s_faceit, sigma_series=a.sigma_series,
                       faceit_lineups=a.faceit_lineups)
    p = predict(a.data, targets, a.faceit, cfg).merge(targets[["match_id", "y", "is_lan", "team_a_rank", "team_b_rank"]],
                                                      on="match_id")
    p["month"] = pd.to_datetime(p.fit_cutoff).dt.to_period("M").astype(str)
    p["q_cal"] = recalibrated(p.q, p.y.to_numpy(float), p.month.to_numpy())
    p["cov"] = coverage(a.data, targets, p)
    a.out.mkdir(parents=True, exist_ok=True)
    p.drop(columns=["five_a", "five_b"]).to_parquet(a.out / f"dev_{a.tag}.parquet", index=False)
    res = {"tag": a.tag, "faceit": a.faceit, "cfg": vars(a) | {"data": str(a.data), "out": str(a.out)},
           "n": len(p), "ll": float(_ll(p.q, p.y).mean()), "ll_cal": float(_ll(p.q_cal, p.y).mean())}
    for name, g in (("cov0", p[p["cov"] == 0]), ("cov1-3", p[p["cov"].between(1, 3)]), ("cov4-5", p[p["cov"] >= 4]),
                    ("jan-mar", p[p.month >= "2026-01"])):
        res[name] = {"n": len(g), "ll": float(_ll(g.q, g.y).mean()), "ll_cal": float(_ll(g.q_cal, g.y).mean())}
    res["seconds"] = round(time.time() - t0)
    print(json.dumps(res))
    with open(a.out / "results.jsonl", "a") as fh:
        fh.write(json.dumps(res, default=str) + "\n")


if __name__ == "__main__":
    main()
