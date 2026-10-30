"""
Development diagnostics on series before the tune period, scored against HLTV outcomes only (no market).

Every HLTV bo3 series from --first up to the tune period becomes a target with its own week cutoff, and is priced
by the two skill filters exactly as predict.py prices the tune targets (the Bayes part is left out: its fits need
Batch). Reports raw log loss of the HLTV + FACEIT filter, log loss after a month-held-out logistic recalibration (the
scale the organizer's pool would apply), and what the FACEIT delta adds on top of the HLTV-only filter in a
month-held-out logistic regression, overall and by FACEIT coverage of the two fives.

With --bayes-cache (a directory of posteriors from bayes_part.py) it also prices each series with the Bayes model,
from the latest posterior fitted at or before its week (sparser than predict.py's refits, to save Batch time), and
fits the weight of the gated FACEIT delta on top of logit(q_bayes): K_FACEIT in predict.py is that weight over the
Bayes weight.

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

from gnomepy_research.models.cs2.hltv.bayes_series import SeriesPosterior
from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.e11.bayes_part import rank_strength_before
from gnomepy_research.sessions.cs2_comp.e11.predict import MIN_COVERAGE, filter_prices
from gnomepy_research.sessions.cs2_comp.e11.skill import FilterConfig

TUNE_START = pd.Timestamp("2026-03-30")


def dev_targets(data: Path, first: pd.Timestamp) -> pd.DataFrame:
    h = bundle.load(data, "hltv/cs2_match_history", TUNE_START)
    h = h[~showmatch_mask(h) & (h.bo_type == 3)]
    h = h[pd.to_datetime(h.match_date) >= first]
    won = h.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() * 2 > len(w)))
    s = h.drop_duplicates("match_id")[["match_id", "match_date", "team_a_id", "team_b_id", "is_lan", "team_a_rank",
                                       "team_b_rank", "team_a_lineup_ids", "team_b_lineup_ids"]].copy()
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


def _fit(X, y) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def held_out(X, y, months) -> tuple[np.ndarray, np.ndarray]:
    """Month-held-out logistic predictions with no intercept, and the weights fitted on all months."""
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ _fit(X[tr], y[tr])))
    return out, _fit(X, y)


def recalibrated(q, y, months) -> np.ndarray:
    return held_out(_logit(q)[:, None], y, months)[0]


def bayes_dev_prices(data: Path, priced: pd.DataFrame, targets: pd.DataFrame, cache: Path) -> np.ndarray:
    files = sorted(cache.glob("posterior_*.npz"))
    anchors = [pd.Timestamp(f.stem.split("_")[1]) for f in files]
    team = targets.set_index("match_id")[["team_a_id", "team_b_id"]]
    rankings = bundle.load(data, "hltv/cs2_team_rankings", pd.to_datetime(targets.week_cutoff).max())
    out = np.full(len(priced), np.nan)
    for cutoff, rows in priced.groupby("fit_cutoff", sort=True):
        usable = [f for f, a in zip(files, anchors) if a <= cutoff]
        if not usable:
            continue
        posterior = SeriesPosterior.load(usable[-1])
        ranks = rank_strength_before(rankings, cutoff)
        for i, r in zip(rows.index, rows.itertuples()):
            a, b = (int(x) for x in team.loc[r.match_id])
            out[priced.index.get_loc(i)] = posterior.price(a, list(r.five_a), ranks(a, cutoff),
                                                           b, list(r.five_b), ranks(b, cutoff))
    return out


def bayes_report(p: pd.DataFrame) -> dict:
    p = p[p.q_bayes.notna()]
    y, months = p.y.to_numpy(float), p.month.to_numpy()
    xb = _logit(p.q_bayes)
    delta = _logit(p.q_f) - _logit(p.q_h)
    gate = (p["cov"] >= MIN_COVERAGE).to_numpy(float)
    base = _ll(held_out(xb[:, None], y, months)[0], y)
    res = {"n": len(p), "ll_bayes_cal": float(base.mean())}
    for name, d in (("gated", gate * delta), ("ungated", delta)):
        q, w = held_out(np.column_stack([xb, d]), y, months)
        gain = _ll(q, y) - base
        res[name] = {"w_bayes": float(w[0]), "w_delta": float(w[1]), "k": float(w[1] / w[0]),
                     "gain": float(gain.mean()),
                     **{c: float(gain[g].mean()) for c, g in (("cov0", p["cov"].to_numpy() == 0),
                                                              ("cov1-3", p["cov"].between(1, 3).to_numpy()),
                                                              ("cov4-5", p["cov"].to_numpy() >= 4))}}
    return res


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
    ap.add_argument("--bayes-cache", type=Path)
    ap.add_argument("--latest-fives", dest="announced", action="store_false",
                    help="price on each team's latest five instead of the announced lineup")
    a = ap.parse_args()
    t0 = time.time()
    targets = dev_targets(a.data, pd.Timestamp(a.first))
    if not a.announced:
        targets = targets.drop(columns=["team_a_lineup_ids", "team_b_lineup_ids"])
    cfg = FilterConfig(w_faceit=a.w_faceit, a_faceit=a.a_faceit, s_faceit=a.s_faceit, sigma_series=a.sigma_series,
                       faceit_lineups=a.faceit_lineups)
    p = filter_prices(a.data, targets, a.faceit, cfg).merge(
        targets[["match_id", "y", "is_lan", "team_a_rank", "team_b_rank"]], on="match_id")
    p["q"] = p.q_f
    p["month"] = pd.to_datetime(p.fit_cutoff).dt.to_period("M").astype(str)
    y, months = p.y.to_numpy(float), p.month.to_numpy()
    p["q_cal"] = recalibrated(p.q, y, months)
    p["q_h_cal"] = recalibrated(p.q_h, y, months)
    delta = _logit(p.q_f) - _logit(p.q_h)
    p["q_hd"], w_hd = held_out(np.column_stack([_logit(p.q_h), delta]), y, months)
    if a.bayes_cache is not None:
        p["q_bayes"] = bayes_dev_prices(a.data, p, targets, a.bayes_cache)
    a.out.mkdir(parents=True, exist_ok=True)
    p.drop(columns=["five_a", "five_b"]).to_parquet(a.out / f"dev_{a.tag}.parquet", index=False)
    cfg_json = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    res = {"tag": a.tag, "faceit": a.faceit, "cfg": cfg_json,
           "n": len(p), "ll": float(_ll(p.q, p.y).mean()), "ll_cal": float(_ll(p.q_cal, p.y).mean()),
           "w_hltv": float(w_hd[0]), "w_delta": float(w_hd[1]), "delta_sd": float(np.std(delta))}
    for name, g in (("cov0", p[p["cov"] == 0]), ("cov1-3", p[p["cov"].between(1, 3)]), ("cov4-5", p[p["cov"] >= 4]),
                    ("jan-mar", p[p.month >= "2026-01"])):
        res[name] = {"n": len(g), "ll": float(_ll(g.q, g.y).mean()), "ll_cal": float(_ll(g.q_cal, g.y).mean()),
                     "delta_gain": float((_ll(g.q_hd, g.y) - _ll(g.q_h_cal, g.y)).mean())}
    res["delta_gain"] = float((_ll(p.q_hd, p.y) - _ll(p.q_h_cal, p.y)).mean())
    if a.bayes_cache is not None:
        res["bayes"] = bayes_report(p)
    res["seconds"] = round(time.time() - t0)
    print(json.dumps(res))
    with open(a.out / "results.jsonl", "a") as fh:
        fh.write(json.dumps(res, default=str) + "\n")


if __name__ == "__main__":
    main()
