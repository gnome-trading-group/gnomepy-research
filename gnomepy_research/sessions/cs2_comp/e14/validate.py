"""
Walk-forward check on HLTV bo3 series before the tune period, against outcomes only (no market).

    python -m gnomepy_research.sessions.cs2_comp.e14.validate --data <bundle> --first 2026-01-05 --last 2026-03-23
        [--variants base,nofaceit,...] [--out <parquet>]

Each Monday cutoff prices that week's bo3 series from data before it; prints log loss by variant, overall and for
online series where a team is thin (at most 15 HLTV maps in the prior 90 days) or neither team is ranked top 100.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e14.data import load_bundle
from gnomepy_research.sessions.cs2_comp.e14.model import Config, fit

VARIANTS = {
    "base": (True, Config()),
    "nofaceit": (False, Config(faceit_priors=False)),
    "nopriors": (True, Config(faceit_priors=False)),
    "noratings": (True, Config(use_ratings=False)),
    "hl60": (True, Config(half_life_days=60.0)),
    "hl240": (True, Config(half_life_days=240.0)),
    "phif05": (True, Config(phi=(0.5, 0.5))),
    "phif015": (True, Config(phi=(0.5, 0.15))),
    "phi03": (True, Config(phi=(0.3, 0.2))),
    "sig02": (True, Config(sigma0=0.2)),
    "fall": (True, Config(faceit_scope="all")),
    "luhltv": (True, Config(lineup_source="hltv")),
    "sig045": (True, Config(sigma0=0.45)),
}


def pseudo_targets(data, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    h = load(data, "hltv/cs2_match_history", last + pd.Timedelta(days=7))
    h = h[~showmatch_mask(h) & (h.bo_type == 3)]
    s = h.drop_duplicates("match_id").copy()
    t = pd.to_datetime(s.match_time.fillna(s.match_date))
    s["t"] = t.dt.tz_convert(None) if t.dt.tz is not None else t
    s["week_cutoff"] = s.t.dt.normalize() - pd.to_timedelta(s.t.dt.weekday, unit="D")
    s = s[(s.week_cutoff >= first) & (s.week_cutoff <= last)]
    decisive = h[h.team_a_score != h.team_b_score].drop_duplicates(["match_id", "map_name"])
    won = decisive.groupby("match_id").team_a_won.agg(["sum", "size"])
    won = won[won["sum"] * 2 != won["size"]]
    s = s[s.match_id.isin(won.index)].copy()
    s["y"] = (s.match_id.map(won["sum"] * 2 > won["size"])).astype(int)
    hist = h.drop_duplicates(["match_id", "map_name"])
    days = pd.concat([hist[["match_date", "team_a_id"]].set_axis(["d", "t"], axis=1),
                      hist[["match_date", "team_b_id"]].set_axis(["d", "t"], axis=1)])
    by_t = {k: np.sort(pd.to_datetime(g.d).to_numpy()) for k, g in days.groupby("t")}

    def maps90(team, day):
        a = by_t.get(team, np.array([], "datetime64[ns]"))
        return np.searchsorted(a, np.datetime64(day)) - np.searchsorted(a, np.datetime64(day - pd.Timedelta(days=90)))

    best = np.fmin(s.team_a_rank.astype(float), s.team_b_rank.astype(float)).fillna(999)
    mm = np.array([min(maps90(a, c), maps90(b, c)) for a, b, c in zip(s.team_a_id, s.team_b_id, s.week_cutoff)])
    s["thin"] = (s.is_lan == 0) & ((mm <= 15) | (best > 100))
    return s[["match_id", "team_a_id", "team_b_id", "bo_type", "week_cutoff", "y", "thin", "event_tier", "is_lan"]] \
        .reset_index(drop=True)


def calibrated(q, y):
    """Log loss after the best single slope on logit(q): the pool rescales q, so this is what a variant adds."""
    z = np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    a = 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-a * z))
        a += np.sum(z * (y - p)) / max(np.sum(z * z * p * (1 - p)), 1e-9)
    return 1 / (1 + np.exp(-a * z)), a


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--first", default="2026-01-05")
    ap.add_argument("--last", default="2026-03-23")
    ap.add_argument("--variants", default="base,nofaceit")
    ap.add_argument("--out")
    args = ap.parse_args()
    first, last = pd.Timestamp(args.first), pd.Timestamp(args.last)
    tg = pseudo_targets(args.data, first, last)
    names = args.variants.split(",")
    bundles = {}
    for name in names:
        faceit, _ = VARIANTS[name]
        if faceit not in bundles:
            bundles[faceit] = load_bundle(args.data, last, faceit=faceit)
    cuts = sorted(tg.week_cutoff.unique())
    for name in names:
        faceit, cfg = VARIANTS[name]
        q = np.empty(len(tg))
        t0 = time.time()
        for c in cuts:
            f = fit(bundles[faceit], pd.Timestamp(c), cfg)
            m = (tg.week_cutoff == c).to_numpy()
            q[m] = [f.price(a, b) for a, b in zip(tg.team_a_id[m], tg.team_b_id[m])]
        tg[name + "_raw"] = q
        tg[name], slope = calibrated(q, tg.y.to_numpy())
        l = ll(tg[name], tg.y.to_numpy())
        th = tg.thin.to_numpy()
        print(f"{name:10s} raw {ll(q, tg.y.to_numpy()).mean():.4f} calibrated (slope {slope:.2f}) all {l.mean():.4f} "
              f"(n={len(l)})  thin {l[th].mean():.4f} (n={th.sum()})  rest {l[~th].mean():.4f}  {time.time() - t0:.0f}s",
              flush=True)
    if len(names) > 1:
        y = tg.y.to_numpy()
        rng = np.random.default_rng(0)
        for name in names[1:]:
            d = ll(tg[names[0]], y) - ll(tg[name], y)
            b = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            print(f"{names[0]} - {name}: {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]")
    out = args.out or (str(Path(os.environ["OUTPUT_DIR"]) / f"validate_{'_'.join(names)}.parquet")
                       if "OUTPUT_DIR" in os.environ else None)
    if out:
        tg.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
