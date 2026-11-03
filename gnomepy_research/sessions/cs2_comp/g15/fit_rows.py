"""
Batch-fit features per series: one skill fit per distinct week cutoff, pricing each job's announced fives.

    python -m gnomepy_research.sessions.cs2_comp.g15.fit_rows --data <bundle> --out <parquet>
        (--first 2025-07-07 --last 2026-03-23 | --targets <targets parquet>) [--weeks d1,d2] [--faceit]
        [--half-life 300] [--sigma 0.5] [--team-sigma 0] [--faceit-weight 0.5] [--window 365]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g15.data import attach_ratings, load_maps, load_rankings, load_series
from gnomepy_research.sessions.cs2_comp.g15.faceit import FaceitLink, load_faceit_maps
from gnomepy_research.sessions.cs2_comp.g15.features import _lineup, history_jobs, target_jobs
from gnomepy_research.sessions.cs2_comp.g15.ratings import Rankings
from gnomepy_research.sessions.cs2_comp.g15.skillfit import Fit, FitConfig, MapSet, fit_window, veto_series


def load_mapset(data_dir: str, until: pd.Timestamp, faceit: bool, ratings: bool = False) -> MapSet:
    h = load_maps(data_dir, until)
    h["known"] = np.maximum(h.time, h.match_date)
    if ratings:
        h = attach_ratings(data_dir, h, until)
    parts = [MapSet.from_maps(h, "hltv")]
    if faceit:
        parts.append(MapSet.from_maps(load_faceit_maps(data_dir, until, FaceitLink(data_dir)), "faceit"))
    return MapSet.concat(parts)


def price_jobs(fit: Fit, week: pd.DataFrame, rk: Rankings, cutoff: pd.Timestamp, prefix: str) -> list[dict]:
    out = []
    for j in week.itertuples(index=False):
        ta, tb = int(j.team_a), int(j.team_b)
        fa, fb = _lineup(j.lineup_a), _lineup(j.lineup_b)
        ra, rb = rk.rank_of(ta, cutoff), rk.rank_of(tb, cutoff)
        sa, unseen_a, few_a = fit.lineup(fa, ra)
        sb, unseen_b, few_b = fit.lineup(fb, rb)
        lan = bool(j.is_lan)
        if lan:
            sa += fit.lineup_lan(fa)
            sb += fit.lineup_lan(fb)
        row = {"match_id": j.match_id, "week_cutoff": cutoff, f"{prefix}_pgap": sa - sb,
               f"{prefix}_tgap": fit.team(ta) - fit.team(tb), f"{prefix}_unseen_a": unseen_a, f"{prefix}_unseen_b": unseen_b,
               f"{prefix}_few_a": few_a, f"{prefix}_few_b": few_b}
        if fit.map_dev is not None:
            ma, mb = fit.lineup_on_maps(fa, ra, lan), fit.lineup_on_maps(fb, rb, lan)
            if ma is not None and mb is not None:
                gaps = ma - mb
                p_veto = float(np.clip(veto_series(gaps), 1e-4, 1 - 1e-4))
                row[f"{prefix}_veto_logit"] = float(np.log(p_veto / (1 - p_veto)))
                row[f"{prefix}_map_spread"] = float(gaps.std())
            else:
                row[f"{prefix}_veto_logit"] = np.nan
                row[f"{prefix}_map_spread"] = np.nan
        out.append(row)
    return out


def fit_rows(data_dir: str, jobs: pd.DataFrame, cfg: FitConfig, faceit: bool, prefix: str = "bf", verbose: bool = False) -> pd.DataFrame:
    jobs = jobs.copy()
    jobs["week_cutoff"] = pd.to_datetime(jobs.week_cutoff)
    until = jobs.week_cutoff.max()
    t0 = time.time()
    ms = load_mapset(data_dir, until, faceit, cfg.rating_gamma > 0)
    rk = Rankings(load_rankings(data_dir, until))
    if verbose:
        print(f"{len(ms.frame)} maps loaded in {time.time() - t0:.1f}s", flush=True)
    out = []
    for cutoff in sorted(jobs.week_cutoff.unique()):
        cutoff = pd.Timestamp(cutoff)
        t0 = time.time()
        window = ms.between(cutoff - pd.Timedelta(days=cfg.window_days), cutoff)
        fit = fit_window(window, cutoff, cfg)
        week = jobs[jobs.week_cutoff == cutoff]
        out.extend(price_jobs(fit, week, rk, cutoff, prefix))
        if verbose:
            print(f"{cutoff:%Y-%m-%d} {fit.info} series {len(week)} {time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(out)
    df[f"{prefix}_gap"] = df[f"{prefix}_pgap"] + df[f"{prefix}_tgap"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first")
    ap.add_argument("--last")
    ap.add_argument("--targets")
    ap.add_argument("--weeks")
    ap.add_argument("--faceit", action="store_true")
    ap.add_argument("--half-life", type=float, default=300.0)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--team-sigma", type=float, default=0.0)
    ap.add_argument("--faceit-weight", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=365)
    ap.add_argument("--rating-gamma", type=float, default=0.0)
    ap.add_argument("--rating-sd", type=float, default=0.31)
    ap.add_argument("--map-sigma", type=float, default=0.0)
    ap.add_argument("--lan-sigma", type=float, default=0.0)
    ap.add_argument("--prefix", default="bf")
    a = ap.parse_args()
    cfg = FitConfig(window_days=a.window, half_life_days=a.half_life, sigma=a.sigma, team_sigma=a.team_sigma, faceit_weight=a.faceit_weight,
                    rating_gamma=a.rating_gamma, rating_sd=a.rating_sd, map_sigma=a.map_sigma,
                    lan_sigma=a.lan_sigma)
    if a.targets:
        jobs = target_jobs(pd.read_parquet(a.targets))
    else:
        first, last = pd.Timestamp(a.first), pd.Timestamp(a.last)
        jobs = history_jobs(load_series(a.data, last + pd.Timedelta(days=7)), first, last)
    if a.weeks:
        jobs = jobs[pd.to_datetime(jobs.week_cutoff).isin(pd.to_datetime(a.weeks.split(",")))]
    rows = fit_rows(a.data, jobs, cfg, a.faceit, a.prefix, verbose=True)
    rows.to_parquet(a.out, index=False)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
