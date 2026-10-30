"""
The HLTV Bayes series model (models/cs2/bayes, unchanged), fitted point in time on the bundle.

A posterior is fitted at an anchor cutoff every REFIT_WEEKS weeks and prices every target week from that anchor up
to the next one. The anchor depends only on the target week, so a run on a bundle cut at any week cutoff fits the
same posterior from the same rows. Fits are slow (NUTS), so a cache directory can hold them between runs; on Batch,

    python -m gnomepy_research.sessions.cs2_comp.e11.bayes_part --data <bundle> --anchors 2026-03-30,2026-04-13

fits those anchors into $OUTPUT_DIR, which predict.py then reads as its --cache.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from gnomepy_research.models.cs2.bayes.data import prepare_maps
from gnomepy_research.models.cs2.bayes.model import BayesSeriesConfig, fit_month
from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength, SeriesPosterior
from gnomepy_research.sessions.cs2_comp import bundle

ANCHOR_EPOCH = pd.Timestamp("2024-01-08")
REFIT_WEEKS = 2
STATS_COLUMNS = ["match_id", "match_date", "map_name", "player_id", "team_id", "side", "rating"]


def anchor(week_cutoff: pd.Timestamp) -> pd.Timestamp:
    week_cutoff = pd.Timestamp(week_cutoff)
    return week_cutoff - pd.Timedelta(weeks=((week_cutoff - ANCHOR_EPOCH).days // 7) % REFIT_WEEKS)


def fit_posterior(data: Path, cutoff: pd.Timestamp, cfg: BayesSeriesConfig = BayesSeriesConfig()) -> tuple[SeriesPosterior, dict]:
    history = bundle.load(data, "hltv/cs2_match_history", cutoff)
    maps = prepare_maps(history, bundle.load(data, "hltv/cs2_match_lineups", cutoff))
    stats = bundle.load(data, "hltv/cs2_player_map_stats", cutoff, STATS_COLUMNS)
    ranks = RankStrength(bundle.load(data, "hltv/cs2_team_rankings", cutoff))
    return fit_month(maps, stats, ranks, cutoff, cfg)


def posterior_at(data: Path, cutoff: pd.Timestamp, cache: Path | None) -> SeriesPosterior:
    path = cache / f"posterior_{cutoff:%Y-%m-%d}.npz" if cache is not None else None
    if path is not None and path.exists():
        return SeriesPosterior.load(path)
    t0 = time.time()
    posterior, diag = fit_posterior(data, cutoff)
    print(f"bayes fit {cutoff:%Y-%m-%d} {time.time() - t0:.0f}s {json.dumps(diag)}", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        posterior.save(path)
        (cache / f"diag_{cutoff:%Y-%m-%d}.json").write_text(json.dumps(diag | {"seconds": round(time.time() - t0)}))
    return posterior


def rank_strength_before(rankings: pd.DataFrame, cutoff: pd.Timestamp) -> RankStrength:
    return RankStrength(rankings[pd.to_datetime(rankings["date"]) < cutoff])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--anchors", required=True, help="comma-separated anchor dates to fit")
    ap.add_argument("--out", type=Path, help="default: $OUTPUT_DIR")
    a = ap.parse_args()
    out = a.out or Path(os.environ["OUTPUT_DIR"])
    for day in a.anchors.split(","):
        cutoff = pd.Timestamp(day)
        if anchor(cutoff) != cutoff:
            raise SystemExit(f"{day} is not an anchor (REFIT_WEEKS={REFIT_WEEKS})")
        posterior_at(a.data, cutoff, out)


if __name__ == "__main__":
    main()
