"""
e11: the HLTV Bayes series price, moved by what FACEIT results say about the two fives.

    logit q = logit(q_bayes) + K_FACEIT * (logit(q_filter_faceit) - logit(q_filter_hltv))

q_bayes is the current Bayes series model (bayes_part.py) priced on each team's latest five. The two filters are the
same player-skill filter (skill.py) run over HLTV maps alone and over HLTV + FACEIT maps, priced on the same fives;
their gap is FACEIT's evidence, added to the Bayes price at a weight fitted on series before the tune period. With
--no-faceit the price is q_bayes on HLTV fives alone.

Every target is priced from data available before its week_cutoff: the filters see exactly the maps available
before it (stream.py), and the Bayes posterior is fitted at the week's anchor, at most REFIT_WEEKS - 1 weeks earlier.

    python -m gnomepy_research.sessions.cs2_comp.e11.predict --data <bundle> --targets <targets> --out <parquet> \
        [--no-faceit] [--weeks 2026-04-06,2026-04-13] [--cache <dir for fitted posteriors>]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.e11.bayes_part import anchor, posterior_at, rank_strength_before
from gnomepy_research.sessions.cs2_comp.e11.skill import FilterConfig, SkillFilter, run_stream
from gnomepy_research.sessions.cs2_comp.e11.stream import FACEIT, HLTV, build_stream, day_number

FACEIT_CONFIG = FilterConfig(w_faceit=0.5, a_faceit=-0.2, s_faceit=0.05)
HLTV_CONFIG = FilterConfig()
K_FACEIT = 1.0
UNKNOWN_BASE = -4 * 10 ** 18


def _placeholder(team: int) -> tuple[int, ...]:
    """Five never-seen players for a team with no five before the cutoff: they take the newcomer prior."""
    return tuple(UNKNOWN_BASE - 8 * team - k for k in range(5))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def filter_prices(data: Path, targets: pd.DataFrame, use_faceit: bool, cfg_f: FilterConfig = FACEIT_CONFIG,
                  cfg_h: FilterConfig = HLTV_CONFIG) -> pd.DataFrame:
    """Per target: the fives it is priced on, and the HLTV-only and (if use_faceit) HLTV + FACEIT filter prices."""
    targets = targets.assign(week_cutoff=pd.to_datetime(targets.week_cutoff))
    last = targets.week_cutoff.max()
    rankings = bundle.load(data, "hltv/cs2_team_rankings", last)
    full = build_stream(data, last, use_faceit)
    runs = [(SkillFilter(cfg_h, rankings), full.subset(full.source == HLTV), (HLTV,), not use_faceit)]
    if use_faceit:
        lineup_sources = (HLTV, FACEIT) if cfg_f.faceit_lineups > 0 else (HLTV,)
        runs.append((SkillFilter(cfg_f, rankings), full, lineup_sources, True))
    pos = [0] * len(runs)
    lineups: list[dict] = [{} for _ in runs]
    rows = []
    for cutoff, week in targets.groupby("week_cutoff", sort=True):
        stop = np.datetime64(cutoff.to_datetime64(), "ns")
        for k, (f, stream, sources, _) in enumerate(runs):
            pos[k] = run_stream(f, stream, pos[k], stop, lineups[k], sources)
        fives = lineups[[k for k, r in enumerate(runs) if r[3]][0]]
        day = day_number(cutoff)
        for r in week.itertuples():
            a, b = int(r.team_a_id), int(r.team_b_id)
            five_a = fives[a][1] if a in fives else _placeholder(a)
            five_b = fives[b][1] if b in fives else _placeholder(b)
            q = [f.price(five_a, a, five_b, b, day)[0] for f, *_ in runs]
            rows.append({"match_id": r.match_id, "fit_cutoff": cutoff, "five_a": five_a, "five_b": five_b,
                         "q_h": q[0], "q_f": q[-1]})
    return pd.DataFrame(rows)


def bayes_prices(data: Path, priced: pd.DataFrame, targets: pd.DataFrame, cache: Path | None) -> np.ndarray:
    """q_bayes for each row of `priced` (from filter_prices), from the posterior at its week's anchor."""
    team = targets.set_index("match_id")[["team_a_id", "team_b_id"]]
    rankings = bundle.load(data, "hltv/cs2_team_rankings", pd.to_datetime(targets.week_cutoff).max())
    out = np.empty(len(priced))
    for cutoff, rows in priced.groupby("fit_cutoff", sort=True):
        posterior = posterior_at(data, anchor(cutoff), cache)
        ranks = rank_strength_before(rankings, cutoff)
        for i, r in zip(rows.index, rows.itertuples()):
            a, b = (int(x) for x in team.loc[r.match_id])
            out[priced.index.get_loc(i)] = posterior.price(a, list(r.five_a), ranks(a, cutoff),
                                                           b, list(r.five_b), ranks(b, cutoff))
    return out


def predict(data: Path, targets: pd.DataFrame, use_faceit: bool, cache: Path | None = None) -> pd.DataFrame:
    p = filter_prices(data, targets, use_faceit)
    p["q_bayes"] = bayes_prices(data, p, targets, cache)
    z = _logit(p.q_bayes) + (K_FACEIT * (_logit(p.q_f) - _logit(p.q_h)) if use_faceit else 0.0)
    p["q"] = 1 / (1 + np.exp(-z))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: $OUTPUT_DIR/predictions.parquet")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--cache", type=Path, help="fitted posteriors are read from and saved to this directory")
    ap.add_argument("--cache-in-output", action="store_true", help="use $OUTPUT_DIR as the cache (Batch)")
    a = ap.parse_args()
    t0 = time.time()
    out_path = a.out or Path(os.environ["OUTPUT_DIR"]) / "predictions.parquet"
    cache = Path(os.environ["OUTPUT_DIR"]) if a.cache_in_output else a.cache
    targets = pd.read_parquet(a.targets)
    if a.weeks:
        targets = targets[pd.to_datetime(targets.week_cutoff).isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, a.faceit, cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["match_id", "q", "fit_cutoff"]].to_parquet(out_path, index=False)
    out.drop(columns=["five_a", "five_b"]).to_parquet(out_path.with_name(out_path.stem + "_parts.parquet"), index=False)
    print(f"{len(out)} predictions over {out.fit_cutoff.nunique()} weeks in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
