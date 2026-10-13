"""Leak and reproducibility audit on real data.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.features.leak_audit

For each audited cutoff:
1. Leak: scramble every value that became public at or after the cutoff (future results, future
   efficiency rows, future rosters, a destination not yet announced, an interim coach not yet
   hired...), rebuild the features, and require them to be identical.
2. Reproducibility: rebuild from row-shuffled tables and require identical features.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.features.cutoffs import week_cutoffs
from gnomepy_research.models.cfb.features.source import (
    GAME_RESULT_COLUMNS,
    PREGAME_COLUMNS,
    RESULT_DELAY,
    SourceData,
)
from gnomepy_research.models.cfb.features.team_features import (
    preseason_features,
    season_to_date_features,
)


def _scramble(df: pd.DataFrame, mask: pd.Series, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    idx = out.index[mask.to_numpy()]
    if len(idx) == 0:
        return out
    for col in cols:
        if col not in out.columns:
            continue
        if pd.api.types.is_bool_dtype(out[col]):
            out.loc[idx, col] = rng.random(len(idx)) < 0.5
        elif pd.api.types.is_numeric_dtype(out[col]):
            vals = out[col].to_numpy(dtype=float, na_value=np.nan)
            out[col] = vals
            out.loc[idx, col] = rng.normal(0, 100, len(idx))
        else:
            out.loc[idx, col] = rng.permutation(out.loc[idx, col].to_numpy())
    return out


def scramble_future(src: SourceData, cutoff: pd.Timestamp, seed: int = 0) -> SourceData:
    rng = np.random.default_rng(seed)
    g = src.games
    games = _scramble(g, g["kickoff"] + RESULT_DELAY >= cutoff, [*GAME_RESULT_COLUMNS, "completed"], rng)
    games = _scramble(games, games["kickoff"] >= cutoff, list(PREGAME_COLUMNS), rng)
    eff_cols = ["plays", "off_ppa", "off_pass_ppa", "off_rush_ppa", "success_rate", "explosiveness"]
    portal = _scramble(src.portal, src.portal["origin_available_at"] >= cutoff,
                       ["stars", "rating", "destination_id", "origin_id"], rng)
    portal = _scramble(portal, ~(portal["destination_available_at"] < cutoff), ["destination_id"], rng)
    coaches = _scramble(src.coaches, src.coaches["assignment_available_at"] >= cutoff, ["coach_id"], rng)
    coaches = _scramble(coaches, coaches["results_available_at"] >= cutoff,
                        ["srs", "games", "wins", "losses", "sp_overall"], rng)
    return replace(
        src,
        games=games,
        efficiency=_scramble(src.efficiency, src.efficiency["available_at"] >= cutoff, eff_cols, rng),
        passers=_scramble(src.passers, src.passers["available_at"] >= cutoff, ["passer_key", "ppa"], rng),
        talent=_scramble(src.talent, src.talent["available_at"] >= cutoff, ["talent"], rng),
        recruiting_teams=_scramble(src.recruiting_teams, src.recruiting_teams["available_at"] >= cutoff, ["points"], rng),
        recruits=_scramble(src.recruits, src.recruits["available_at"] >= cutoff, ["rating", "athlete_id"], rng),
        returning=_scramble(src.returning, src.returning["available_at"] >= cutoff, ["pct_ppa", "usage"], rng),
        portal=portal,
        coaches=coaches,
        rosters=_scramble(src.rosters, src.rosters["available_at"] >= cutoff, ["position", "player_id"], rng),
    )


def shuffle_rows(src: SourceData, seed: int = 0) -> SourceData:
    rng = np.random.default_rng(seed)
    return replace(src, **{name: df.sample(frac=1.0, random_state=rng.integers(1 << 31)).reset_index(drop=True)
                           for name, df in src.tables().items()})


def features_at(src: SourceData, cutoff: pd.Timestamp, season: int, preseason: bool) -> pd.DataFrame:
    view = src.as_of(cutoff)
    f = preseason_features(view, season) if preseason else season_to_date_features(view, season)
    return f.sort_values("team_id").reset_index(drop=True)


def audit(src: SourceData, cutoffs: pd.DataFrame) -> list[str]:
    failures = []
    for c in cutoffs.itertuples():
        first = c.week == 1 and c.season_type == "regular"
        for kind, preseason in (("preseason", True), ("season_to_date", False)):
            if preseason and not first:
                continue
            base = features_at(src, c.cutoff, c.season, preseason)
            for check, other in (("leak", scramble_future(src, c.cutoff)), ("reproducibility", shuffle_rows(src))):
                f = features_at(other, c.cutoff, c.season, preseason)
                try:
                    pd.testing.assert_frame_equal(base, f, check_dtype=False, check_exact=False, rtol=1e-9)
                    print(f"[PASS] {check:15s} {kind:14s} {c.season} {c.season_type} wk{c.week}")
                except AssertionError as e:
                    failures.append(f"{check} {kind} {c.season} wk{c.week}: {str(e)[:300]}")
                    print(f"[FAIL] {check:15s} {kind:14s} {c.season} {c.season_type} wk{c.week}\n{str(e)[:600]}")
    return failures


DEFAULT_AUDIT = [(2019, "regular", 1), (2021, "regular", 1), (2021, "regular", 7), (2023, "regular", 12),
                 (2024, "postseason", 1), (2025, "regular", 1), (2025, "regular", 9)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    src = SourceData.load()
    cuts = week_cutoffs(src.games).set_index(["season", "season_type", "week"])
    sel = cuts.loc[DEFAULT_AUDIT].reset_index()
    failures = audit(src, sel)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
