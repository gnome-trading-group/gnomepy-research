"""
Point-in-time scan shared by every feature block.

This exists because the leak it prevents already happened once. An earlier
lineup-strength feature froze Elo per series but let per-player ratings update
between maps, so the series DP priced node (1,0) with a prior that had already
seen map 1. The effect looked like a real 3x signal until it was traced.

Two rules make that unrepresentable rather than merely discouraged:

Snapshot once per series. An accumulator is handed the whole series at once and
emits before anything in it is observed, so there is no per-map boundary at
which it could update. Per-map outputs (map-specific ratings, say) are still
fine - they are projections of one pre-series snapshot, not separate reads.

Batch by day. match_date is date-granular and match_id is not monotone in date
(237 of 243 days overlap the next day's id range), so intra-day ordering is
arbitrary. Every series on a date is scored against the state that stood at the
start of that date, and sibling matches cannot leak into each other.

The invariant is checkable, not just asserted: assert_pit_consistent recomputes
a series against a history truncated to strictly-earlier days and requires the
emitted features to be identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_KEY = ("match_id", "map_name")


@dataclass(frozen=True)
class SeriesBatch:
    """One series, plus its slice of every auxiliary dataset."""

    match_id: int
    match_date: pd.Timestamp
    rows: pd.DataFrame
    aux: dict[str, pd.DataFrame]

    def aux_frame(self, name: str) -> pd.DataFrame:
        return self.aux.get(name, _EMPTY)


_EMPTY = pd.DataFrame()


@runtime_checkable
class PitAccumulator(Protocol):
    """
    Stateful history walker.

    snapshot reads current state for a series and must not mutate it; update
    folds that series in. The driver guarantees every snapshot on a date runs
    before every update on that date.

    snapshot returns either a dict (broadcast to each map row in the series) or
    a list of dicts already carrying the key columns.
    """

    def snapshot(self, batch: SeriesBatch) -> dict | list[dict]: ...

    def update(self, batch: SeriesBatch) -> None: ...


def _group_aux(aux: dict[str, pd.DataFrame] | None) -> dict[str, dict]:
    if not aux:
        return {}
    grouped = {}
    for name, frame in aux.items():
        if frame is None or frame.empty or "match_id" not in frame.columns:
            grouped[name] = {}
            continue
        grouped[name] = {mid: sub for mid, sub in frame.groupby("match_id", sort=False)}
    return grouped


def pit_scan(
    history: pd.DataFrame,
    accumulator: PitAccumulator,
    *,
    aux: dict[str, pd.DataFrame] | None = None,
    key: tuple[str, ...] = _DEFAULT_KEY,
) -> pd.DataFrame:
    """
    Walk history in date order, emitting pre-series features for every map row.

    Returns a frame keyed by `key`. Emitting a whole day before observing any of
    it is what makes the output causal.
    """
    if history.empty:
        return pd.DataFrame(columns=list(key))

    grouped_aux = _group_aux(aux)
    sort_cols = [c for c in ("match_date", "match_id", "map_position_in_series") if c in history.columns]
    hist = history.sort_values(sort_cols).reset_index(drop=True)

    out: list[dict] = []
    for match_date, day in hist.groupby("match_date", sort=True):
        batches = [
            SeriesBatch(
                match_id=mid,
                match_date=match_date,
                rows=rows,
                aux={name: by_id.get(mid, _EMPTY) for name, by_id in grouped_aux.items()},
            )
            for mid, rows in day.groupby("match_id", sort=True)
        ]
        for batch in batches:
            emitted = accumulator.snapshot(batch)
            if isinstance(emitted, dict):
                for row in batch.rows.itertuples():
                    out.append({**{k: getattr(row, k) for k in key}, **emitted})
            else:
                out.extend(emitted)
        for batch in batches:
            accumulator.update(batch)

    frame = pd.DataFrame(out)
    logger.info("pit_scan: emitted %d rows over %d series", len(frame), hist["match_id"].nunique())
    return frame




def _diff_cols(expected: pd.DataFrame, actual: pd.DataFrame) -> list[str]:
    actual = actual.reindex(columns=expected.columns)
    mismatch = ~((expected == actual) | (expected.isna() & actual.isna()))
    return mismatch.columns[mismatch.any(axis=0)].tolist()


def assert_pit_consistent(
    history: pd.DataFrame,
    make_accumulator,
    *,
    aux: dict[str, pd.DataFrame] | None = None,
    key: tuple[str, ...] = _DEFAULT_KEY,
    sample: int = 8,
    seed: int = 0,
    outcome_aux: tuple[str, ...] = (),
) -> None:
    """
    Verify a series reads neither its own result nor any later one.

    Two checks, because neither alone is sufficient. Truncation - recompute
    against strictly-earlier days plus the series itself - catches reading
    forward. It cannot catch a series reading its own result, since that result
    survives truncation; the most dangerous leak we have actually hit was of
    exactly that kind, map 1 feeding map 2 of the same series. So the second
    check perturbs the series' own outcome columns and named post-hoc aux
    frames (`outcome_aux`) and requires its features not to move.

    Aux frames known before the match - veto, h2h history, announced lineups -
    are legitimately readable and must stay out of `outcome_aux`.

    `make_accumulator` is a zero-arg factory: each run needs fresh state.
    """
    full = pit_scan(history, make_accumulator(), aux=aux, key=key).set_index(list(key)).sort_index()

    dates = history[["match_id", "match_date"]].drop_duplicates()
    candidates = dates[dates["match_date"] > dates["match_date"].min()]
    if candidates.empty:
        return
    picks = candidates.sample(n=min(sample, len(candidates)), random_state=seed)

    for match_id, match_date in picks.itertuples(index=False):
        idx = [i for i in full.index if i[0] == match_id]
        if not idx:
            continue
        expected = full.loc[idx]

        truncated = history[
            (history["match_date"] < match_date) | (history["match_id"] == match_id)
        ]
        partial = pit_scan(truncated, make_accumulator(), aux=aux, key=key)
        if not partial.empty:
            partial = partial.set_index(list(key)).sort_index()
            shared = [i for i in idx if i in partial.index]
            if shared:
                cols = _diff_cols(expected.loc[shared], partial.loc[shared])
                if cols:
                    raise AssertionError(
                        f"lookahead in series {match_id} on {match_date.date()}: "
                        f"columns {cols} differ when later matches are withheld"
                    )

        perturbed = history.copy()
        mask = perturbed["match_id"] == match_id
        for a_col, b_col in (("team_a_score", "team_b_score"), ("team_a_h1_score", "team_b_h1_score")):
            if a_col in perturbed.columns and b_col in perturbed.columns:
                a_vals = perturbed.loc[mask, a_col].to_numpy().copy()
                perturbed.loc[mask, a_col] = perturbed.loc[mask, b_col].to_numpy()
                perturbed.loc[mask, b_col] = a_vals
        for col in ("team_a_won", "team_a_started_ct"):
            if col in perturbed.columns:
                perturbed.loc[mask, col] = 1 - perturbed.loc[mask, col].astype(float)

        perturbed_aux = dict(aux or {})
        for name in outcome_aux:
            frame = perturbed_aux.get(name)
            if frame is not None and not frame.empty and "match_id" in frame.columns:
                perturbed_aux[name] = frame[frame["match_id"] != match_id]

        shifted = pit_scan(perturbed, make_accumulator(), aux=perturbed_aux, key=key)
        if shifted.empty:
            continue
        shifted = shifted.set_index(list(key)).sort_index()
        shared = [i for i in idx if i in shifted.index]
        if not shared:
            continue
        cols = _diff_cols(expected.loc[shared], shifted.loc[shared])
        if cols:
            raise AssertionError(
                f"self-read in series {match_id} on {match_date.date()}: "
                f"columns {cols} move when the series' own result is perturbed"
            )
