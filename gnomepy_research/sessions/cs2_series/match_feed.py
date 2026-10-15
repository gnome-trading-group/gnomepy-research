"""
What was known about one upcoming match at any moment: its HLTV state (kickoff, status) and its
series price, from the rows the cs2_upcoming pipeline writes.

Every lookup is as of an event time - the newest row generated at or before it - so a replay
never trades on a row before it existed, and a live session sees each row from the moment the
pipeline wrote it. Each features row is priced once, when it is first loaded.

Live, a daemon thread re-reads both tables every reload interval and swaps in a new snapshot,
so trading callbacks never wait on S3 or the model. A reload interval of 0 loads once, for
replays against a fixed table.
"""
from __future__ import annotations

import bisect
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd
from gnomepy.data import DatasetStore
from pyiceberg.expressions import EqualTo

logger = logging.getLogger(__name__)

FEATURES = "cs2_upcoming_features"
MATCHES = "cs2_upcoming_matches"

# Prices a batch of features rows: P(team A wins the series), one per row, in order.
Pricer = Callable[[pd.DataFrame], Sequence[float]]
Lineups = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class Quote:
    at_ns: int
    p_team_a: float
    # Both lineups announced: player form, which the model leans on, is blank until they are.
    lineup_known: bool
    lineups: Lineups


@dataclass(frozen=True)
class MatchState:
    at_ns: int
    kickoff_ns: int
    status: str


class Snapshot:
    """One load of a match's rows, searchable by time."""

    def __init__(self, quotes: list[Quote], states: list[MatchState]) -> None:
        self._quotes = sorted(quotes, key=lambda q: q.at_ns)
        self._states = sorted(states, key=lambda s: s.at_ns)
        self._quote_at = [q.at_ns for q in self._quotes]
        self._state_at = [s.at_ns for s in self._states]

    def quote(self, now_ns: int) -> Quote | None:
        i = bisect.bisect_right(self._quote_at, now_ns)
        return self._quotes[i - 1] if i else None

    def state(self, now_ns: int) -> MatchState | None:
        i = bisect.bisect_right(self._state_at, now_ns)
        return self._states[i - 1] if i else None


def read_rows(source: str, match_id: int) -> pd.DataFrame:
    """One match's rows of a DatasetStore table, or of a parquet extract of one (for replays)."""
    if source.endswith(".parquet"):
        rows = pd.read_parquet(source)
        return rows[rows.match_id == match_id].reset_index(drop=True)
    return DatasetStore().load(source, row_filter=EqualTo("match_id", match_id))


def _ns(ts) -> int:
    return pd.Timestamp(ts).value


def _ids(value) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    return tuple(int(x) for x in value)


class MatchFeed:
    def __init__(self, match_id: int, pricer: Pricer, features_source: str = FEATURES,
                 matches_source: str = MATCHES, reload_seconds: float = 300.0) -> None:
        self._match_id = match_id
        self._pricer = pricer
        self._features_source = features_source
        self._matches_source = matches_source
        self._reload_seconds = reload_seconds
        # Only the loading thread touches this after the first load.
        self._prices: dict[int, float] = {}
        self._snapshot = self._load()
        if reload_seconds > 0:
            threading.Thread(target=self._reload_forever, name=f"cs2-feed-{match_id}", daemon=True).start()

    def quote(self, now_ns: int) -> Quote | None:
        return self._snapshot.quote(now_ns)

    def state(self, now_ns: int) -> MatchState | None:
        return self._snapshot.state(now_ns)

    def _reload_forever(self) -> None:
        while True:
            time.sleep(self._reload_seconds)
            try:
                self._snapshot = self._load()
            except Exception:
                # A failed refresh keeps trading on the rows already held rather than stopping.
                logger.exception("reloading match %d failed", self._match_id)

    def _load(self) -> Snapshot:
        features = read_rows(self._features_source, self._match_id)
        matches = read_rows(self._matches_source, self._match_id)
        return Snapshot(self._quotes(features), [
            MatchState(_ns(r.generated_at), _ns(r.kickoff), r.status)
            for r in matches.itertuples() if not pd.isna(r.kickoff)])

    def _quotes(self, features: pd.DataFrame) -> list[Quote]:
        if features.empty:
            return []
        at = [_ns(t) for t in features.generated_at]
        fresh = [i for i, t in enumerate(at) if t not in self._prices]
        if fresh:
            prices = self._pricer(features.iloc[fresh].reset_index(drop=True))
            self._prices.update(zip((at[i] for i in fresh), (float(p) for p in prices)))
        return [Quote(t, self._prices[t], bool(r.form_known_min > 0), (_ids(r.team_a_lineup_ids), _ids(r.team_b_lineup_ids)))
                for t, r in zip(at, features.itertuples())]
