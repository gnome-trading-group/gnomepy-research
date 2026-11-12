"""
The reviews of one film known at any moment, from the rt_reviews table gnome-pipelines' rt_reviews source writes.

A row is a review once Claude has read it: is_review true and a fresh/rotten call. Its first_seen_at is when that
became known (a verdict that came after the page's first row brings a newer version of the row, seen when the verdict
was), so every lookup is as of an event time and a replay never trades on a review before it was seen. Backfilled rows
were seen when published.

Live, a daemon thread re-reads the film's rows every reload_seconds and swaps in a new snapshot, so trading callbacks
never wait on S3. A reload interval of 0 loads once, for replays against a fixed extract.
"""
from __future__ import annotations

import bisect
import logging
import threading
import time

import pandas as pd
from gnomepy.data import DatasetStore
from pyiceberg.expressions import EqualTo

from gnomepy_research.sessions.rt_review_wave.decision import Review

logger = logging.getLogger(__name__)

REVIEWS = "rt_reviews"


def read_rows(source: str, film_id: str) -> pd.DataFrame:
    """One film's rows of the rt_reviews table (the newest version of each), or of a parquet extract (for replays)."""
    if source.endswith(".parquet"):
        rows = pd.read_parquet(source)
        return rows[rows.film_id == film_id].reset_index(drop=True)
    return DatasetStore().load(source, row_filter=EqualTo("film_id", film_id))


def _ns(ts) -> int | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).value


def _known_true(value) -> bool:
    # A row without a verdict holds a null (None, NaN or pd.NA, by the column's dtype), which bool() can't take.
    return value is not None and not pd.isna(value) and bool(value)


def reviews_of(rows: pd.DataFrame) -> list[Review]:
    """The rows that are reviews with a verdict, newest version per URL, ordered by when they were seen."""
    if rows.empty:
        return []
    rows = rows.sort_values("first_seen_at", kind="stable").drop_duplicates("url", keep="last")
    out = []
    for r in rows.itertuples():
        seen = _ns(r.first_seen_at)
        if seen is None or not _known_true(r.is_review) or r.fresh is None or pd.isna(r.fresh):
            continue
        out.append(Review(r.url, seen, _ns(r.published_at), bool(r.fresh)))
    return sorted(out, key=lambda x: x.seen_ns)


class Snapshot:
    def __init__(self, reviews: list[Review]) -> None:
        self._reviews = reviews
        self._seen = [r.seen_ns for r in reviews]
        self.key = tuple(reviews)

    def known(self, now_ns: int) -> list[Review]:
        return self._reviews[:bisect.bisect_right(self._seen, now_ns)]


class ReviewFeed:
    def __init__(self, film_id: str, source: str = REVIEWS, reload_seconds: float = 5.0) -> None:
        self._film_id = film_id
        self._source = source
        self._reload_seconds = reload_seconds
        self._snapshot = self._load()
        if reload_seconds > 0:
            threading.Thread(target=self._reload_forever, name=f"rt-feed-{film_id}", daemon=True).start()

    def known(self, now_ns: int) -> list[Review]:
        return self._snapshot.known(now_ns)

    def _reload_forever(self) -> None:
        while True:
            time.sleep(self._reload_seconds)
            try:
                self._snapshot = self._load()
            except Exception:
                # A failed refresh keeps the reviews already held rather than stopping the session.
                logger.exception("reloading the reviews of %s failed", self._film_id)

    def _load(self) -> Snapshot:
        return Snapshot(reviews_of(read_rows(self._source, self._film_id)))
