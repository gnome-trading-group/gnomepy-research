"""
The reviews of each of a session's films known at any moment, from the rt_reviews table gnome-pipelines' rt_reviews
source writes, and how many reviews each film's Tomatometer counted (rt_scores, the same source's reading of the film's
public RT page).

A row is a review once Claude has read it: is_review true and a fresh/rotten call. Its first_seen_at is when that
became known (a verdict that came after the page's first row brings a newer version of the row, seen when the verdict
was), so every lookup is as of an event time and a replay never trades on a review before it was seen. Backfilled rows
were seen when published.

A Tomatometer reading is likewise as of when it was seen (a row is written when the counts change).

Live, a daemon thread re-reads every film's rows every reload_seconds, one scan of each table for all of them, and swaps
in a new snapshot per film, so trading callbacks never wait on S3 and a session holding many films reads no more often
than one holding one. A reload interval of 0 loads once, for replays against a fixed extract. The feed opens the catalog
once and keeps it: opening it reads its configuration and looks up AWS credentials, which every few seconds for the
life of a session is needless calls to the instance's metadata service and SSM. A kept catalog still signs with
current credentials, since its signer asks its boto3 session for them on every request.
"""
from __future__ import annotations

import bisect
import logging
import threading
import time
from dataclasses import dataclass

import pandas as pd
from gnomepy.data import DatasetStore
from pyiceberg.expressions import In

logger = logging.getLogger(__name__)

REVIEWS = "rt_reviews"
SCORES = "rt_scores"


@dataclass(frozen=True)
class Review:
    url: str
    seen_ns: int
    published_ns: int | None
    fresh: bool


def read_rows(source: str, film_ids: list[str], store: DatasetStore | None = None) -> pd.DataFrame:
    """The films' rows of a table, or of a parquet extract of it (for replays). A table is read through store, or a
    new one when none is given."""
    if source.endswith(".parquet"):
        rows = pd.read_parquet(source)
        return rows[rows.film_id.isin(film_ids)].reset_index(drop=True)
    return (store or DatasetStore()).load(source, row_filter=In("film_id", film_ids))


def by_film(rows: pd.DataFrame, film_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Each film's rows, an empty frame (with the table's columns) for a film without any."""
    groups = {fid: g for fid, g in rows.groupby("film_id")} if len(rows) else {}
    return {fid: groups.get(fid, rows.iloc[0:0]) for fid in film_ids}


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


def counts_of(rows: pd.DataFrame) -> list[tuple[int, int]]:
    """(seen, reviews the Tomatometer counted) per reading, by when it was seen."""
    out = [(_ns(r.first_seen_at), int(r.liked) + int(r.not_liked)) for r in rows.itertuples()
           if _ns(r.first_seen_at) is not None and not pd.isna(r.liked) and not pd.isna(r.not_liked)]
    return sorted(out)


class Snapshot:
    def __init__(self, reviews: list[Review], counts: list[tuple[int, int]] = ()) -> None:
        self._reviews = reviews
        self._seen = [r.seen_ns for r in reviews]
        self._counts = list(counts)
        self._counted_at = [t for t, _ in self._counts]
        self._changed_at = [t for i, (t, n) in enumerate(self._counts) if n != (self._counts[i - 1][1] if i else 0)]

    def known(self, now_ns: int) -> list[Review]:
        return self._reviews[:bisect.bisect_right(self._seen, now_ns)]

    def counted(self, now_ns: int) -> int:
        i = bisect.bisect_right(self._counted_at, now_ns)
        return self._counts[i - 1][1] if i else 0

    def count_changed(self, now_ns: int) -> int | None:
        i = bisect.bisect_right(self._changed_at, now_ns)
        return self._changed_at[i - 1] if i else None


class ReviewFeed:
    def __init__(self, film_ids: list[str], source: str = REVIEWS, reload_seconds: float = 5.0,
                 scores_source: str = SCORES) -> None:
        if not film_ids:
            raise ValueError("a review feed needs at least one film")
        self._film_ids = list(dict.fromkeys(film_ids))
        self._source = source
        self._scores_source = scores_source
        self._reload_seconds = reload_seconds
        self._store: DatasetStore | None = None
        self._snapshots = self._load()
        if reload_seconds > 0:
            threading.Thread(target=self._reload_forever, name="rt-feed", daemon=True).start()

    def snapshot(self, film_id: str) -> Snapshot:
        """The film's current snapshot: a new object after every reload, so a caller can tell one has happened."""
        return self._snapshots[film_id]

    def known(self, film_id: str, now_ns: int) -> list[Review]:
        return self._snapshots[film_id].known(now_ns)

    def counted(self, film_id: str, now_ns: int) -> int:
        """How many reviews the film's Tomatometer counted, as last seen by now (0 before any reading)."""
        return self._snapshots[film_id].counted(now_ns)

    def count_changed(self, film_id: str, now_ns: int) -> int | None:
        """When the film's Tomatometer count last changed, as seen by now (its first reading counts), or None."""
        return self._snapshots[film_id].count_changed(now_ns)

    def _reload_forever(self) -> None:
        while True:
            time.sleep(self._reload_seconds)
            try:
                self._snapshots = self._load()
            except Exception:
                # A failed refresh keeps the reviews already held rather than stopping the session.
                logger.exception("reloading the reviews of %d films failed", len(self._film_ids))

    def _load(self) -> dict[str, Snapshot]:
        if self._store is None and not (self._source.endswith(".parquet") and self._scores_source.endswith(".parquet")):
            self._store = DatasetStore()
        reviews = by_film(read_rows(self._source, self._film_ids, self._store), self._film_ids)
        scores = by_film(read_rows(self._scores_source, self._film_ids, self._store), self._film_ids)
        return {fid: Snapshot(reviews_of(reviews[fid]), counts_of(scores[fid])) for fid in self._film_ids}
