"""
What the model said about each game of the slate at any moment, from the rows cfb_pregame_predictions holds.

The predictions are B0's pregame win probabilities, refreshed on a schedule (gnome-pipelines' cfb/pregame.py publishes
them; gnomepy-research's cfb_win_probability live predictor computes them). Every lookup is as of an event time - the
newest row generated at or before it - so a replay never trades on a prediction before it existed.

Live, a daemon thread re-reads the table every reload interval and swaps in a new snapshot, so trading callbacks never
wait on S3. A failed re-read keeps the snapshot already held. A reload interval of 0 loads once, for replays.
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

PREDICTIONS = "cfb_pregame_predictions"


@dataclass(frozen=True)
class Prediction:
    at_ns: int
    game_id: int
    kickoff_ns: int
    week: int
    p_home: float
    # The QB status service's p_start for the starter the model assumed; NaN when it had no snapshot for the team.
    home_qb_p: float
    away_qb_p: float
    # B0's strength rank among FBS teams at the game's cutoff (1 = best), None when unranked.
    home_rank: int | None
    away_rank: int | None
    fbs_teams: int
    home_newcomer: bool
    away_newcomer: bool


def read_rows(source: str, game_ids: list[int]) -> pd.DataFrame:
    """The slate's rows of the predictions table, or of a parquet extract of it (for replays)."""
    if source.endswith(".parquet"):
        rows = pd.read_parquet(source)
        return rows[rows.game_id.isin(game_ids)].reset_index(drop=True)
    return DatasetStore().load(source, row_filter=In("game_id", [int(g) for g in game_ids]))


def _rank(value) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _prediction(r) -> Prediction:
    return Prediction(
        at_ns=pd.Timestamp(r.generated_at).value, game_id=int(r.game_id), kickoff_ns=pd.Timestamp(r.kickoff).value,
        week=int(r.week), p_home=float(r.p_home), home_qb_p=float(r.home_qb_p), away_qb_p=float(r.away_qb_p),
        home_rank=_rank(r.home_rank), away_rank=_rank(r.away_rank), fbs_teams=int(r.fbs_teams),
        home_newcomer=bool(r.home_newcomer), away_newcomer=bool(r.away_newcomer))


class Snapshot:
    """One load of the slate's rows, searchable by game and time."""

    def __init__(self, predictions: list[Prediction]) -> None:
        self._by_game: dict[int, list[Prediction]] = {}
        for p in sorted(predictions, key=lambda p: p.at_ns):
            self._by_game.setdefault(p.game_id, []).append(p)
        self._at = {g: [p.at_ns for p in ps] for g, ps in self._by_game.items()}

    def latest(self, game_id: int, now_ns: int) -> Prediction | None:
        at = self._at.get(game_id)
        if not at:
            return None
        i = bisect.bisect_right(at, now_ns)
        return self._by_game[game_id][i - 1] if i else None

    def kickoffs(self) -> dict[int, int]:
        return {g: ps[-1].kickoff_ns for g, ps in self._by_game.items()}


class PredictionFeed:
    def __init__(self, game_ids: list[int], source: str = PREDICTIONS, reload_seconds: float = 300.0) -> None:
        self._game_ids = [int(g) for g in game_ids]
        self._source = source
        self._reload_seconds = reload_seconds
        self._snapshot = self._load()
        if reload_seconds > 0:
            threading.Thread(target=self._reload_forever, name="cfb-predictions", daemon=True).start()

    def latest(self, game_id: int, now_ns: int) -> Prediction | None:
        return self._snapshot.latest(game_id, now_ns)

    def kickoffs(self) -> dict[int, int]:
        """The newest known kickoff per game, whatever the time: the cutoffs follow a moved kickoff at once."""
        return self._snapshot.kickoffs()

    def _reload_forever(self) -> None:
        while True:
            time.sleep(self._reload_seconds)
            try:
                self._snapshot = self._load()
            except Exception:
                logger.exception("reloading the slate's predictions failed")

    def _load(self) -> Snapshot:
        rows = read_rows(self._source, self._game_ids)
        return Snapshot([_prediction(r) for r in rows.itertuples() if not pd.isna(r.kickoff)])
