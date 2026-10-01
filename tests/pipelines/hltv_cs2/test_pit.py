import pandas as pd
import pytest

from gnomepy_research.pipelines.hltv_cs2.pit import (
    SeriesBatch,
    assert_pit_consistent,
    pit_scan,
)


def _history() -> pd.DataFrame:
    rows = []
    match_id = 1000
    for day in range(1, 7):
        for _ in range(2):
            match_id += 1
            for pos in range(1, 4):
                rows.append({
                    "match_id": match_id,
                    "match_date": pd.Timestamp(f"2026-01-0{day}"),
                    "map_name": f"de_map{pos}",
                    "map_position_in_series": pos,
                    "team_a_id": match_id % 4,
                    "team_b_id": (match_id + 1) % 4,
                    "team_a_won": pos % 2,
                })
    return pd.DataFrame(rows)


class CausalWinCount:
    """Counts wins per team, frozen per series, folded in a day at a time."""

    def __init__(self):
        self.wins: dict[int, int] = {}

    def snapshot(self, batch: SeriesBatch) -> dict:
        first = batch.rows.iloc[0]
        return {
            "a_wins": self.wins.get(first["team_a_id"], 0),
            "b_wins": self.wins.get(first["team_b_id"], 0),
        }

    def update(self, batch: SeriesBatch) -> None:
        for row in batch.rows.itertuples():
            winner = row.team_a_id if row.team_a_won else row.team_b_id
            self.wins[winner] = self.wins.get(winner, 0) + 1


class LeaksOwnSeries(CausalWinCount):
    """Reads the result of the series it is pricing."""

    def snapshot(self, batch: SeriesBatch) -> dict:
        base = super().snapshot(batch)
        base["a_wins"] += int(batch.rows["team_a_won"].sum())
        return base


class MutatesInSnapshot(CausalWinCount):
    """Updates during the emit pass, so same-day siblings leak into each other."""

    def snapshot(self, batch: SeriesBatch) -> dict:
        base = super().snapshot(batch)
        self.update(batch)
        return base

    def update(self, batch: SeriesBatch) -> None:
        if getattr(self, "_in_snapshot", False):
            return
        self._in_snapshot = True
        super().update(batch)
        self._in_snapshot = False


def test_scan_emits_one_row_per_map():
    hist = _history()
    out = pit_scan(hist, CausalWinCount())
    assert len(out) == len(hist)
    assert set(out.columns) == {"match_id", "map_name", "a_wins", "b_wins"}


def test_dict_snapshot_is_frozen_across_maps_in_a_series():
    out = pit_scan(_history(), CausalWinCount())
    per_series = out.groupby("match_id")[["a_wins", "b_wins"]].nunique()
    assert (per_series == 1).all().all(), "a broadcast snapshot must not vary by map"


def test_first_day_sees_nothing():
    hist = _history()
    out = pit_scan(hist, CausalWinCount()).merge(
        hist[["match_id", "map_name", "match_date"]], on=["match_id", "map_name"]
    )
    first = out[out["match_date"] == hist["match_date"].min()]
    assert (first[["a_wins", "b_wins"]] == 0).all().all()


def test_same_day_siblings_do_not_see_each_other():
    hist = _history()
    out = pit_scan(hist, CausalWinCount()).merge(
        hist[["match_id", "map_name", "match_date"]], on=["match_id", "map_name"]
    )
    for _, day in out.groupby("match_date"):
        for col in ("a_wins", "b_wins"):
            by_series = day.groupby("match_id")[col].first()
            teams = hist.groupby("match_id")["team_a_id"].first()
            shared = teams[teams.index.isin(by_series.index)]
            for team in shared.unique():
                vals = by_series[shared[shared == team].index].unique()
                assert len(vals) == 1, "same-day series disagree on a team's pre-day state"


def test_causal_accumulator_passes_the_check():
    assert_pit_consistent(_history(), CausalWinCount)


def test_own_series_leak_is_caught():
    """Truncation cannot see this one - the series' own result survives it."""
    with pytest.raises(AssertionError, match="self-read in series"):
        assert_pit_consistent(_history(), LeaksOwnSeries)


def test_intra_day_leak_is_caught():
    with pytest.raises(AssertionError, match="lookahead in series"):
        assert_pit_consistent(_history(), MutatesInSnapshot)


def test_aux_frames_are_sliced_per_series():
    hist = _history()
    stats = pd.DataFrame([
        {"match_id": mid, "player_id": p, "rating": 1.0}
        for mid in hist["match_id"].unique() for p in range(5)
    ])
    seen = {}

    class Recorder(CausalWinCount):
        def snapshot(self, batch: SeriesBatch) -> dict:
            seen[batch.match_id] = batch.aux_frame("stats")
            return super().snapshot(batch)

    pit_scan(hist, Recorder(), aux={"stats": stats})
    assert len(seen) == hist["match_id"].nunique()
    for mid, frame in seen.items():
        assert len(frame) == 5
        assert set(frame["match_id"]) == {mid}


def test_missing_aux_is_an_empty_frame_not_an_error():
    hist = _history()
    seen = []

    class Recorder(CausalWinCount):
        def snapshot(self, batch: SeriesBatch) -> dict:
            seen.append(batch.aux_frame("absent"))
            return super().snapshot(batch)

    pit_scan(hist, Recorder(), aux={"stats": pd.DataFrame()})
    assert all(f.empty for f in seen)


def test_empty_history_returns_keyed_frame():
    out = pit_scan(pd.DataFrame(), CausalWinCount())
    assert out.empty
    assert list(out.columns) == ["match_id", "map_name"]
