"""
Compute point-in-time team priors from cs2_match_history and cs2_team_rankings datasets.

For each map in cs2_match_history, computes prior features using only
data strictly before that match's date — no lookahead bias.

Output dataset: cs2_match_priors
  match_id, map_name, team_a_name, team_b_name,
  team_a_rating_diff,
  team_a_rank, team_b_rank, rank_diff,
  team_a_map_winrate_long, team_b_map_winrate_long,   (last 20 maps, max 6 months)
  team_a_map_winrate_short, team_b_map_winrate_short, (last 5 maps, max 3 months)
  team_a_overall_winrate, team_b_overall_winrate,     (last 20 maps all maps, max 6 months)
  h2h_win_rate,                                       (last 10 maps, max 1 year)
  recent_form_diff,                                   (last 10 maps, max 2 months)
  team_a_picked_map, is_lan, is_decider, bo_type, map_position_in_series,
  event_tier, team_a_series_score, team_b_series_score
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime
import logging

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.models.cs2.hltv.context_features import (
    EVENT_FEATURES,
    PAGE_H2H_FEATURES,
    SCHEDULE_FEATURES,
    VETO_FEATURES,
    compute_event_features,
    compute_page_h2h_features,
    compute_schedule_features,
    compute_veto_features,
)
from gnomepy_research.models.cs2.hltv.elo import compute_elo_features
from gnomepy_research.models.cs2.hltv.pit import SeriesBatch, pit_scan
from gnomepy_research.models.cs2.hltv.player_skill import (
    PLAYER_SKILL_FEATURES,
    compute_player_skill_features,
)
from gnomepy_research.models.cs2.hltv.roster_features import ROSTER_FEATURES, compute_roster_features

logger = logging.getLogger(__name__)

_MAP_WIN_RATE_WINDOW_LONG = 20
_MAP_WIN_RATE_WINDOW_SHORT = 5
_RECENT_FORM_WINDOW = 10
_H2H_WINDOW = 10
_OVERALL_WINRATE_WINDOW = 20

_MAP_WIN_RATE_MAX_AGE_LONG = datetime.timedelta(days=180)
_MAP_WIN_RATE_MAX_AGE_SHORT = datetime.timedelta(days=90)
_RECENT_FORM_MAX_AGE = datetime.timedelta(days=60)
_H2H_MAX_AGE = datetime.timedelta(days=365)
_OVERALL_WINRATE_MAX_AGE = datetime.timedelta(days=180)
_RANKING_MAX_AGE = datetime.timedelta(days=120)


def _team_wins(sub: pd.DataFrame, team_id: int) -> int:
    return int(
        np.where(sub["team_a_id"].values == team_id, sub["team_a_won"].values, 1 - sub["team_a_won"].values).sum()
    )


def _plays_mask(history: pd.DataFrame, team_id: int) -> np.ndarray:
    return (history["team_a_id"].values == team_id) | (history["team_b_id"].values == team_id)


def _window_winrate(history: pd.DataFrame, mask: np.ndarray, team_id: int, window: int) -> float:
    sub = history[mask].tail(window)
    if len(sub) == 0:
        return float("nan")
    return float(_team_wins(sub, team_id) / len(sub))


def _team_map_winrate(
    history: pd.DataFrame,
    team_id: int,
    map_name: str,
    before_date: pd.Timestamp,
    window: int,
    max_age: datetime.timedelta,
) -> float:
    """Win rate on map_name over the last `window` maps strictly before before_date, within max_age."""
    mask = (
        _plays_mask(history, team_id) &
        (history["map_name"].values == map_name) &
        (history["match_date"].values < before_date) &
        (history["match_date"].values >= before_date - max_age)
    )
    return _window_winrate(history, mask, team_id, window)


def _team_recent_form(history: pd.DataFrame, team_id: int, before_date: pd.Timestamp) -> float:
    mask = (
        _plays_mask(history, team_id) &
        (history["match_date"].values < before_date) &
        (history["match_date"].values >= before_date - _RECENT_FORM_MAX_AGE)
    )
    return _window_winrate(history, mask, team_id, _RECENT_FORM_WINDOW)


def _team_overall_winrate(history: pd.DataFrame, team_id: int, before_date: pd.Timestamp) -> float:
    mask = (
        _plays_mask(history, team_id) &
        (history["match_date"].values < before_date) &
        (history["match_date"].values >= before_date - _OVERALL_WINRATE_MAX_AGE)
    )
    return _window_winrate(history, mask, team_id, _OVERALL_WINRATE_WINDOW)


def _h2h_winrate(
    history: pd.DataFrame,
    team_a_id: int,
    team_b_id: int,
    before_date: pd.Timestamp,
) -> float:
    a, b = history["team_a_id"].values, history["team_b_id"].values
    mask = (
        (((a == team_a_id) & (b == team_b_id)) | ((a == team_b_id) & (b == team_a_id))) &
        (history["match_date"].values < before_date) &
        (history["match_date"].values >= before_date - _H2H_MAX_AGE)
    )
    return _window_winrate(history, mask, team_a_id, _H2H_WINDOW)


def _ranking_lookup(
    rankings: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
    max_age: datetime.timedelta = _RANKING_MAX_AGE,
) -> tuple[float, float, float]:
    """
    Most recent (points, rank, age_days) published strictly before before_date.

    Ranks older than max_age are treated as unknown rather than carried forward —
    a disbanded or inactive team would otherwise keep an elite rank indefinitely.
    """
    mask = (rankings["team_id"].values == team_id) & (rankings["date"].values < before_date)
    sub = rankings[mask]
    if len(sub) == 0:
        return float("nan"), float("nan"), float("nan")
    latest = sub.iloc[-1]
    age_days = float((before_date - pd.Timestamp(latest["date"])).days)
    if age_days > max_age.days:
        return float("nan"), float("nan"), age_days
    return float(latest["points"]), float(latest["rank"]), age_days


def _resolve_team_id(history: pd.DataFrame, team_name: str) -> int | None:
    """Map a display name to the most recent team_id seen for it, for live callers keyed on names."""
    lowered = team_name.lower()
    for name_col, id_col in (("team_a_name", "team_a_id"), ("team_b_name", "team_b_id")):
        sub = history[history[name_col].str.lower() == lowered]
        if len(sub):
            return int(sub.iloc[-1][id_col])
    logger.warning("could not resolve team_id for %r — priors will be NaN", team_name)
    return None


class _TeamHistoryAccumulator:
    """
    Rankings, map and overall win rates, head-to-head and recent form for every map row, in one pass.

    These were once recomputed per row by masking the whole history, which cost the square of its length
    (minutes for 33k rows, worse with every backfilled year). The windows only ever look at a team's last 20
    maps, a (team, map)'s last 20 or a pair's last 10, so bounded queues per key give the same answers. Each
    window is "the last N maps strictly before the day, within max age": the queues hold maps from earlier days
    only, so dropping the over-age ones from the last N is the same as taking the last N of the in-age ones.
    """

    def __init__(self, rankings: pd.DataFrame):
        self._team: dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=_OVERALL_WINRATE_WINDOW))
        self._team_map: dict[tuple, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=_MAP_WIN_RATE_WINDOW_LONG))
        self._pair: dict[tuple, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=_H2H_WINDOW))
        self._rank_dates: dict[int, list] = collections.defaultdict(list)
        self._rank_rows: dict[int, list] = collections.defaultdict(list)
        for r in rankings.itertuples():
            self._rank_dates[int(r.team_id)].append(pd.Timestamp(r.date))
            self._rank_rows[int(r.team_id)].append((float(r.points), float(r.rank), pd.Timestamp(r.date)))

    def _ranking(self, team_id: int, before: pd.Timestamp) -> tuple[float, float, float]:
        i = bisect.bisect_left(self._rank_dates.get(team_id, []), before) - 1
        if i < 0:
            return float("nan"), float("nan"), float("nan")
        points, rank, date = self._rank_rows[team_id][i]
        age_days = float((before - date).days)
        if age_days > _RANKING_MAX_AGE.days:
            return float("nan"), float("nan"), age_days
        return points, rank, age_days

    @staticmethod
    def _rate(results, before: pd.Timestamp, window: int, max_age: datetime.timedelta) -> float:
        cutoff = before - max_age
        recent = [won for date, won in list(results)[-window:] if date >= cutoff]
        if not recent:
            return float("nan")
        return float(int(sum(recent)) / len(recent))

    def _h2h(self, team_a: int, team_b: int, before: pd.Timestamp) -> float:
        meetings = self._pair.get(tuple(sorted((team_a, team_b))), ())
        as_a = [(date, won if a == team_a else 1 - won) for date, a, won in meetings]
        return self._rate(as_a, before, _H2H_WINDOW, _H2H_MAX_AGE)

    def snapshot(self, batch: SeriesBatch) -> list[dict]:
        day = pd.Timestamp(batch.match_date)
        out = []
        for row in batch.records:
            id_a, id_b = int(row.team_a_id), int(row.team_b_id)
            map_name = row.map_name
            pts_a, rank_a, age_a = self._ranking(id_a, day)
            pts_b, rank_b, age_b = self._ranking(id_b, day)
            form_a = self._rate(self._team.get(id_a, ()), day, _RECENT_FORM_WINDOW, _RECENT_FORM_MAX_AGE)
            form_b = self._rate(self._team.get(id_b, ()), day, _RECENT_FORM_WINDOW, _RECENT_FORM_MAX_AGE)

            def map_rate(team_id, window, max_age):
                if not isinstance(map_name, str):
                    return float("nan")
                return self._rate(self._team_map.get((team_id, map_name), ()), day, window, max_age)

            out.append({
                "match_id": row.match_id,
                "map_name": map_name,
                "match_date": day,
                "team_a_name": row.team_a_name,
                "team_b_name": row.team_b_name,
                "team_a_id": id_a,
                "team_b_id": id_b,
                "team_a_rating_diff": pts_a - pts_b,
                "team_a_rank": rank_a,
                "team_b_rank": rank_b,
                "rank_diff": rank_a - rank_b,
                "ranking_age_days": np.nanmax([age_a, age_b]) if not (np.isnan(age_a) and np.isnan(age_b)) else float("nan"),
                "team_a_map_winrate_long": map_rate(id_a, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
                "team_b_map_winrate_long": map_rate(id_b, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
                "team_a_map_winrate_short": map_rate(id_a, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
                "team_b_map_winrate_short": map_rate(id_b, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
                "team_a_overall_winrate": self._rate(self._team.get(id_a, ()), day, _OVERALL_WINRATE_WINDOW, _OVERALL_WINRATE_MAX_AGE),
                "team_b_overall_winrate": self._rate(self._team.get(id_b, ()), day, _OVERALL_WINRATE_WINDOW, _OVERALL_WINRATE_MAX_AGE),
                "h2h_win_rate": self._h2h(id_a, id_b, day),
                "team_a_recent_form": form_a,
                "team_b_recent_form": form_b,
                "recent_form_diff": form_a - form_b,
                "team_a_picked_map": float(getattr(row, "team_a_picked_map", float("nan"))),
                "is_lan": float(getattr(row, "is_lan", float("nan"))),
                "is_decider": float(getattr(row, "is_decider", float("nan"))),
                "bo_type": float(getattr(row, "bo_type", float("nan"))),
                "map_position_in_series": float(getattr(row, "map_position_in_series", float("nan"))),
                "event_tier": float(getattr(row, "event_tier", float("nan"))),
                "team_a_series_score": float(getattr(row, "team_a_series_score", 0)),
                "team_b_series_score": float(getattr(row, "team_b_series_score", 0)),
            })
        return out

    def update(self, batch: SeriesBatch) -> None:
        day = pd.Timestamp(batch.match_date)
        for row in batch.records:
            id_a, id_b, won = int(row.team_a_id), int(row.team_b_id), float(row.team_a_won)
            self._team[id_a].append((day, won))
            self._team[id_b].append((day, 1 - won))
            if isinstance(row.map_name, str):
                self._team_map[(id_a, row.map_name)].append((day, won))
                self._team_map[(id_b, row.map_name)].append((day, 1 - won))
            self._pair[tuple(sorted((id_a, id_b)))].append((day, id_a, won))


def compute_team_history_priors(history: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Rankings and team-history priors for every map row with a date, keyed (match_id, map_name)."""
    dated = history[history["match_date"].notna()]
    return pit_scan(dated, _TeamHistoryAccumulator(rankings.sort_values("date").reset_index(drop=True)))


def _frame(value: pd.DataFrame | None) -> pd.DataFrame:
    return value if value is not None else pd.DataFrame()


def _merge_block(base: pd.DataFrame, block: pd.DataFrame, declared: list[str]) -> pd.DataFrame:
    """
    Left-merge a feature block, guaranteeing every declared column exists.

    A block whose source dataset is absent returns no rows; the extractor treats
    an absent key as a bug rather than a missing value, so the columns are
    materialised as NaN instead of silently vanishing.
    """
    if not block.empty:
        base = base.merge(block, on=["match_id", "map_name"], how="left")
    for col in declared:
        if col not in base.columns:
            base[col] = np.nan
    return base


def build_priors(
    match_history: pd.DataFrame,
    team_rankings: pd.DataFrame,
    *,
    lineups: pd.DataFrame | None = None,
    roster_moves: pd.DataFrame | None = None,
    veto: pd.DataFrame | None = None,
    h2h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute point-in-time priors for every row in match_history.
    Returns a DataFrame with one row per map, keyed by match_id + map_name.
    """
    history_sorted = match_history[~showmatch_mask(match_history)].sort_values("match_date").reset_index(drop=True)
    rankings = team_rankings.sort_values("date").reset_index(drop=True)

    logger.info("Computing priors for %d map rows...", len(history_sorted))
    priors = compute_team_history_priors(history_sorted, rankings)
    elo = compute_elo_features(history_sorted, rankings)
    out = priors.merge(elo, on=["match_id", "map_name"], how="left")

    blocks = [
        (compute_player_skill_features(history_sorted, rankings, _frame(lineups)), PLAYER_SKILL_FEATURES),
        (compute_roster_features(history_sorted, _frame(lineups), _frame(roster_moves)), ROSTER_FEATURES),
        (compute_schedule_features(history_sorted), SCHEDULE_FEATURES),
        (compute_event_features(history_sorted), EVENT_FEATURES),
        (compute_veto_features(history_sorted, _frame(veto)), VETO_FEATURES),
        (compute_page_h2h_features(history_sorted, _frame(h2h)), PAGE_H2H_FEATURES),
    ]
    for block, declared in blocks:
        out = _merge_block(out, block, declared)
    return out


def build_priors_for(
    match_history: pd.DataFrame,
    team_rankings: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    lineups: pd.DataFrame | None = None,
    roster_moves: pd.DataFrame | None = None,
    veto: pd.DataFrame | None = None,
    h2h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Priors for the target rows only - upcoming maps whose outcome is unknown -
    computed exactly as build_priors would compute them with the targets appended
    to history.

    The point-in-time scans are linear and run over everything; the legacy per-row
    priors are quadratic, so they run for the targets alone (seconds, not the
    minutes a full rebuild takes every 30 min).

    Each day's targets are priced against history plus that day's targets only.
    A target's outcome is unknown, so a later day's target must never see an
    earlier one as played: its missing result would count as a win in h2h, turn
    Elo into NaN, and crash recent form. Same-day targets can share a pass because
    the scans score a whole day against the state at its start.
    """
    rankings = team_rankings.sort_values("date").reset_index(drop=True)
    match_history = match_history[~showmatch_mask(match_history)]
    targets = targets.assign(match_date=pd.to_datetime(targets["match_date"]))
    days = [
        _priors_for_day(match_history, rankings, day_targets, lineups, roster_moves, veto, h2h)
        for _, day_targets in targets.groupby("match_date", sort=True)
    ]
    days = [d for d in days if not d.empty]
    return pd.concat(days, ignore_index=True) if days else pd.DataFrame()


def _priors_for_day(
    match_history: pd.DataFrame,
    rankings: pd.DataFrame,
    targets: pd.DataFrame,
    lineups: pd.DataFrame | None,
    roster_moves: pd.DataFrame | None,
    veto: pd.DataFrame | None,
    h2h: pd.DataFrame | None,
) -> pd.DataFrame:
    # HLTV keeps a delayed match on its scheduled day, so a target can predate history already played; the scans
    # would then carry its unknown result into those later days.
    day = targets["match_date"].iloc[0]
    played = match_history[pd.to_datetime(match_history["match_date"]) <= day]
    combined = pd.concat([played, targets], ignore_index=True)
    combined["match_date"] = pd.to_datetime(combined["match_date"])
    combined = combined.sort_values("match_date").reset_index(drop=True)
    keys = set(zip(targets.match_id, targets.map_name))

    team_history = compute_team_history_priors(combined, rankings)
    out = team_history[[k in keys for k in zip(team_history.match_id, team_history.map_name)]].reset_index(drop=True)
    if out.empty:
        return out
    out = out.merge(compute_elo_features(combined, rankings), on=["match_id", "map_name"], how="left")
    blocks = [
        (compute_player_skill_features(combined, rankings, _frame(lineups)), PLAYER_SKILL_FEATURES),
        (compute_roster_features(combined, _frame(lineups), _frame(roster_moves)), ROSTER_FEATURES),
        (compute_schedule_features(combined), SCHEDULE_FEATURES),
        (compute_event_features(combined), EVENT_FEATURES),
        (compute_veto_features(combined, _frame(veto)), VETO_FEATURES),
        (compute_page_h2h_features(combined, _frame(h2h)), PAGE_H2H_FEATURES),
    ]
    for block, declared in blocks:
        out = _merge_block(out, block, declared)
    return out


def compute_live_priors(
    team_a_name: str,
    team_b_name: str,
    map_name: str,
    team_a_picked_map: bool | None,
    is_lan: bool,
    match_history: pd.DataFrame,
    team_rankings: pd.DataFrame,
    as_of_date: datetime.date | None = None,
    is_decider: bool | None = None,
    bo_type: int | None = None,
    map_position_in_series: int | None = None,
    event_tier: int | None = None,
    team_a_series_score: int = 0,
    team_b_series_score: int = 0,
) -> dict:
    """
    Compute priors for a live/upcoming match using the most recent available data.
    as_of_date defaults to today. Used by the live strategy at startup.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today()

    history_sorted = match_history.sort_values("match_date").reset_index(drop=True)
    rankings = team_rankings.sort_values("date").reset_index(drop=True)

    future_date = pd.Timestamp(as_of_date + datetime.timedelta(days=1))

    id_a = _resolve_team_id(history_sorted, team_a_name)
    id_b = _resolve_team_id(history_sorted, team_b_name)
    pts_a, rank_a, _ = _ranking_lookup(rankings, id_a, future_date)
    pts_b, rank_b, _ = _ranking_lookup(rankings, id_b, future_date)

    _picked = float("nan") if team_a_picked_map is None else (1.0 if team_a_picked_map else 0.0)

    return {
        "team_a_avg_rating": pts_a,
        "team_b_avg_rating": pts_b,
        "team_a_rating_diff": pts_a - pts_b,
        "team_a_rank": rank_a,
        "team_b_rank": rank_b,
        "rank_diff": rank_a - rank_b,
        "team_a_map_winrate_long": _team_map_winrate(history_sorted, id_a, map_name, future_date, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
        "team_b_map_winrate_long": _team_map_winrate(history_sorted, id_b, map_name, future_date, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
        "team_a_map_winrate_short": _team_map_winrate(history_sorted, id_a, map_name, future_date, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
        "team_b_map_winrate_short": _team_map_winrate(history_sorted, id_b, map_name, future_date, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
        "team_a_overall_winrate": _team_overall_winrate(history_sorted, id_a, future_date),
        "team_b_overall_winrate": _team_overall_winrate(history_sorted, id_b, future_date),
        "h2h_win_rate": _h2h_winrate(history_sorted, id_a, id_b, future_date),
        "team_a_recent_form": _team_recent_form(history_sorted, id_a, future_date),
        "team_b_recent_form": _team_recent_form(history_sorted, id_b, future_date),
        "team_a_picked_map": _picked,
        "is_lan": int(is_lan),
        "is_decider": float(is_decider) if is_decider is not None else float("nan"),
        "bo_type": float(bo_type) if bo_type is not None else float("nan"),
        "map_position_in_series": float(map_position_in_series) if map_position_in_series is not None else float("nan"),
        "event_tier": float(event_tier) if event_tier is not None else float("nan"),
        "team_a_series_score": float(team_a_series_score),
        "team_b_series_score": float(team_b_series_score),
    }


def compute_live_priors_multi_map(
    team_a_name: str,
    team_b_name: str,
    map_names: list[str],
    map_pickers: dict[str, str | None],
    is_lan: bool,
    match_history: pd.DataFrame,
    team_rankings: pd.DataFrame,
    as_of_date: datetime.date | None = None,
    bo_type: int | None = None,
    event_tier: int | None = None,
    map_series_scores: dict[str, tuple[int, int]] | None = None,
    map_decider: dict[str, bool] | None = None,
) -> dict[str, dict]:
    """
    Compute priors for multiple maps at once. Returns {map_name: priors_dict}.
    Computes team-level stats (rankings, form, overall winrate, h2h) once and reuses them
    across all maps; only map-specific winrates require per-map computation.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today()

    history_sorted = match_history.sort_values("match_date").reset_index(drop=True)
    rankings = team_rankings.sort_values("date").reset_index(drop=True)

    future_date = pd.Timestamp(as_of_date + datetime.timedelta(days=1))

    id_a = _resolve_team_id(history_sorted, team_a_name)
    id_b = _resolve_team_id(history_sorted, team_b_name)
    pts_a, rank_a, _ = _ranking_lookup(rankings, id_a, future_date)
    pts_b, rank_b, _ = _ranking_lookup(rankings, id_b, future_date)
    form_a = _team_recent_form(history_sorted, id_a, future_date)
    form_b = _team_recent_form(history_sorted, id_b, future_date)
    overall_a = _team_overall_winrate(history_sorted, id_a, future_date)
    overall_b = _team_overall_winrate(history_sorted, id_b, future_date)
    h2h = _h2h_winrate(history_sorted, id_a, id_b, future_date)

    result = {}
    for idx, map_name in enumerate(map_names):
        picker = map_pickers.get(map_name)
        picked = float("nan") if picker is None else (1.0 if picker == team_a_name else 0.0)
        series_a, series_b = (map_series_scores or {}).get(map_name, (0, 0))
        decider = (map_decider or {}).get(map_name)

        result[map_name] = {
            "team_a_avg_rating": pts_a,
            "team_b_avg_rating": pts_b,
            "team_a_rating_diff": pts_a - pts_b,
            "team_a_rank": rank_a,
            "team_b_rank": rank_b,
            "rank_diff": rank_a - rank_b,
            "team_a_map_winrate_long": _team_map_winrate(history_sorted, id_a, map_name, future_date, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
            "team_b_map_winrate_long": _team_map_winrate(history_sorted, id_b, map_name, future_date, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
            "team_a_map_winrate_short": _team_map_winrate(history_sorted, id_a, map_name, future_date, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
            "team_b_map_winrate_short": _team_map_winrate(history_sorted, id_b, map_name, future_date, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
            "team_a_overall_winrate": overall_a,
            "team_b_overall_winrate": overall_b,
            "h2h_win_rate": h2h,
            "team_a_recent_form": form_a,
            "team_b_recent_form": form_b,
            "team_a_picked_map": picked,
            "is_lan": int(is_lan),
            "is_decider": float(decider) if decider is not None else float("nan"),
            "bo_type": float(bo_type) if bo_type is not None else float("nan"),
            "map_position_in_series": float(idx + 1),
            "event_tier": float(event_tier) if event_tier is not None else float("nan"),
            "team_a_series_score": float(series_a),
            "team_b_series_score": float(series_b),
        }

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-history", required=True)
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    history = pd.read_parquet(args.match_history)
    rankings_df = pd.read_parquet(args.rankings)
    priors_df = build_priors(history, rankings_df)
    priors_df.to_parquet(args.out, index=False)
    logger.info("Saved %d rows to %s", len(priors_df), args.out)
