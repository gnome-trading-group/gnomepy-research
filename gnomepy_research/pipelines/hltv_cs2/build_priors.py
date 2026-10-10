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
import datetime
import logging

import numpy as np
import pandas as pd

from gnomepy_research.pipelines.hltv_cs2.context_features import (
    EVENT_FEATURES,
    PAGE_H2H_FEATURES,
    RANK_FEATURES,
    SCHEDULE_FEATURES,
    VETO_FEATURES,
    compute_event_features,
    compute_page_h2h_features,
    compute_rank_features,
    compute_schedule_features,
    compute_veto_features,
)
from gnomepy_research.pipelines.hltv_cs2.elo import compute_elo_features
from gnomepy_research.pipelines.hltv_cs2.player_form import (
    PLAYER_FORM_FEATURES,
    compute_player_form_features,
)

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


def compute_priors_for_row(
    row: pd.Series,
    history_sorted: pd.DataFrame,
    rankings: pd.DataFrame,
) -> dict | None:
    """Compute all prior features for a single map row, using only data strictly before its date."""
    if pd.isna(row["match_date"]):
        return None
    match_date = pd.Timestamp(row["match_date"])
    map_name = row["map_name"]
    id_a, id_b = int(row["team_a_id"]), int(row["team_b_id"])

    pts_a, rank_a, age_a = _ranking_lookup(rankings, id_a, match_date)
    pts_b, rank_b, age_b = _ranking_lookup(rankings, id_b, match_date)
    form_a = _team_recent_form(history_sorted, id_a, match_date)
    form_b = _team_recent_form(history_sorted, id_b, match_date)

    return {
        "match_id": row["match_id"],
        "map_name": map_name,
        "match_date": match_date,
        "team_a_name": row["team_a_name"],
        "team_b_name": row["team_b_name"],
        "team_a_id": id_a,
        "team_b_id": id_b,
        "team_a_rating_diff": pts_a - pts_b,
        "team_a_rank": rank_a,
        "team_b_rank": rank_b,
        "rank_diff": rank_a - rank_b,
        "ranking_age_days": np.nanmax([age_a, age_b]) if not (np.isnan(age_a) and np.isnan(age_b)) else float("nan"),
        "team_a_map_winrate_long": _team_map_winrate(history_sorted, id_a, map_name, match_date, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
        "team_b_map_winrate_long": _team_map_winrate(history_sorted, id_b, map_name, match_date, _MAP_WIN_RATE_WINDOW_LONG, _MAP_WIN_RATE_MAX_AGE_LONG),
        "team_a_map_winrate_short": _team_map_winrate(history_sorted, id_a, map_name, match_date, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
        "team_b_map_winrate_short": _team_map_winrate(history_sorted, id_b, map_name, match_date, _MAP_WIN_RATE_WINDOW_SHORT, _MAP_WIN_RATE_MAX_AGE_SHORT),
        "team_a_overall_winrate": _team_overall_winrate(history_sorted, id_a, match_date),
        "team_b_overall_winrate": _team_overall_winrate(history_sorted, id_b, match_date),
        "h2h_win_rate": _h2h_winrate(history_sorted, id_a, id_b, match_date),
        "team_a_recent_form": form_a,
        "team_b_recent_form": form_b,
        "recent_form_diff": form_a - form_b,
        "team_a_picked_map": float(row.get("team_a_picked_map", float("nan"))),
        "is_lan": float(row.get("is_lan", float("nan"))),
        "is_decider": float(row.get("is_decider", float("nan"))),
        "bo_type": float(row.get("bo_type", float("nan"))),
        "map_position_in_series": float(row.get("map_position_in_series", float("nan"))),
        "event_tier": float(row.get("event_tier", float("nan"))),
        "team_a_series_score": float(row.get("team_a_series_score", 0)),
        "team_b_series_score": float(row.get("team_b_series_score", 0)),
    }


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
    player_stats: pd.DataFrame | None = None,
    veto: pd.DataFrame | None = None,
    h2h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute point-in-time priors for every row in match_history.
    Returns a DataFrame with one row per map, keyed by match_id + map_name.
    """
    history_sorted = match_history.sort_values("match_date").reset_index(drop=True)
    rankings = team_rankings.sort_values("date").reset_index(drop=True)

    logger.info("Computing priors for %d map rows...", len(history_sorted))
    rows = []
    for i, row in history_sorted.iterrows():
        result = compute_priors_for_row(row, history_sorted, rankings)
        if result is not None:
            rows.append(result)
        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d rows", i + 1, len(history_sorted))

    priors = pd.DataFrame(rows)
    elo = compute_elo_features(history_sorted, rankings)
    out = priors.merge(elo, on=["match_id", "map_name"], how="left")

    blocks = [
        (compute_player_form_features(history_sorted, _frame(player_stats)), PLAYER_FORM_FEATURES),
        (compute_schedule_features(history_sorted), SCHEDULE_FEATURES),
        (compute_rank_features(history_sorted), RANK_FEATURES),
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
    player_stats: pd.DataFrame | None = None,
    veto: pd.DataFrame | None = None,
    h2h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Priors for the target rows only - upcoming maps whose outcome is unknown -
    computed exactly as build_priors would compute them with the targets appended
    to history.

    The point-in-time scans are linear and run over everything; the legacy per-row
    priors are quadratic, so they run for the targets alone (seconds, not the
    minutes a full rebuild takes every 30 min). Targets are only ever read by the
    scans before their own day is observed, so their missing outcomes never leak.
    """
    combined = pd.concat([match_history, targets], ignore_index=True)
    combined["match_date"] = pd.to_datetime(combined["match_date"])
    combined = combined.sort_values("match_date").reset_index(drop=True)
    rankings = team_rankings.sort_values("date").reset_index(drop=True)
    keys = set(zip(targets.match_id, targets.map_name))
    is_target = [k in keys for k in zip(combined.match_id, combined.map_name)]

    rows = [compute_priors_for_row(row, combined, rankings) for _, row in combined[is_target].iterrows()]
    out = pd.DataFrame([r for r in rows if r is not None])
    if out.empty:
        return out
    out = out.merge(compute_elo_features(combined, rankings), on=["match_id", "map_name"], how="left")
    blocks = [
        (compute_player_form_features(combined, _frame(player_stats)), PLAYER_FORM_FEATURES),
        (compute_schedule_features(combined), SCHEDULE_FEATURES),
        (compute_rank_features(combined), RANK_FEATURES),
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
