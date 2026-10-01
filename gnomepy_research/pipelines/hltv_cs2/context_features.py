"""
Schedule, head-to-head, veto and event-context features.

A deliberate split runs through this module: features derived by walking our own
history, and features read off the match page.

Own-history features (rest days, recent workload, head-to-head counts) are
causal by construction - we only ever count matches we have already seen at an
earlier date. They are bounded by the scrape window, so head-to-head counts run
short for pairs that met before it opens.

Page features (HLTV's own head-to-head and recent-results boxes) reach further
back, but only if those boxes render as of the match rather than as of the
fetch. That is an empirical question about HLTV, not an assumption we get to
make: `audit_page_box_pit` answers it by looking for referenced matches dated
after the match itself. Until it passes, prefer the own-history versions.

Veto and event context carry no such risk. Both are settled before the first
round and are read straight off the row.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnomepy_research.pipelines.hltv_cs2.pit import SeriesBatch, pit_scan

logger = logging.getLogger(__name__)

VETO_AUX = "veto"

# HLTV's veto vocabulary. A ban is "removed"; "left_over" is the decider. Getting
# this wrong is silent: filtering on a name that never matches yields a constant
# feature that still reports as fully dense.
_BAN_ACTION = "removed"
_VETO_ACTIONS = frozenset({"removed", "picked", "left_over"})
H2H_AUX = "h2h"
FORM_AUX = "recent_form"

_WORKLOAD_WINDOW_DAYS = 14


@dataclass
class _TeamSchedule:
    last_date: pd.Timestamp | None = None
    dates: list[pd.Timestamp] = field(default_factory=list)


def _rest_days(sched: _TeamSchedule, as_of: pd.Timestamp) -> float:
    if sched.last_date is None:
        return np.nan
    return float((as_of - sched.last_date).days)


class ScheduleAccumulator:
    """Rest and recent workload per team, and prior meetings per pair."""

    def __init__(self, window_days: int = _WORKLOAD_WINDOW_DAYS):
        self.window = pd.Timedelta(days=window_days)
        self.teams: dict[int, _TeamSchedule] = {}
        self.pairs: dict[tuple[int, int], list[int]] = {}

    def _workload(self, team_id, as_of: pd.Timestamp) -> tuple[float, float]:
        sched = self.teams.get(team_id)
        if sched is None:
            return np.nan, np.nan
        recent = [d for d in sched.dates if as_of - d <= self.window]
        return float(len(recent)), _rest_days(sched, as_of)

    def snapshot(self, batch: SeriesBatch) -> dict:
        row = next(batch.rows.itertuples())
        a_id, b_id = getattr(row, "team_a_id", None), getattr(row, "team_b_id", None)
        a_maps, a_rest = self._workload(a_id, batch.match_date)
        b_maps, b_rest = self._workload(b_id, batch.match_date)

        key = (a_id, b_id) if (a_id or 0) <= (b_id or 0) else (b_id, a_id)
        prior = self.pairs.get(key, [])
        a_wins = sum(1 for w in prior if w == a_id)
        total = len(prior)

        return {
            "team_a_rest_days": a_rest,
            "team_b_rest_days": b_rest,
            "rest_days_diff": a_rest - b_rest,
            "team_a_maps_14d": a_maps,
            "team_b_maps_14d": b_maps,
            "maps_14d_diff": a_maps - b_maps,
            "own_h2h_maps": float(total),
            "own_h2h_rate": (a_wins / total) if total else np.nan,
            "own_h2h_never_met": float(total == 0),
        }

    def update(self, batch: SeriesBatch) -> None:
        row = next(batch.rows.itertuples())
        a_id, b_id = getattr(row, "team_a_id", None), getattr(row, "team_b_id", None)
        for team_id in (a_id, b_id):
            if team_id is None:
                continue
            sched = self.teams.setdefault(team_id, _TeamSchedule())
            sched.last_date = batch.match_date
            sched.dates.extend([batch.match_date] * len(batch.rows))

        if a_id is None or b_id is None:
            return
        key = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        bucket = self.pairs.setdefault(key, [])
        for r in batch.rows.itertuples():
            bucket.append(a_id if getattr(r, "team_a_won", 0) else b_id)


SCHEDULE_FEATURES = [
    "team_a_rest_days", "team_b_rest_days", "rest_days_diff",
    "team_a_maps_14d", "team_b_maps_14d", "maps_14d_diff",
    "own_h2h_maps", "own_h2h_rate", "own_h2h_never_met",
]


def compute_schedule_features(history: pd.DataFrame) -> pd.DataFrame:
    """Rest, workload and own-history head-to-head, keyed (match_id, map_name)."""
    return pit_scan(history, ScheduleAccumulator())


def _rank(value) -> float:
    f = pd.to_numeric(value, errors="coerce")
    return float(f) if f == f else np.nan


def compute_rank_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Combine HLTV and VRS ranks.

    rank_diff alone is NaN for 19.4% of rows, concentrated on lower-tier teams
    where the model is weakest. VRS covers a different population, so the union
    is denser than either, and the two disagreeing is itself informative.
    """
    out = pd.DataFrame({
        "match_id": history["match_id"],
        "map_name": history["map_name"],
    })
    has_vrs = "team_a_vrs_rank" in history.columns

    for side in ("a", "b"):
        hltv = pd.to_numeric(history.get(f"team_{side}_rank"), errors="coerce")
        vrs = pd.to_numeric(history.get(f"team_{side}_vrs_rank"), errors="coerce") if has_vrs else pd.Series(np.nan, index=history.index)
        out[f"team_{side}_rank_best"] = pd.concat([hltv, vrs], axis=1).min(axis=1)
        out[f"team_{side}_log_rank"] = np.log1p(out[f"team_{side}_rank_best"])

    out["rank_best_diff"] = out["team_a_rank_best"] - out["team_b_rank_best"]
    out["log_rank_diff"] = out["team_a_log_rank"] - out["team_b_log_rank"]
    out["rank_known_both"] = (out["team_a_rank_best"].notna() & out["team_b_rank_best"].notna()).astype(float)

    if has_vrs:
        a_gap = pd.to_numeric(history["team_a_rank"], errors="coerce") - pd.to_numeric(history["team_a_vrs_rank"], errors="coerce")
        b_gap = pd.to_numeric(history["team_b_rank"], errors="coerce") - pd.to_numeric(history["team_b_vrs_rank"], errors="coerce")
        out["rank_source_disagreement"] = a_gap - b_gap
    else:
        out["rank_source_disagreement"] = np.nan
    return out


RANK_FEATURES = [
    "team_a_rank_best", "team_b_rank_best", "team_a_log_rank", "team_b_log_rank",
    "rank_best_diff", "log_rank_diff", "rank_known_both", "rank_source_disagreement",
]


EVENT_FEATURES = ["is_elimination", "is_qualifier", "is_playoff", "is_group"]


def compute_event_features(history: pd.DataFrame) -> pd.DataFrame:
    """Stakes flags, straight off the row - all settled before the match."""
    out = pd.DataFrame({"match_id": history["match_id"], "map_name": history["map_name"]})
    for col in EVENT_FEATURES:
        out[col] = pd.to_numeric(history.get(col), errors="coerce").fillna(0.0) if col in history.columns else 0.0
    return out


VETO_FEATURES = [
    "team_a_picked_map_exact", "map_pick_index", "team_a_banned_first",
    "team_a_bans_before_pick", "team_b_bans_before_pick", "veto_known",
]


def compute_veto_features(history: pd.DataFrame, veto: pd.DataFrame) -> pd.DataFrame:
    """
    Veto-derived features, replacing the substring team-name match.

    The old team_a_picked_map matched team names by substring and was NaN for
    19.6% of rows. Joining on team_id through the veto rows is exact.
    """
    base = pd.DataFrame({
        "match_id": history["match_id"],
        "map_name": history["map_name"],
        "subject_team_a": pd.to_numeric(history.get("team_a_id"), errors="coerce"),
        "subject_team_b": pd.to_numeric(history.get("team_b_id"), errors="coerce"),
    })
    if veto is None or veto.empty:
        for col in VETO_FEATURES:
            base[col] = np.nan
        base["veto_known"] = 0.0
        return base.drop(columns=["subject_team_a", "subject_team_b"])

    v = veto.copy()
    v["order"] = pd.to_numeric(v["order"], errors="coerce")
    v["team_id"] = pd.to_numeric(v["team_id"], errors="coerce")
    unknown = set(v["action"].dropna().unique()) - _VETO_ACTIONS
    if unknown:
        logger.warning("unrecognised veto actions %s — ban features will be incomplete", sorted(unknown))
    picks = v[v["action"] == "picked"]
    bans = v[v["action"] == _BAN_ACTION]
    if picks.empty or bans.empty:
        logger.warning("veto frame has %d picks and %d bans — check the action vocabulary",
                       len(picks), len(bans))

    pick_by = picks.set_index(["match_id", "map_name"])["team_id"].to_dict()
    pick_order = picks.set_index(["match_id", "map_name"])["order"].to_dict()
    first_ban = bans.sort_values("order").groupby("match_id")["team_id"].first().to_dict()
    known = set(v["match_id"].unique())

    def _picked(row):
        team = pick_by.get((row.match_id, row.map_name))
        if team is None or row.subject_team_a is None:
            return np.nan
        return float(team == row.subject_team_a)

    base["team_a_picked_map_exact"] = [
        _picked(r) for r in base.itertuples()
    ]
    base["map_pick_index"] = [
        pick_order.get((r.match_id, r.map_name), np.nan) for r in base.itertuples()
    ]
    base["team_a_banned_first"] = [
        float(first_ban[r.match_id] == r.subject_team_a) if r.match_id in first_ban else np.nan
        for r in base.itertuples()
    ]
    bans_before = (
        bans.groupby(["match_id", "team_id"])["order"].min().rename("first_ban_order").reset_index()
    )
    first_orders = bans_before.set_index(["match_id", "team_id"])["first_ban_order"].to_dict()
    base["team_a_bans_before_pick"] = [
        first_orders.get((r.match_id, r.subject_team_a), np.nan) for r in base.itertuples()
    ]
    base["team_b_bans_before_pick"] = [
        first_orders.get((r.match_id, r.subject_team_b), np.nan) for r in base.itertuples()
    ]
    base["veto_known"] = base["match_id"].isin(known).astype(float)
    return base.drop(columns=["subject_team_a", "subject_team_b"])


def _naive(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", utc=True)
    return out.dt.tz_localize(None)


def audit_page_box_pit(
    history: pd.DataFrame,
    frame: pd.DataFrame,
    date_col: str,
    *,
    label: str,
) -> dict:
    """
    Does an HLTV page box render as of the match, or as of the fetch?

    If as of the fetch, it lists matches that had not happened yet when the page
    subject was played, and any feature built on it reads the future. Counts
    referenced rows dated after their own match.
    """
    dates = history.groupby("match_id")["match_date"].min()
    merged = frame.merge(dates.rename("subject_date"), left_on="match_id", right_index=True, how="inner")
    ref = _naive(merged[date_col])
    subject = _naive(merged["subject_date"])

    after = ref.dt.normalize() > subject.dt.normalize()
    same_day = ref.dt.normalize() == subject.dt.normalize()

    result = {
        "label": label,
        "rows": int(len(merged)),
        "after_rows": int(after.sum()),
        "same_day_rows": int(same_day.sum()),
        "after_share": float(after.mean()) if len(merged) else 0.0,
        "matches_affected": int(merged.loc[after.values, "match_id"].nunique()),
        "pit_safe": bool(after.sum() == 0),
    }
    if result["pit_safe"]:
        logger.info("%s PIT audit: clean — 0 rows dated after their match; %d same-day rows "
                    "(excluded by the strict < filter)", label, result["same_day_rows"])
    else:
        logger.warning("%s PIT audit: %d/%d rows dated AFTER their match (%.2f%%), %d matches affected — "
                       "the box renders as of fetch, not as of match",
                       label, result["after_rows"], result["rows"],
                       100 * result["after_share"], result["matches_affected"])
    return result


PAGE_H2H_FEATURES = [
    "page_h2h_maps", "page_h2h_rate", "page_h2h_recent_rate",
    "page_h2h_days_since", "page_h2h_never_met",
]


def compute_page_h2h_features(history: pd.DataFrame, h2h: pd.DataFrame) -> pd.DataFrame:
    """
    Head-to-head from HLTV's own box, which reaches back past the scrape window.

    Audited PIT-safe under a strict date filter: the box lists no meeting dated
    after its match, but it does list same-day meetings - separate series between
    the same teams earlier that day. Date granularity cannot order those against
    the subject, so `<` drops them rather than risk a sibling leak.
    """
    base = pd.DataFrame({
        "match_id": history["match_id"],
        "map_name": history["map_name"],
    })
    if h2h is None or h2h.empty:
        for col in PAGE_H2H_FEATURES:
            base[col] = np.nan
        base["page_h2h_maps"] = 0.0
        base["page_h2h_never_met"] = 1.0
        return base

    subject = history.groupby("match_id").agg(
        subject_date=("match_date", "min"),
        a_name=("team_a_name", "first"),
        b_name=("team_b_name", "first"),
    )
    m = h2h.merge(subject, left_on="match_id", right_index=True, how="inner")
    m = m[_naive(m["h2h_date"]).dt.normalize() < _naive(m["subject_date"]).dt.normalize()].copy()

    t1, t2 = m["team1_name"].astype(str), m["team2_name"].astype(str)
    a, b = m["a_name"].astype(str), m["b_name"].astype(str)
    a_is_t1, a_is_t2 = t1.eq(a), t2.eq(a)
    m = m[(a_is_t1 & t2.eq(b)) | (a_is_t2 & t1.eq(b))].copy()
    if m.empty:
        for col in PAGE_H2H_FEATURES:
            base[col] = np.nan
        base["page_h2h_maps"] = 0.0
        base["page_h2h_never_met"] = 1.0
        return base

    m["a_won"] = np.where(t1.loc[m.index].eq(a.loc[m.index]),
                          m["team1_won"].astype(float),
                          1.0 - m["team1_won"].astype(float))
    m["age_days"] = (_naive(m["subject_date"]) - _naive(m["h2h_date"])).dt.days

    agg = m.groupby("match_id").agg(
        page_h2h_maps=("a_won", "size"),
        page_h2h_rate=("a_won", "mean"),
        page_h2h_days_since=("age_days", "min"),
    )
    recent = (
        m.sort_values("age_days").groupby("match_id").head(5)
        .groupby("match_id")["a_won"].mean().rename("page_h2h_recent_rate")
    )
    agg = agg.join(recent)

    out = base.merge(agg, left_on="match_id", right_index=True, how="left")
    out["page_h2h_never_met"] = out["page_h2h_maps"].isna().astype(float)
    out["page_h2h_maps"] = out["page_h2h_maps"].fillna(0.0)
    return out
