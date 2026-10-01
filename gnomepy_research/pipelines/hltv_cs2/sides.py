"""
CS2 side bookkeeping: which team is on CT in a given round, and how to turn
half scores into side-attributed round tallies.

Two consumers:
  - repairing cs2_round_features, whose ct_score/t_score are side-keyed and
    therefore meaningless after the round-12 swap
  - the map DP, which needs p_CT / p_T and the swap schedule

CS2 is MR12: first to 13, sides swap after round 12, overtime is MR3 (first to 4
within a period, 3 rounds per side, new period on 3-3).

Overtime swap parity was determined empirically against 91 OT maps with known
HLTV final scores: teams predominantly do NOT swap at the start of overtime
(offset 0) but a minority of maps do (offset 1). The two are never both
consistent with the final score, so the offset is identifiable per map rather
than assumed — see resolve_ot_offset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGULATION_HALF_LENGTH = 12
REGULATION_ROUNDS = 24
ROUNDS_TO_WIN = 13
OT_HALF_LENGTH = 3
OT_ROUNDS_TO_WIN = 4
OT_PERIOD_LENGTH = 6


def side_swaps_before_round(round_number, ot_offset: int = 0):
    """Number of side swaps that have occurred before `round_number` (1-indexed)."""
    r = np.asarray(round_number)
    ot_swaps = 1 + ot_offset + (r - REGULATION_ROUNDS - 1) // OT_HALF_LENGTH
    out = np.where(
        r <= REGULATION_HALF_LENGTH, 0,
        np.where(r <= REGULATION_ROUNDS, 1, ot_swaps),
    )
    return out if out.ndim else out.item()


def team_a_is_ct(round_number, team_a_started_ct, ot_offset: int = 0):
    """Whether team_a is on CT for `round_number`."""
    swaps = np.asarray(side_swaps_before_round(round_number, ot_offset))
    started = np.asarray(team_a_started_ct).astype(bool)
    out = started ^ (swaps % 2 == 1)
    return out if out.ndim else out.item()


def team_a_won_round(ct_won, round_number, team_a_started_ct, ot_offset: int = 0):
    """Convert a side-keyed round result into a team_a-keyed one."""
    is_ct = np.asarray(team_a_is_ct(round_number, team_a_started_ct, ot_offset))
    cw = np.asarray(ct_won)
    return np.where(is_ct, cw, 1 - cw)


def resolve_ot_offset(
    round_numbers,
    ct_won,
    team_a_started_ct: bool,
    team_a_final_score: int,
) -> int | None:
    """
    Pick the overtime swap parity consistent with the known final score.

    Returns 0 or 1, or None when neither reconciles (the map must be dropped).
    Returns 0 for regulation maps, where the offset is unused.
    """
    if np.max(round_numbers) <= REGULATION_ROUNDS:
        candidates = [0]
    else:
        candidates = [0, 1]

    consistent = [
        off for off in candidates
        if team_a_won_round(ct_won, round_numbers, team_a_started_ct, off).sum() == team_a_final_score
    ]
    if len(consistent) == 1:
        return consistent[0]
    return None


@dataclass(frozen=True)
class SideSplit:
    """Regulation rounds won by each team on each side, from half scores alone."""
    team_a_ct_won: int
    team_a_ct_played: int
    team_a_t_won: int
    team_a_t_played: int
    valid: bool


def regulation_side_splits(
    team_a_score: int,
    team_b_score: int,
    team_a_h1: int,
    team_b_h1: int,
    team_a_started_ct: bool,
) -> SideSplit:
    """
    Recover side-attributed regulation rounds for one map.

    A map cannot be won inside half 1 (13 > 12), so h1 always totals 12. When the
    map went to overtime, regulation necessarily ended 12-12, so half 2 is the
    complement of half 1 rather than the difference from the final score.
    """
    invalid = SideSplit(0, 0, 0, 0, False)
    if team_a_h1 + team_b_h1 != REGULATION_HALF_LENGTH:
        return invalid

    went_ot = team_a_score + team_b_score > REGULATION_ROUNDS
    h2_a = REGULATION_HALF_LENGTH - team_a_h1 if went_ot else team_a_score - team_a_h1
    if not 0 <= h2_a <= REGULATION_HALF_LENGTH:
        return invalid

    if team_a_started_ct:
        ct_won, t_won = team_a_h1, h2_a
    else:
        ct_won, t_won = h2_a, team_a_h1

    return SideSplit(
        team_a_ct_won=int(ct_won),
        team_a_ct_played=REGULATION_HALF_LENGTH,
        team_a_t_won=int(t_won),
        team_a_t_played=REGULATION_HALF_LENGTH,
        valid=True,
    )


def build_side_round_dataset(history: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (map, side) with (successes, trials) for a binomial round-rate fit.

    Yields four side-attributed tallies per map — both teams, both sides — from
    half scores alone, so it covers every map rather than the demo subset.
    """
    required = ["team_a_score", "team_b_score", "team_a_h1_score", "team_b_h1_score", "team_a_started_ct"]
    usable = history.dropna(subset=required)
    if len(usable) < len(history):
        logger.info("side rounds: skipping %d map(s) missing half scores", len(history) - len(usable))

    rows = []
    for row in usable.itertuples():
        split = regulation_side_splits(
            int(row.team_a_score), int(row.team_b_score),
            int(row.team_a_h1_score), int(row.team_b_h1_score),
            bool(row.team_a_started_ct),
        )
        if not split.valid:
            continue
        base = {"match_id": row.match_id, "map_name": row.map_name, "match_date": row.match_date}
        rows.append({**base, "team_is_a": True, "side": 1,
                     "successes": split.team_a_ct_won, "trials": split.team_a_ct_played})
        rows.append({**base, "team_is_a": True, "side": -1,
                     "successes": split.team_a_t_won, "trials": split.team_a_t_played})
        rows.append({**base, "team_is_a": False, "side": 1,
                     "successes": split.team_a_t_played - split.team_a_t_won,
                     "trials": split.team_a_t_played})
        rows.append({**base, "team_is_a": False, "side": -1,
                     "successes": split.team_a_ct_played - split.team_a_ct_won,
                     "trials": split.team_a_ct_played})
    return pd.DataFrame(rows)


def attach_team_keyed_scores(
    round_df: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add team_a-keyed columns to round snapshots and drop maps that cannot be reconciled.

    The demo parser accumulates ct_score/t_score against the *current* side, so
    after the round-12 swap each column mixes rounds won by both teams. Round
    outcomes (ct_won) are sound, so team-keyed running scores are recoverable by
    replaying them through the swap schedule.

    Returns (repaired_rows, dropped_report). A map is dropped when its replayed
    final score disagrees with HLTV under either overtime parity — most often a
    demo covering only part of a map.
    """
    meta_cols = ["match_id", "map_name", "team_a_started_ct", "team_a_score", "team_b_score"]
    df = round_df.merge(history[meta_cols], on=["match_id", "map_name"], how="inner")
    df = df.dropna(subset=["team_a_started_ct"])

    kept, dropped = [], []
    for (match_id, map_name), g in df.groupby(["match_id", "map_name"], sort=False):
        g = g.sort_values(["round_number", "bomb_planted"]).copy()
        per_round = g.drop_duplicates("round_number", keep="last")
        started = bool(per_round.team_a_started_ct.iloc[0] == 1)
        final_a = int(per_round.team_a_score.iloc[0])
        final_b = int(per_round.team_b_score.iloc[0])

        reason = None
        if per_round.round_number.nunique() != final_a + final_b:
            reason = "incomplete_demo"
            offset = None
        else:
            offset = resolve_ot_offset(
                per_round.round_number.values, per_round.ct_won.values, started, final_a,
            )
            if offset is None:
                reason = "no_consistent_ot_parity"

        if reason is not None:
            dropped.append({"match_id": match_id, "map_name": map_name, "reason": reason,
                            "rounds": int(per_round.round_number.nunique()),
                            "hltv_rounds": final_a + final_b})
            continue

        g["ot_offset"] = offset
        g["team_a_is_ct"] = team_a_is_ct(g.round_number.values, started, offset)
        g["team_a_won_round"] = team_a_won_round(g.ct_won.values, g.round_number.values, started, offset)

        won = per_round.assign(
            w=team_a_won_round(per_round.ct_won.values, per_round.round_number.values, started, offset)
        ).set_index("round_number").w
        cum_a = won.cumsum().shift(1).fillna(0).astype(int)
        g["team_a_score_before"] = g.round_number.map(cum_a).astype(int)
        g["team_b_score_before"] = (g.round_number - 1 - g.team_a_score_before).astype(int)
        kept.append(g)

    repaired = pd.concat(kept, ignore_index=True) if kept else round_df.iloc[0:0].copy()
    report = pd.DataFrame(dropped, columns=["match_id", "map_name", "reason", "rounds", "hltv_rounds"])
    logger.info(
        "attach_team_keyed_scores: kept %d maps (%d rows), dropped %d",
        repaired.groupby(["match_id", "map_name"]).ngroups if len(repaired) else 0,
        len(repaired), len(report),
    )
    return repaired, report
