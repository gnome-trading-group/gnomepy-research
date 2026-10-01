"""
Layer 4: P(team_a wins the series), composed from per-map probabilities.

The naive composition — one probability applied to every remaining map — assumes
map outcomes are independent. They are not: measured corr(map1, map2) = 0.188,
with P(win m2 | won m1) = 0.642 against 0.455 after losing it. Independent
composition therefore amplifies a map edge into a series edge that is too
extreme, and measurably loses to a direct series model.

This DP conditions each node on the series score that reaches it, which is the
form the Layer-1 model was trained on (team_a_series_score / team_b_series_score
are real features). Whether that recovers enough of the dependence is an
empirical question settled by the gate in evaluate_series, not an assumption.

The lattice is tiny — 4 non-terminal nodes for a BO3, 9 for a BO5 — so the cost
is one model inference per node, which is batched.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SeriesVetoTooShort(ValueError):
    """Raised when the veto cannot carry the series to a decision."""


def series_nodes(maps_to_win: int) -> list[tuple[int, int]]:
    """Reachable, non-terminal (a_wins, b_wins) states, in play order."""
    return [
        (a, b)
        for total in range(2 * maps_to_win - 1)
        for a in range(total + 1)
        for b in [total - a]
        if a < maps_to_win and b < maps_to_win
    ]


def series_win_prob(
    node_probs: dict[tuple[int, int], float],
    maps_to_win: int = 2,
    team_a_maps_won: int = 0,
    team_b_maps_won: int = 0,
    n_veto_maps: int | None = None,
) -> float:
    """
    P(team_a wins the series) from per-node map probabilities.

    `node_probs[(a, b)]` is P(team_a wins the map played at that score.)
    """
    if n_veto_maps is not None and n_veto_maps < 2 * maps_to_win - 1:
        raise SeriesVetoTooShort(
            f"veto lists {n_veto_maps} map(s) but a first-to-{maps_to_win} series can need "
            f"{2 * maps_to_win - 1}; the shortfall would silently accrue to team_b"
        )

    cache: dict[tuple[int, int], float] = {}

    def s(a: int, b: int) -> float:
        if a >= maps_to_win:
            return 1.0
        if b >= maps_to_win:
            return 0.0
        if (a, b) in cache:
            return cache[(a, b)]
        if (a, b) not in node_probs:
            raise SeriesVetoTooShort(f"no map probability supplied for series state {a}-{b}")
        p = node_probs[(a, b)]
        value = p * s(a + 1, b) + (1.0 - p) * s(a, b + 1)
        cache[(a, b)] = value
        return value

    return s(team_a_maps_won, team_b_maps_won)


# Features that describe a particular map rather than the matchup. An unknown
# map has none of them; everything else is frozen per series and carries over.
_MAP_SPECIFIC = (
    "team_a_map_winrate_long", "team_b_map_winrate_long",
    "team_a_map_winrate_short", "team_b_map_winrate_short",
    "elo_map_diff", "elo_map_resid_diff",
    "team_a_picked_map", "team_a_picked_map_exact", "map_pick_index",
)


def build_node_frame(
    series_priors: pd.DataFrame,
    maps_to_win: int,
    feature_names: list[str],
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """
    One feature row per lattice node for a single series, from historical priors.

    Positions 1..maps_to_win are always played, so they use their real rows.
    Later positions may not be played, and history only records the maps that
    were. Using a real row whenever one happens to exist would make the input
    depend on whether the series went the distance - which is the outcome. So
    those positions are always priced as an unknown map: series-level features
    carry over, map-specific ones are NaN and the map one-hot is empty.

    Production does not go through here: it prices the decider from the veto via
    map_model.series_win_prob_per_map, where the decider is genuinely known.
    """
    by_position = {int(r.map_position_in_series): r for r in series_priors.itertuples()}
    guaranteed = [p for p in sorted(by_position) if p <= maps_to_win]
    if not guaranteed:
        raise ValueError(f"series has no played map within the first {maps_to_win} positions")
    carrier = by_position[guaranteed[-1]]
    decider = 2 * maps_to_win - 1

    rows, nodes = [], []
    for a, b in series_nodes(maps_to_win):
        position = a + b + 1
        known = position <= maps_to_win and position in by_position
        source = by_position[position] if known else carrier
        row = {c: getattr(source, c, np.nan) for c in series_priors.columns}
        row["team_a_series_score"] = float(a)
        row["team_b_series_score"] = float(b)
        row["map_position_in_series"] = float(position)
        if not known:
            row["is_decider"] = float(position == decider)
            row["map_name"] = ""
            for c in _MAP_SPECIFIC:
                row[c] = np.nan
        rows.append(row)
        nodes.append((a, b))
    return pd.DataFrame(rows), nodes
