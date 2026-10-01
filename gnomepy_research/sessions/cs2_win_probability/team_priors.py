from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import pandas as pd

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.pipelines.hltv_cs2.build_priors import (
    compute_live_priors,
    compute_live_priors_multi_map,
)

logger = logging.getLogger(__name__)


@dataclass
class TeamPriors:
    team_name: str
    avg_rating: float = float("nan")
    map_win_rates_long: dict[str, float] = field(default_factory=dict)
    map_win_rates_short: dict[str, float] = field(default_factory=dict)
    recent_form: float = float("nan")
    world_rank: float = float("nan")

    def map_win_rate_long(self, map_name: str) -> float:
        return self.map_win_rates_long.get(map_name, float("nan"))

    def map_win_rate_short(self, map_name: str) -> float:
        return self.map_win_rates_short.get(map_name, float("nan"))


@dataclass
class MatchPriors:
    team_a: TeamPriors
    team_b: TeamPriors
    map_name: str
    h2h_win_rate: float = float("nan")
    team_a_picked_map: float = float("nan")
    is_lan: bool = True
    team_a_series_score: int = 0
    team_b_series_score: int = 0

    def to_feature_dict(self) -> dict:
        return {
            "team_a_avg_rating": self.team_a.avg_rating,
            "team_b_avg_rating": self.team_b.avg_rating,
            "team_a_rating_diff": self.team_a.avg_rating - self.team_b.avg_rating,
            "team_a_rank": self.team_a.world_rank,
            "team_b_rank": self.team_b.world_rank,
            "rank_diff": self.team_a.world_rank - self.team_b.world_rank,
            "team_a_map_winrate_long": self.team_a.map_win_rate_long(self.map_name),
            "team_b_map_winrate_long": self.team_b.map_win_rate_long(self.map_name),
            "team_a_map_winrate_short": self.team_a.map_win_rate_short(self.map_name),
            "team_b_map_winrate_short": self.team_b.map_win_rate_short(self.map_name),
            "team_a_overall_winrate": self.team_a.map_win_rates_long.get("_overall", float("nan")),
            "team_b_overall_winrate": self.team_b.map_win_rates_long.get("_overall", float("nan")),
            "h2h_win_rate": self.h2h_win_rate,
            "team_a_recent_form": self.team_a.recent_form,
            "team_b_recent_form": self.team_b.recent_form,
            "team_a_picked_map": self.team_a_picked_map,
            "is_lan": int(self.is_lan),
            "map_name": self.map_name,
            "team_a_series_score": self.team_a_series_score,
            "team_b_series_score": self.team_b_series_score,
        }


@dataclass
class SeriesVetoInfo:
    map_names: list[str]
    map_pickers: dict[str, str | None]
    bo_type: int


def load_match_priors(
    team_a_name: str,
    team_b_name: str,
    map_name: str,
    team_a_picked_map: float = float("nan"),
    is_lan: bool = True,
    team_a_series_score: int = 0,
    team_b_series_score: int = 0,
) -> MatchPriors:
    """
    Load point-in-time priors for a live match from the latest datasets.
    Returns neutral defaults if datasets are unavailable.
    """
    try:
        ds = DatasetStore()
        match_history = ds.load("cs2_match_history")
        team_rankings = ds.load("cs2_team_rankings")
    except Exception as exc:
        logger.warning("Could not load priors datasets (%s) — using neutral defaults", exc)
        return _neutral_priors(team_a_name, team_b_name, map_name, team_a_picked_map, is_lan, team_a_series_score, team_b_series_score)

    priors_dict = compute_live_priors(
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        map_name=map_name,
        team_a_picked_map=bool(team_a_picked_map) if not _is_nan(team_a_picked_map) else None,
        is_lan=is_lan,
        match_history=match_history,
        team_rankings=team_rankings,
    )

    mp = _priors_from_dict(priors_dict, team_a_name, team_b_name, map_name, is_lan)
    mp.team_a_series_score = team_a_series_score
    mp.team_b_series_score = team_b_series_score
    return mp


def load_series_priors(
    team_a_name: str,
    team_b_name: str,
    veto: SeriesVetoInfo,
    is_lan: bool = True,
    event_tier: int | None = None,
    map_series_scores: dict[str, tuple[int, int]] | None = None,
) -> dict[str, dict]:
    """
    Load point-in-time priors for all maps in a series veto.
    Returns {map_name: priors_dict}. Uses compute_live_priors_multi_map for efficiency.

    map_series_scores: {map_name: (team_a_maps_won, team_b_maps_won)} before that map is played.
    Defaults to (0, 0) for all maps if not provided.
    """
    try:
        ds = DatasetStore()
        match_history = ds.load("cs2_match_history")
        team_rankings = ds.load("cs2_team_rankings")
    except Exception as exc:
        logger.warning("Could not load priors datasets (%s) — returning empty priors", exc)
        return {}

    map_decider = {
        m: (veto.map_pickers.get(m) is None)
        for m in veto.map_names
    }

    return compute_live_priors_multi_map(
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        map_names=veto.map_names,
        map_pickers=veto.map_pickers,
        is_lan=is_lan,
        match_history=match_history,
        team_rankings=team_rankings,
        bo_type=veto.bo_type,
        event_tier=event_tier,
        map_series_scores=map_series_scores,
        map_decider=map_decider,
    )


def _is_nan(v) -> bool:
    try:
        import math
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _priors_from_dict(priors_dict: dict, team_a_name: str, team_b_name: str, map_name: str, is_lan: bool) -> MatchPriors:
    team_a = TeamPriors(
        team_name=team_a_name,
        avg_rating=priors_dict.get("team_a_avg_rating", float("nan")),
        map_win_rates_long={
            map_name: priors_dict.get("team_a_map_winrate_long", float("nan")),
            "_overall": priors_dict.get("team_a_overall_winrate", float("nan")),
        },
        map_win_rates_short={map_name: priors_dict.get("team_a_map_winrate_short", float("nan"))},
        recent_form=priors_dict.get("team_a_recent_form", float("nan")),
        world_rank=priors_dict.get("team_a_rank", float("nan")),
    )
    team_b = TeamPriors(
        team_name=team_b_name,
        avg_rating=priors_dict.get("team_b_avg_rating", float("nan")),
        map_win_rates_long={
            map_name: priors_dict.get("team_b_map_winrate_long", float("nan")),
            "_overall": priors_dict.get("team_b_overall_winrate", float("nan")),
        },
        map_win_rates_short={map_name: priors_dict.get("team_b_map_winrate_short", float("nan"))},
        recent_form=priors_dict.get("team_b_recent_form", float("nan")),
        world_rank=priors_dict.get("team_b_rank", float("nan")),
    )
    return MatchPriors(
        team_a=team_a,
        team_b=team_b,
        map_name=map_name,
        h2h_win_rate=priors_dict.get("h2h_win_rate", float("nan")),
        team_a_picked_map=priors_dict.get("team_a_picked_map", float("nan")),
        is_lan=is_lan,
    )


def _neutral_priors(
    team_a_name: str,
    team_b_name: str,
    map_name: str,
    team_a_picked_map: float,
    is_lan: bool,
    team_a_series_score: int = 0,
    team_b_series_score: int = 0,
) -> MatchPriors:
    return MatchPriors(
        team_a=TeamPriors(team_name=team_a_name),
        team_b=TeamPriors(team_name=team_b_name),
        map_name=map_name,
        team_a_picked_map=team_a_picked_map,
        is_lan=is_lan,
        team_a_series_score=team_a_series_score,
        team_b_series_score=team_b_series_score,
    )
