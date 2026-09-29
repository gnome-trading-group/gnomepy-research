from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.pipelines.hltv_cs2.build_priors import compute_live_priors

logger = logging.getLogger(__name__)


@dataclass
class TeamPriors:
    team_name: str
    avg_rating: float = 1.0
    map_win_rates_long: dict[str, float] = field(default_factory=dict)
    map_win_rates_short: dict[str, float] = field(default_factory=dict)
    recent_form: float = 0.5
    world_rank: int = 999

    def map_win_rate_long(self, map_name: str) -> float:
        return self.map_win_rates_long.get(map_name, 0.5)

    def map_win_rate_short(self, map_name: str) -> float:
        return self.map_win_rates_short.get(map_name, 0.5)


@dataclass
class MatchPriors:
    team_a: TeamPriors
    team_b: TeamPriors
    map_name: str
    h2h_win_rate: float = 0.5
    team_a_picked_map: bool = False
    is_lan: bool = True

    def to_feature_dict(self) -> dict:
        return {
            "team_a_avg_rating": self.team_a.avg_rating,
            "team_b_avg_rating": self.team_b.avg_rating,
            "team_a_map_winrate_long": self.team_a.map_win_rate_long(self.map_name),
            "team_b_map_winrate_long": self.team_b.map_win_rate_long(self.map_name),
            "team_a_map_winrate_short": self.team_a.map_win_rate_short(self.map_name),
            "team_b_map_winrate_short": self.team_b.map_win_rate_short(self.map_name),
            "h2h_win_rate": self.h2h_win_rate,
            "team_a_recent_form": self.team_a.recent_form,
            "team_b_recent_form": self.team_b.recent_form,
            "team_a_picked_map": self.team_a_picked_map,
            "is_lan": int(self.is_lan),
        }


def load_match_priors(
    team_a_name: str,
    team_b_name: str,
    map_name: str,
    team_a_picked_map: bool = False,
    is_lan: bool = True,
) -> MatchPriors:
    """
    Load point-in-time priors for a live match from the latest cs2_match_history
    and cs2_team_rankings datasets. Used at strategy startup.

    Requires cs2_match_history and cs2_team_rankings to be published and up-to-date.
    Returns neutral defaults if datasets are unavailable.
    """
    try:
        ds = DatasetStore()
        match_history = ds.load("cs2_match_history")
        team_rankings = ds.load("cs2_team_rankings")
    except Exception as exc:
        logger.warning("Could not load priors datasets (%s) — using neutral defaults", exc)
        return _neutral_priors(team_a_name, team_b_name, map_name, team_a_picked_map, is_lan)

    priors_dict = compute_live_priors(
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        map_name=map_name,
        team_a_picked_map=team_a_picked_map,
        is_lan=is_lan,
        match_history=match_history,
        team_rankings=team_rankings,
    )

    team_a = TeamPriors(
        team_name=team_a_name,
        avg_rating=priors_dict["team_a_avg_rating"],
        map_win_rates_long={map_name: priors_dict["team_a_map_winrate_long"]},
        map_win_rates_short={map_name: priors_dict["team_a_map_winrate_short"]},
        recent_form=priors_dict["team_a_recent_form"],
    )
    team_b = TeamPriors(
        team_name=team_b_name,
        avg_rating=priors_dict["team_b_avg_rating"],
        map_win_rates_long={map_name: priors_dict["team_b_map_winrate_long"]},
        map_win_rates_short={map_name: priors_dict["team_b_map_winrate_short"]},
        recent_form=priors_dict["team_b_recent_form"],
    )

    return MatchPriors(
        team_a=team_a,
        team_b=team_b,
        map_name=map_name,
        h2h_win_rate=priors_dict["h2h_win_rate"],
        team_a_picked_map=team_a_picked_map,
        is_lan=is_lan,
    )


def _neutral_priors(
    team_a_name: str,
    team_b_name: str,
    map_name: str,
    team_a_picked_map: bool,
    is_lan: bool,
) -> MatchPriors:
    return MatchPriors(
        team_a=TeamPriors(team_name=team_a_name),
        team_b=TeamPriors(team_name=team_b_name),
        map_name=map_name,
        team_a_picked_map=team_a_picked_map,
        is_lan=is_lan,
    )
