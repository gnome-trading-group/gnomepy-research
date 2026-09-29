from __future__ import annotations

import numpy as np

from gnomepy_research.pipelines.hltv_cs2.config import MAP_POOL
from gnomepy_research.sessions.cs2_win_probability.game_state import BombState, CS2GameState

FEATURE_NAMES = [
    "equip_value_diff",
    "equip_ratio",
    "ct_equip_value",
    "t_equip_value",
    "alive_equip_diff",
    "ct_alive_equip",
    "t_alive_equip",
    "round_start_equip_diff",
    "ct_round_start_equip",
    "t_round_start_equip",
    "money_diff",
    "money_ratio",
    "players_alive_diff",
    "ct_players_alive",
    "t_players_alive",
    "hp_diff",
    "ct_total_hp",
    "t_total_hp",
    "armor_diff",
    "ct_armor",
    "t_armor",
    "armored_diff",
    "ct_armored",
    "t_armored",
    "bomb_planted",
    "bomb_time_remaining",
    "ct_consecutive_losses",
    "t_consecutive_losses",
    "round_number",
    "score_diff",
    "ct_score",
    "t_score",
    "current_half",
    "ct_win_rate_last_5",
    "team_a_rating_diff",
    "team_a_map_winrate_long",
    "team_b_map_winrate_long",
    "team_a_map_winrate_short",
    "team_b_map_winrate_short",
    "h2h_win_rate",
    "recent_form_diff",
    "team_a_picked_map",
    "is_lan",
] + [f"map_{m}" for m in MAP_POOL]


def _map_ohe(map_name: str) -> list[float]:
    return [1.0 if m == map_name else 0.0 for m in MAP_POOL]


def extract_features(state: CS2GameState, priors: dict | None = None) -> np.ndarray:
    """
    Extract feature vector from live game state + static team priors.
    Returns a 1D array aligned with FEATURE_NAMES.

    priors dict (optional):
        team_a_avg_rating: float   — HLTV Rating 3.0 average for team A
        team_b_avg_rating: float   — HLTV Rating 3.0 average for team B
        team_a_map_winrate_long: float
        team_b_map_winrate_long: float
        team_a_map_winrate_short: float
        team_b_map_winrate_short: float
        h2h_win_rate: float
        team_a_recent_form: float
        team_b_recent_form: float
        team_a_picked_map: bool
        is_lan: int
    """
    equip_diff = state.ct_equipment_value - state.t_equipment_value
    equip_sum = max(state.ct_equipment_value + state.t_equipment_value, 1)
    equip_ratio = state.ct_equipment_value / equip_sum

    ct_alive_equip = sum(p.equipment_value for p in state.players if p.team == "CT" and p.is_alive)
    t_alive_equip = sum(p.equipment_value for p in state.players if p.team == "T" and p.is_alive)

    ct_armor = sum(p.armor for p in state.players if p.team == "CT")
    t_armor = sum(p.armor for p in state.players if p.team == "T")
    ct_armored = sum(1 for p in state.players if p.team == "CT" and p.is_alive and p.armor > 0)
    t_armored = sum(1 for p in state.players if p.team == "T" and p.is_alive and p.armor > 0)

    money_diff = state.ct_team_money - state.t_team_money
    money_sum = max(state.ct_team_money + state.t_team_money, 1)
    money_ratio = state.ct_team_money / money_sum

    bomb_planted = 1 if state.bomb_state == BombState.PLANTED else 0
    score_diff = state.ct_score - state.t_score

    _nan = float("nan")
    p = priors or {}
    team_a_rating = p.get("team_a_avg_rating", _nan)
    team_b_rating = p.get("team_b_avg_rating", _nan)
    rating_diff = team_a_rating - team_b_rating
    team_a_map_wr_long = p.get("team_a_map_winrate_long", _nan)
    team_b_map_wr_long = p.get("team_b_map_winrate_long", _nan)
    team_a_map_wr_short = p.get("team_a_map_winrate_short", _nan)
    team_b_map_wr_short = p.get("team_b_map_winrate_short", _nan)
    h2h = p.get("h2h_win_rate", _nan)
    form_a = p.get("team_a_recent_form", _nan)
    form_b = p.get("team_b_recent_form", _nan)
    form_diff = form_a - form_b
    _picked = p.get("team_a_picked_map", _nan)
    picked = _nan if isinstance(_picked, float) and np.isnan(_picked) else (1.0 if _picked else 0.0)
    _is_lan = p.get("is_lan", _nan)
    is_lan = _nan if isinstance(_is_lan, float) and np.isnan(_is_lan) else float(_is_lan)

    return np.array([
        equip_diff,
        equip_ratio,
        state.ct_equipment_value,
        state.t_equipment_value,
        ct_alive_equip - t_alive_equip,
        ct_alive_equip,
        t_alive_equip,
        state.ct_round_start_equip - state.t_round_start_equip,
        state.ct_round_start_equip,
        state.t_round_start_equip,
        money_diff,
        money_ratio,
        state.ct_players_alive - state.t_players_alive,
        state.ct_players_alive,
        state.t_players_alive,
        state.ct_total_hp - state.t_total_hp,
        state.ct_total_hp,
        state.t_total_hp,
        ct_armor - t_armor,
        ct_armor,
        t_armor,
        ct_armored - t_armored,
        ct_armored,
        t_armored,
        bomb_planted,
        state.bomb_time_remaining,
        state.ct_consecutive_losses,
        state.t_consecutive_losses,
        state.round_number,
        score_diff,
        state.ct_score,
        state.t_score,
        state.current_half,
        state.ct_win_rate_last_5,
        rating_diff,
        team_a_map_wr_long,
        team_b_map_wr_long,
        team_a_map_wr_short,
        team_b_map_wr_short,
        h2h,
        form_diff,
        picked,
        is_lan,
        *_map_ohe(state.map_name),
    ], dtype=np.float32)


def extract_features_from_demo_row(row: dict, priors: dict | None = None) -> np.ndarray:
    """
    Extract features from a parsed demo snapshot row (demoparser2 output).
    Used during model training — mirrors extract_features() structure.
    """
    ct_equip = float(row.get("ct_equip_value", 0))
    t_equip = float(row.get("t_equip_value", 0))
    equip_sum = max(ct_equip + t_equip, 1)

    ct_alive_equip = float(row.get("ct_alive_equip", 0))
    t_alive_equip = float(row.get("t_alive_equip", 0))

    ct_round_start_equip = float(row.get("ct_round_start_equip", 0))
    t_round_start_equip = float(row.get("t_round_start_equip", 0))

    ct_money = float(row.get("ct_money", 0))
    t_money = float(row.get("t_money", 0))
    money_sum = max(ct_money + t_money, 1)

    ct_alive = float(row.get("ct_alive", 5))
    t_alive = float(row.get("t_alive", 5))
    ct_hp = float(row.get("ct_hp", 500))
    t_hp = float(row.get("t_hp", 500))

    ct_armor = float(row.get("ct_armor", 0))
    t_armor = float(row.get("t_armor", 0))
    ct_armored = float(row.get("ct_armored", 0))
    t_armored = float(row.get("t_armored", 0))

    bomb_planted = 1.0 if row.get("bomb_planted", False) else 0.0
    bomb_time = float(row.get("bomb_time_remaining", 0.0))

    map_name = str(row.get("map_name", ""))
    round_number = float(row.get("round_number", 0))
    ct_score = float(row.get("ct_score", 0))
    t_score = float(row.get("t_score", 0))
    half = float(row.get("current_half", 1))

    ct_losses = float(row.get("ct_consecutive_losses", 0))
    t_losses = float(row.get("t_consecutive_losses", 0))
    ct_win_rate_last_5 = float(row.get("ct_win_rate_last_5", 0.5))

    p = priors or {}
    _nan = float("nan")
    team_a_rating = p.get("team_a_avg_rating", _nan)
    team_b_rating = p.get("team_b_avg_rating", _nan)
    rating_diff = team_a_rating - team_b_rating
    team_a_map_wr_long = p.get("team_a_map_winrate_long", _nan)
    team_b_map_wr_long = p.get("team_b_map_winrate_long", _nan)
    team_a_map_wr_short = p.get("team_a_map_winrate_short", _nan)
    team_b_map_wr_short = p.get("team_b_map_winrate_short", _nan)
    h2h = p.get("h2h_win_rate", _nan)
    form_a = p.get("team_a_recent_form", _nan)
    form_b = p.get("team_b_recent_form", _nan)
    form_diff = form_a - form_b
    _picked = p.get("team_a_picked_map", _nan)
    picked = _nan if isinstance(_picked, float) and np.isnan(_picked) else (1.0 if _picked else 0.0)
    _is_lan = p.get("is_lan", _nan)
    is_lan = _nan if isinstance(_is_lan, float) and np.isnan(_is_lan) else float(_is_lan)

    return np.array([
        ct_equip - t_equip,
        ct_equip / equip_sum,
        ct_equip,
        t_equip,
        ct_alive_equip - t_alive_equip,
        ct_alive_equip,
        t_alive_equip,
        ct_round_start_equip - t_round_start_equip,
        ct_round_start_equip,
        t_round_start_equip,
        ct_money - t_money,
        ct_money / money_sum,
        ct_alive - t_alive,
        ct_alive,
        t_alive,
        ct_hp - t_hp,
        ct_hp,
        t_hp,
        ct_armor - t_armor,
        ct_armor,
        t_armor,
        ct_armored - t_armored,
        ct_armored,
        t_armored,
        bomb_planted,
        bomb_time,
        ct_losses,
        t_losses,
        round_number,
        ct_score - t_score,
        ct_score,
        t_score,
        half,
        ct_win_rate_last_5,
        rating_diff,
        team_a_map_wr_long,
        team_b_map_wr_long,
        team_a_map_wr_short,
        team_b_map_wr_short,
        h2h,
        form_diff,
        picked,
        is_lan,
        *_map_ohe(map_name),
    ], dtype=np.float32)
