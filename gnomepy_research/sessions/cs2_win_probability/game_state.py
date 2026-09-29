from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RoundPhase(str, Enum):
    FREEZETIME = "freezetime"
    LIVE = "live"
    OVER = "over"
    WARMUP = "warmup"


class BombState(str, Enum):
    NONE = "none"
    PLANTED = "planted"
    DEFUSED = "defused"
    EXPLODED = "exploded"


@dataclass
class PlayerState:
    steam_id: str
    team: str  # "CT" or "T"
    health: int
    armor: int
    money: int
    equipment_value: int
    is_alive: bool


@dataclass
class CS2GameState:
    map_name: str = ""
    map_number: int = 1
    round_number: int = 0
    round_phase: RoundPhase = RoundPhase.WARMUP
    ct_score: int = 0
    t_score: int = 0
    ct_players_alive: int = 5
    t_players_alive: int = 5
    ct_total_hp: int = 500
    t_total_hp: int = 500
    ct_equipment_value: int = 0
    t_equipment_value: int = 0
    ct_team_money: int = 0
    t_team_money: int = 0
    bomb_state: BombState = BombState.NONE
    bomb_time_remaining: float = 0.0
    ct_consecutive_losses: int = 0
    t_consecutive_losses: int = 0
    current_half: int = 1
    ct_round_start_equip: int = 0
    t_round_start_equip: int = 0
    players: list[PlayerState] = field(default_factory=list)
    timestamp_ms: int = 0
    _round_results: list[str] = field(default_factory=list)  # 'CT' or 'T' per completed round

    @property
    def ct_win_rate_last_5(self) -> float:
        recent = self._round_results[-5:]
        return sum(r == "CT" for r in recent) / len(recent) if recent else float("nan")

    def is_ready(self) -> bool:
        """True once we have received at least one GRID event with map data."""
        return self.map_name != "" and self.round_number > 0

    def ct_side_team(self) -> str:
        """Return which series-team ('team_a' or 'team_b') is currently on CT side."""
        return self._ct_side_team

    def update_from_grid_event(self, event: dict) -> None:
        """Apply a parsed GRID event to update state in-place."""
        event_type = event.get("type", "")

        if event_type == "series_state":
            self._apply_series_state(event)
        elif event_type == "round_started":
            self.round_phase = RoundPhase.FREEZETIME
            self.bomb_state = BombState.NONE
            self.bomb_time_remaining = 0.0
            rn = event.get("round_number")
            if rn is not None:
                self.round_number = rn
        elif event_type == "freeze_time_ended":
            self.round_phase = RoundPhase.LIVE
            self.ct_round_start_equip = self.ct_equipment_value
            self.t_round_start_equip = self.t_equipment_value
        elif event_type == "round_ended":
            self.round_phase = RoundPhase.OVER
            winner = event.get("winner_side", "")
            if winner == "CT":
                self.ct_score += 1
                self.t_consecutive_losses += 1
                self.ct_consecutive_losses = 0
                self._round_results.append("CT")
            elif winner == "T":
                self.t_score += 1
                self.ct_consecutive_losses += 1
                self.t_consecutive_losses = 0
                self._round_results.append("T")
            # Half transition at round 12 (standard half length)
            if self.round_number == 12:
                self.current_half = 2
        elif event_type == "bomb_planted":
            self.bomb_state = BombState.PLANTED
            self.bomb_time_remaining = event.get("fuse_time", 40.0)
        elif event_type == "bomb_defused":
            self.bomb_state = BombState.DEFUSED
            self.bomb_time_remaining = 0.0
        elif event_type == "bomb_exploded":
            self.bomb_state = BombState.EXPLODED
            self.bomb_time_remaining = 0.0
        elif event_type == "player_killed":
            self._apply_player_killed(event)
        elif event_type == "player_damaged":
            self._apply_player_damaged(event)
        elif event_type == "economy_updated":
            self._apply_economy(event)

        self.timestamp_ms = event.get("timestamp_ms", self.timestamp_ms)

    def _apply_series_state(self, event: dict) -> None:
        state = event.get("state", {})
        maps = state.get("maps", [])
        if not maps:
            return
        current_map = maps[-1]
        self.map_name = current_map.get("name", self.map_name)
        self.map_number = len(maps)

        ct_team = current_map.get("ct_team", {})
        t_team = current_map.get("t_team", {})
        self.ct_score = ct_team.get("score", self.ct_score)
        self.t_score = t_team.get("score", self.t_score)

        players = current_map.get("players", [])
        self._rebuild_player_state(players)

    def _rebuild_player_state(self, players: list[dict]) -> None:
        self.players = []
        ct_hp = 0
        t_hp = 0
        ct_alive = 0
        t_alive = 0
        ct_equip = 0
        t_equip = 0
        ct_money = 0
        t_money = 0

        for p in players:
            side = p.get("side", "")
            hp = p.get("health", 0)
            alive = hp > 0
            equip = p.get("equipment_value", 0)
            money = p.get("money", 0)

            ps = PlayerState(
                steam_id=str(p.get("id", "")),
                team=side,
                health=hp,
                armor=p.get("armor", 0),
                money=money,
                equipment_value=equip,
                is_alive=alive,
            )
            self.players.append(ps)

            if side == "CT":
                if alive:
                    ct_hp += hp
                    ct_alive += 1
                ct_equip += equip
                ct_money += money
            elif side == "T":
                if alive:
                    t_hp += hp
                    t_alive += 1
                t_equip += equip
                t_money += money

        self.ct_total_hp = ct_hp
        self.t_total_hp = t_hp
        self.ct_players_alive = ct_alive
        self.t_players_alive = t_alive
        self.ct_equipment_value = ct_equip
        self.t_equipment_value = t_equip
        self.ct_team_money = ct_money
        self.t_team_money = t_money

    def _apply_player_killed(self, event: dict) -> None:
        victim_id = str(event.get("victim_id", ""))
        victim_side = event.get("victim_side", "")
        for p in self.players:
            if p.steam_id == victim_id:
                p.is_alive = False
                p.health = 0
                break
        if victim_side == "CT":
            self.ct_players_alive = max(0, self.ct_players_alive - 1)
            self.ct_total_hp = sum(p.health for p in self.players if p.team == "CT" and p.is_alive)
        elif victim_side == "T":
            self.t_players_alive = max(0, self.t_players_alive - 1)
            self.t_total_hp = sum(p.health for p in self.players if p.team == "T" and p.is_alive)

    def _apply_player_damaged(self, event: dict) -> None:
        victim_id = str(event.get("victim_id", ""))
        damage = event.get("damage_dealt", 0)
        for p in self.players:
            if p.steam_id == victim_id and p.is_alive:
                p.health = max(0, p.health - damage)
                break
        self.ct_total_hp = sum(p.health for p in self.players if p.team == "CT" and p.is_alive)
        self.t_total_hp = sum(p.health for p in self.players if p.team == "T" and p.is_alive)

    def _apply_economy(self, event: dict) -> None:
        ct = event.get("ct", {})
        t = event.get("t", {})
        self.ct_equipment_value = ct.get("equipment_value", self.ct_equipment_value)
        self.t_equipment_value = t.get("equipment_value", self.t_equipment_value)
        self.ct_team_money = ct.get("money", self.ct_team_money)
        self.t_team_money = t.get("money", self.t_team_money)
