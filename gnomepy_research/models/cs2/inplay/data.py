"""The competition bundle: PandaScore CS2 maps as six tables keyed by game_id (one game = one map).

    games    one row per map: match/league/tier, begin/end (UTC), team_a/team_b ids and names, map_a_wins
    rounds   one row per round from team A's view: score before the round, side, round-start economy of both teams,
             and the round's outcome (a_wins, outcome, a_alive_end, b_alive_end) and map_a_wins
    players  one row per player per round: PandaScore player id, side, money before/after the buy, equipment value,
             primary weapon before/after, armor, kit; and end-of-round kills, assists, hp_end, alive_end
    events   2026 timelines, one row per event in time order (seq): round_start, kill, bomb_planted, bomb_exploded,
             round_end, with killer/victim/weapon/modifiers, planter/site, round winner and outcome, scores
    elo      point-in-time map Elo gap for team A (logit scale)
    priors   kickoff market / e12 / pool series prices turned into single-map prices for team A (market-listed BO3s)

Outcome columns (OUTCOME) are blanked for the maps being predicted."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ("games", "rounds", "players", "events", "elo", "priors")
OUTCOME = {
    "games": ["map_a_wins"],
    "rounds": ["a_wins", "outcome", "map_a_wins", "a_alive_end", "b_alive_end"],
    "players": ["alive_end", "kills", "assists", "hp_end"],
}


@dataclass(frozen=True)
class Bundle:
    games: pd.DataFrame
    rounds: pd.DataFrame
    players: pd.DataFrame
    events: pd.DataFrame
    elo: pd.DataFrame
    priors: pd.DataFrame

    @classmethod
    def load(cls, *dirs: str | Path) -> Bundle:
        parts = [cls(**{t: pd.read_parquet(Path(d) / f"{t}.parquet") for t in TABLES}) for d in dirs]
        return parts[0] if len(parts) == 1 else concat(parts)

    def subset(self, game_ids) -> Bundle:
        ids = set(game_ids)
        return Bundle(**{f.name: getattr(self, f.name)[getattr(self, f.name).game_id.isin(ids)] for f in fields(self)})

    def ended_before(self, t: pd.Timestamp) -> Bundle:
        """Maps finished before t: everything about them, outcomes included, is known at t."""
        return self.subset(self.games.game_id[self.games.end < t])

    def without_outcomes(self) -> Bundle:
        out = {}
        for f in fields(self):
            t = getattr(self, f.name)
            cols = [c for c in OUTCOME.get(f.name, []) if c in t]
            out[f.name] = t.assign(**{c: np.nan for c in cols}) if cols else t
        return Bundle(**out)

    def truncate(self, game_id: int, round_no: int, event_seq: int) -> Bundle:
        """This map as it stood just after event `event_seq` of round `round_no`: later rounds and events removed."""
        keep = lambda t, col: t[(t.game_id != game_id) | (t[col] <= round_no)]
        ev = self.events
        return replace(self, rounds=keep(self.rounds, "round"), players=keep(self.players, "round"),
                       events=ev[(ev.game_id != game_id) | (ev.seq <= event_seq)])


def concat(parts: list[Bundle]) -> Bundle:
    return Bundle(**{t: pd.concat([getattr(p, t) for p in parts], ignore_index=True) for t in TABLES})
