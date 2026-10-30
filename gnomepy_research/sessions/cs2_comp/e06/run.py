"""One chronological pass: fold events into the filter, and at each cutoff price the series of the following week."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e06.events import FACEIT, HLTV, MapEvent, build_events
from gnomepy_research.sessions.cs2_comp.e06.roster import RosterParams, Rosters
from gnomepy_research.sessions.cs2_comp.e06.fusion import Combined
from gnomepy_research.sessions.cs2_comp.e06.skill import Params, stacked

logger = logging.getLogger(__name__)


def to_seconds(ts) -> float:
    return pd.Timestamp(ts).value / 1e9


def load_inputs(data_dir: Path, use_faceit: bool):
    data_dir = Path(data_dir)
    events = build_events(data_dir, use_faceit)
    rankings = pd.read_parquet(data_dir / "hltv/cs2_team_rankings.parquet")
    moves = pd.read_parquet(data_dir / "hltv/cs2_roster_moves.parquet").sort_values(["event_date", "person_id"])
    return events, rankings, list(moves.itertuples(index=False))


def run_pass(events: list[MapEvent], rankings, moves: list, params: Params, queries: pd.DataFrame,
             roster_params: RosterParams | None = None, known_lineups: dict | None = None,
             use_faceit: bool = True, joint: bool = True) -> pd.DataFrame:
    """queries: match_id, week_cutoff, kickoff (UTC seconds), team_a_id, team_b_id. known_lineups maps match_id to the
    two announced fives and is for diagnostics only."""
    filt = Combined(params, rankings, use_faceit, joint)
    base = Combined(params, rankings, False) if use_faceit else None
    rosters = Rosters(roster_params)
    ei, mi = 0, 0
    out = []
    for cutoff, wk in queries.groupby("week_cutoff", sort=True):
        cs = to_seconds(cutoff)
        filt.h.rank_limit = cs / 86400.0
        while ei < len(events) and events[ei].t < cs:
            e = events[ei]
            filt.observe(e)
            if base is not None and e.source == HLTV:
                base.observe(e)
            if e.source == HLTV:
                rosters.observe_map(e.team_a, e.day, e.players_a)
                rosters.observe_map(e.team_b, e.day, e.players_b)
            ei += 1
        filt.advance(cs)
        if base is not None:
            base.advance(cs)
            base.h.rank_limit = cs / 86400.0
        while mi < len(moves) and to_seconds(moves[mi].event_date) < cs:
            rosters.observe_move(moves[mi])
            mi += 1
        for r in wk.itertuples(index=False):
            day = r.kickoff / 86400.0
            ea, xa = rosters.expected_five(int(r.team_a_id))
            eb, xb = rosters.expected_five(int(r.team_b_id))
            if known_lineups is not None:
                la, lb = known_lineups[r.match_id]
                res = filt.price_known(day, int(r.team_a_id), la, int(r.team_b_id), lb)
                depth = min(filt.faceit_depth(la), filt.faceit_depth(lb))
            else:
                res = filt.price(day, int(r.team_a_id), ea, xa, int(r.team_b_id), eb, xb)
                depth = min(filt.faceit_depth([pl for pl, _ in ea]), filt.faceit_depth([pl for pl, _ in eb]))
            ta, tb = int(r.team_a_id), int(r.team_b_id)
            thin = not r.is_lan and (min(rosters.maps_within(ta, cs / 86400.0, 90.0), rosters.maps_within(tb, cs / 86400.0, 90.0))
                                     <= params.thin_maps or min(filt.h.rank_of(ta, day), filt.h.rank_of(tb, day)) > params.thin_rank)
            if base is not None:
                bres = base.price(day, ta, ea, xa, tb, eb, xb) if known_lineups is None else base.price_known(day, ta, la, tb, lb)
                q_final = stacked(res["q"], bres["q"], thin, params)
                extra = {"q_faceit": res["q"], "q_base": bres["q"]}
            else:
                q_final = stacked(res["q"], res["q"], thin, params)
                extra = {"q_faceit": res["q"], "q_base": res["q"]}
            out.append({"match_id": r.match_id, "fit_cutoff": cutoff, **res, "q": q_final, **extra, "thin_hat": bool(thin),
                        "faceit_depth": depth})
    return pd.DataFrame(out)
