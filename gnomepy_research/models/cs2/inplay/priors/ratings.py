"""Point-in-time pre-map ratings and series context, one row per map.

Every map is queried at its begin time and its result applied at its end time, so a map's features only use maps that
ended strictly before it began (the harness's rule). Results come from games.map_a_wins where known, otherwise from the
final score implied by the last round's score state (blanked maps in the scored month), which is fixed once the map
has ended. Ratings are logit-scale Elo variants: team, team-on-map, and player (team strength = roster mean)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import Bundle
from ..features import need


@dataclass
class Params:
    k_team: float = 0.15
    k_map: float = 0.12
    k_player: float = 0.12
    k_round: float = 0.8
    k_round_map: float = 0.6
    k_round_player: float = 0.6


@dataclass
class State:
    team: dict = field(default_factory=dict)
    team_map: dict = field(default_factory=dict)
    player: dict = field(default_factory=dict)
    rteam: dict = field(default_factory=dict)
    rmap: dict = field(default_factory=dict)
    rplayer: dict = field(default_factory=dict)
    n_team: dict = field(default_factory=dict)
    n_player: dict = field(default_factory=dict)
    last_team: dict = field(default_factory=dict)
    done: set = field(default_factory=set)
    series: dict = field(default_factory=dict)

    def copy(self) -> State:
        # Pricing continues from the training state without touching it, so every predict call starts identically.
        return State(**{k: (set(v) if isinstance(v, set) else {kk: (list(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()})
                        for k, v in self.__dict__.items()})


def game_info(b: Bundle, game_ids=None) -> pd.DataFrame:
    """Per map: teams, times, map name, result (known or implied), final rounds, and rosters from its first round."""
    g = b.games if game_ids is None else b.games[b.games.game_id.isin(game_ids)]
    g = g[["game_id", "match_id", "map_no", "bo", "tier", "begin", "end", "match_begin", "team_a", "team_b", "map_a_wins"]].copy()
    r = b.rounds[b.rounds.game_id.isin(set(g.game_id))].sort_values(["game_id", "round"])
    first = r.groupby("game_id").first()
    last = r.groupby("game_id").last()
    nd = np.array([need(a, c) for a, c in zip(last.score_a, last.score_b)]).reshape(-1, 2)
    implied = pd.Series(np.where((nd[:, 0] == 1) & (nd[:, 1] > 1), 1.0, np.where((nd[:, 1] == 1) & (nd[:, 0] > 1), 0.0, np.nan)),
                        index=last.index)
    g["map"] = g.game_id.map(first["map"])
    g["side_a1"] = g.game_id.map(first["side_a"])
    g["result"] = g.map_a_wins.where(g.map_a_wins.notna(), g.game_id.map(implied))
    fa = last.score_a + (implied == 1)
    fb = last.score_b + (implied == 0)
    g["rd"] = g.game_id.map(fa - fb)
    g["rounds_a"] = g.game_id.map(fa).fillna(0)
    g["n_rounds"] = g.game_id.map(fa + fb).fillna(0)
    g["has_rounds"] = g.game_id.isin(first.index)
    p = b.players[b.players.game_id.isin(set(g.game_id))][["game_id", "round", "team_id", "player_id"]]
    p = p[p["round"] == p.groupby("game_id")["round"].transform("min")]
    ro = {k: tuple(sorted(set(x))) for k, x in p.groupby(["game_id", "team_id"]).player_id}
    g["roster_a"] = [ro.get((gi, t), ()) for gi, t in zip(g.game_id, g.team_a)]
    g["roster_b"] = [ro.get((gi, t), ()) for gi, t in zip(g.game_id, g.team_b)]
    return g.reset_index(drop=True)


FEATS = ["r_team", "r_map", "r_player", "n_a", "n_b", "np_a", "np_b", "roster_known", "rest_a", "rest_b",
         "s_a_wins", "s_b_wins", "s_rd", "s_missing", "q_team", "q_map", "q_player"]


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def run(info: pd.DataFrame, prm: Params, st: State | None = None, floor: pd.Timestamp | None = None) -> tuple[pd.DataFrame, State]:
    """Without a begin time the harness lets a map see only the training history, so with `floor` (the scoring
    month's start) such maps are queried before any of the month's results; in training they use the match start.
    Maps without an end time are never visible to others, so they never update."""
    st = st or State()
    info = info[~info.game_id.isin(st.done)]
    rows = info.to_dict("records")
    q = [(r["begin"] if pd.notna(r["begin"]) else (floor if floor is not None else r["match_begin"]), 0, i) for i, r in enumerate(rows)]
    order = sorted(q + [(r["end"], 1, i) for i, r in enumerate(rows) if pd.notna(r["end"])], key=lambda x: (x[0], x[1]))
    out = np.full((len(rows), len(FEATS)), np.nan)
    T, M, P, N, NP, L = st.team, st.team_map, st.player, st.n_team, st.n_player, st.last_team
    QT, QM, QP = st.rteam, st.rmap, st.rplayer
    for t, kind, i in order:
        r = rows[i]
        a, bb, mp = r["team_a"], r["team_b"], r["map"]
        ra, rb = r["roster_a"], r["roster_b"]
        known = len(ra) >= 4 and len(rb) >= 4
        if kind == 0:
            pa = np.mean([P.get(x, 0.0) for x in ra]) if known else 0.0
            pb = np.mean([P.get(x, 0.0) for x in rb]) if known else 0.0
            npa = np.mean([NP.get(x, 0) for x in ra]) if known else 0.0
            npb = np.mean([NP.get(x, 0) for x in rb]) if known else 0.0
            ma = M.get((a, mp), 0.0) if isinstance(mp, str) else 0.0
            mb = M.get((bb, mp), 0.0) if isinstance(mp, str) else 0.0
            ser = st.series.get(r["match_id"], [])
            prev = [s for s in ser if s[0] < t and s[1] < r["map_no"]]
            sa = sum(1 for s in prev if s[2] == 1)
            sb = sum(1 for s in prev if s[2] == 0)
            srd = sum(s[3] for s in prev if s[3] == s[3])
            ra_ = (t - L[a]).total_seconds() / 86400 if a in L else np.nan
            rb_ = (t - L[bb]).total_seconds() / 86400 if bb in L else np.nan
            out[i] = [T.get(a, 0.0) - T.get(bb, 0.0), ma - mb, pa - pb, N.get(a, 0), N.get(bb, 0), npa, npb, float(known),
                      ra_, rb_, sa, sb, srd, r["map_no"] - 1 - len(prev), QT.get(a, 0.0) - QT.get(bb, 0.0),
                      (QM.get((a, mp), 0.0) - QM.get((bb, mp), 0.0)) if isinstance(mp, str) else 0.0,
                      (np.mean([QP.get(x, 0.0) for x in ra]) - np.mean([QP.get(x, 0.0) for x in rb])) if known else 0.0]
        else:
            y = r["result"]
            st.done.add(r["game_id"])
            if y != y:
                continue
            e = sig(T.get(a, 0.0) - T.get(bb, 0.0))
            em = sig(T.get(a, 0.0) + M.get((a, mp), 0.0) - T.get(bb, 0.0) - M.get((bb, mp), 0.0)) if isinstance(mp, str) else 0.5
            T[a] = T.get(a, 0.0) + prm.k_team * (y - e)
            T[bb] = T.get(bb, 0.0) - prm.k_team * (y - e)
            N[a] = N.get(a, 0) + 1
            N[bb] = N.get(bb, 0) + 1
            L[a] = L[bb] = t
            if isinstance(mp, str):
                M[(a, mp)] = M.get((a, mp), 0.0) + prm.k_map * (y - em)
                M[(bb, mp)] = M.get((bb, mp), 0.0) - prm.k_map * (y - em)
            if known:
                ep = sig(np.mean([P.get(x, 0.0) for x in ra]) - np.mean([P.get(x, 0.0) for x in rb]))
                for x in ra:
                    P[x] = P.get(x, 0.0) + prm.k_player * (y - ep)
                    NP[x] = NP.get(x, 0) + 1
                for x in rb:
                    P[x] = P.get(x, 0.0) - prm.k_player * (y - ep)
                    NP[x] = NP.get(x, 0) + 1
            if r["has_rounds"] and r["n_rounds"] > 0:
                # Round share carries more per map than the result: a logit-scale round-strength Elo per team,
                # per team on this map, and per player (roster mean).
                share = r["rounds_a"] / r["n_rounds"]
                w = min(r["n_rounds"], 30) / 20
                eq = sig(QT.get(a, 0.0) - QT.get(bb, 0.0))
                QT[a] = QT.get(a, 0.0) + prm.k_round * w * (share - eq) / 4
                QT[bb] = QT.get(bb, 0.0) - prm.k_round * w * (share - eq) / 4
                if isinstance(mp, str):
                    eqm = sig(QT.get(a, 0.0) + QM.get((a, mp), 0.0) - QT.get(bb, 0.0) - QM.get((bb, mp), 0.0))
                    QM[(a, mp)] = QM.get((a, mp), 0.0) + prm.k_round_map * w * (share - eqm) / 4
                    QM[(bb, mp)] = QM.get((bb, mp), 0.0) - prm.k_round_map * w * (share - eqm) / 4
                if known:
                    eqp = sig(np.mean([QP.get(x, 0.0) for x in ra]) - np.mean([QP.get(x, 0.0) for x in rb]))
                    for x in ra:
                        QP[x] = QP.get(x, 0.0) + prm.k_round_player * w * (share - eqp) / 4
                    for x in rb:
                        QP[x] = QP.get(x, 0.0) - prm.k_round_player * w * (share - eqp) / 4
            if r["has_rounds"]:
                st.series.setdefault(r["match_id"], []).append((r["end"], r["map_no"], y, r["rd"]))
    return pd.DataFrame(out, columns=FEATS).assign(game_id=info.game_id.to_numpy()), st
