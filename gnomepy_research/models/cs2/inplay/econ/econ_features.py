"""Economy-aware state features for the in-play map price.

Everything is computed from round-start information of the state's own round (players' money after the buy, loadouts)
and that round's events up to the state's event_seq, so a state never sees its round's outcome."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Bundle

RIFLES = {"ak47", "m4a1_s", "m4a4", "galilar", "famas", "aug", "sg553", "awp", "ssg08", "g3sg1", "scar20", "m249", "negev"}
SIDE_COLS = ["alive", "eq", "awp", "kits", "rifles", "helmets", "kills", "money_alive"]
STATE_COLS = [f"{s}_{c}" for s in ("ct", "t") for c in SIDE_COLS] + ["planted", "since_plant", "t"]
MONEY_CAP = 16000


def player_rounds(b: Bundle, game_ids) -> pd.DataFrame:
    p = b.players[b.players.game_id.isin(set(game_ids))]
    money = p.money.where(p.money < 20000)
    return p.assign(money=money, rifle=p.prim.isin(RIFLES), awp=p.prim == "awp", helmet=p.armor == "kevlar_and_helmet",
                    kit=p.kit.fillna(False).astype(bool))


def state_table(b: Bundle, states: pd.DataFrame) -> pd.DataFrame:
    """CT-vs-T state after each event: round-start loadouts minus the dead, kills each side has made so far (kill
    rewards for the next round's money), money held by the living (it is kept whatever happens)."""
    gids = set(states.game_id)
    P = player_rounds(b, gids)
    roster = {k: list(zip(x.player_id, x.side, x.equip, x.awp, x.kit, x.rifle, x.helmet, x.money.fillna(0)))
              for k, x in P.groupby(["game_id", "round"])}
    ev = b.events[b.events.game_id.isin(gids) & b.events.type.isin(["kill", "bomb_planted"])]
    ev = ev.assign(kside=ev.killer_side.map({"counter_terrorists": "CT", "terrorists": "T"}),
                   tk=(ev.killer_side == ev.victim_side))
    by_round = {k: list(zip(x.seq, x.type, x.victim_id, x.elapsed, x.kside, x.tk))
                for k, x in ev.sort_values("seq").groupby(["game_id", "round"])}
    out = np.full((len(states), len(STATE_COLS)), np.nan)
    for i, (gid, rnd, seq) in enumerate(zip(states.game_id, states["round"], states.event_seq)):
        alive = {pid: v for pid, *v in roster.get((gid, rnd), [])}
        planted, plant_t, t = 0, np.nan, 0.0
        kills = {"CT": 0, "T": 0}
        for s, typ, victim, el, ks, tk in by_round.get((gid, rnd), []):
            if s > seq:
                break
            t = el
            if typ == "kill":
                alive.pop(victim, None)
                if ks in kills and tk is not True:
                    kills[ks] += 1
            else:
                planted, plant_t = 1, el
        v = {"CT": [0.0] * 8, "T": [0.0] * 8}
        for side, eq, awp, kit, rifle, helmet, money in alive.values():
            x = v[side]
            x[0] += 1; x[1] += eq; x[2] += awp; x[3] += kit; x[4] += rifle; x[5] += helmet; x[7] += money
        v["CT"][6], v["T"][6] = kills["CT"], kills["T"]
        out[i] = [*v["CT"], *v["T"], planted, (t - plant_t) if planted else np.nan, t]
    f = pd.DataFrame(out, columns=STATE_COLS, index=states.index)
    f["t"] = states.elapsed.to_numpy()
    return f


def team_money(b: Bundle, game_ids) -> pd.DataFrame:
    """Per round and team A/B: money after the buy (cleaned, per player capped) and the number of players with data."""
    P = player_rounds(b, game_ids)
    g = b.games.drop_duplicates("game_id").set_index("game_id").team_a
    P = P.assign(is_a=P.team_id.to_numpy() == g.reindex(P.game_id).to_numpy())
    agg = P.groupby(["game_id", "round", "is_a"]).agg(m=("money", "sum"), n=("money", "size"),
                                                     mmin=("money", "min")).unstack("is_a")
    out = pd.DataFrame(index=agg.index)
    for lab, flag in (("a", True), ("b", False)):
        out[f"{lab}_mleft"] = agg[("m", flag)] if ("m", flag) in agg else np.nan
        out[f"{lab}_mmin"] = agg[("mmin", flag)] if ("mmin", flag) in agg else np.nan
    return out
