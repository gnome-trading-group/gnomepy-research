"""Economy features for the in-play map price.

Everything is computed from round-start information of the state's own round (players' money after the buy, loadouts),
the score progression of earlier rounds and that round's events up to the state's event_seq, so a state never sees its
round's outcome. post_table (a round's actual pre-buy economy) is used only to build training targets.

Money rules, checked against players.money_pre/money (zero-kill losers of eliminated rounds, 2025-06 onward):
- the loss counter starts at 1 at each half start (a pistol-round loser gets $1,900), a loss pays 1400 + 500 * counter
  and raises it (max 4: $3,400), a win lowers it by one (it does not reset: lose-lose-win-lose pays $1,900);
- a win pays $3,250 ($3,500 for bomb objectives); Ts losing after a plant get a bonus;
- kill rewards depend on the weapon (AWP $100, SMGs $600, shotguns $900, knife $1,500, most others $300)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Bundle
from ..features import resets

RIFLES = {"ak47", "m4a1_s", "m4a4", "galilar", "famas", "aug", "sg553", "awp", "ssg08", "g3sg1", "scar20", "m249", "negev"}
KILL_REWARD = {"awp": 100, "mp9": 600, "mac10": 600, "mp7": 600, "ump45": 600, "mp5": 600, "bizon": 600, "nova": 900,
               "mag7": 900, "xm1014": 900, "sawedoff": 900, "knife": 1500, "taser": 0}
SIDE_COLS = ["alive", "eq", "awp", "kits", "rifles", "helmets", "kills", "proj_w", "proj_l", "nbuy_w", "nbuy_l", "nfull_w",
             "nfull_l"]
STATE_COLS = [f"{s}_{c}" for s in ("ct", "t") for c in SIDE_COLS] + ["planted", "since_plant", "t"]
POST_COLS = [f"{t}_{c}" for t in ("a", "b") for c in ("mp", "eqp", "rif", "kawp", "nbuy")] + ["d_nbuy_post", "d_res_post"]
MONEY_CAP = 16000
WIN_PAY = 3250
PLANT_BONUS = 800
FULL_BUY = 3900
TOPUP = 1000
# Side-specific full buys: AK + armor + utility vs M4 + armor + kit + utility.
SIDE_FULL = {"T": 3700, "CT": 4300}


def player_rounds(b: Bundle, game_ids) -> pd.DataFrame:
    p = b.players[b.players.game_id.isin(set(game_ids))]
    money = p.money.where(p.money < 20000)
    return p.assign(money=money, rifle=p.prim.isin(RIFLES), awp=p.prim == "awp", helmet=p.armor == "kevlar_and_helmet",
                    kit=p.kit.fillna(False).astype(bool))


def loss_counters(rt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each team's loss counter at the start of each round (rt sorted by game, round), from the score progression."""
    games, rounds = rt.game_id.to_numpy(), rt["round"].to_numpy()
    sa = rt.score_a.to_numpy()
    la, lb = np.ones(len(rt), int), np.ones(len(rt), int)
    rs = resets(rounds)
    for i in range(1, len(rt)):
        if games[i] != games[i - 1] or rs[i] or rounds[i] != rounds[i - 1] + 1:
            continue
        a_won = sa[i] > sa[i - 1]
        la[i] = max(la[i - 1] - 1, 0) if a_won else min(la[i - 1] + 1, 4)
        lb[i] = min(lb[i - 1] + 1, 4) if a_won else max(lb[i - 1] - 1, 0)
    return la, lb


def state_table(b: Bundle, states: pd.DataFrame, ct_ctr: np.ndarray, t_ctr: np.ndarray) -> pd.DataFrame:
    """CT-vs-T state after each event: round-start loadouts minus the dead, kills so far, and each side's projected
    pre-buy money of the next round if it wins (proj_w) or loses (proj_l) this round from here, with the number of
    players who could then buy rifle + armor (nbuy: survivors keep their rifle and only need a top-up) and the number
    with full-buy money regardless of weapons (nfull). Dead players lose their weapons but keep their money."""
    gids = set(states.game_id)
    P = player_rounds(b, gids)
    roster = {k: list(zip(x.player_id, x.side, x.equip, x.awp, x.kit, x.rifle, x.helmet, x.money.fillna(0)))
              for k, x in P.groupby(["game_id", "round"])}
    ev = b.events[b.events.game_id.isin(gids) & b.events.type.isin(["kill", "bomb_planted"])]
    ev = ev.assign(kside=ev.killer_side.map({"counter_terrorists": "CT", "terrorists": "T"}),
                   tk=(ev.killer_side == ev.victim_side), reward=ev.weapon.map(KILL_REWARD).fillna(300))
    by_round = {k: list(zip(x.seq, x.type, x.victim_id, x.elapsed, x.kside, x.tk, x.killer_id, x.reward))
                for k, x in ev.sort_values("seq").groupby(["game_id", "round"])}
    out = np.full((len(states), len(STATE_COLS)), np.nan)
    for i, (gid, rnd, seq) in enumerate(zip(states.game_id, states["round"], states.event_seq)):
        players = roster.get((gid, rnd), [])
        alive = {pid for pid, *_ in players}
        planted, plant_t, t = 0, np.nan, 0.0
        kills = {"CT": 0, "T": 0}
        earned = {}
        for s, typ, victim, el, ks, tk, killer, reward in by_round.get((gid, rnd), []):
            if s > seq:
                break
            t = el
            if typ == "kill":
                alive.discard(victim)
                if ks in kills and tk is not True:
                    kills[ks] += 1
                    earned[killer] = earned.get(killer, 0) + reward
            else:
                planted, plant_t = 1, el
        loss_pay = {"CT": 1400 + 500 * min(ct_ctr[i], 4), "T": 1400 + 500 * min(t_ctr[i], 4) + PLANT_BONUS * planted}
        v = {"CT": [0.0] * len(SIDE_COLS), "T": [0.0] * len(SIDE_COLS)}
        for pid, side, eq, awp, kit, rifle, helmet, money in players:
            if side not in v:
                continue
            x = v[side]
            base = money + earned.get(pid, 0)
            w, l = min(base + WIN_PAY, MONEY_CAP), min(base + loss_pay[side], MONEY_CAP)
            x[7] += w; x[8] += l
            keeps_rifle = pid in alive and rifle
            x[9] += (w >= FULL_BUY) or (keeps_rifle and w >= TOPUP)
            x[10] += (l >= FULL_BUY) or (keeps_rifle and l >= TOPUP)
            x[11] += w >= FULL_BUY; x[12] += l >= FULL_BUY
            if pid in alive:
                x[0] += 1; x[1] += eq; x[2] += awp; x[3] += kit; x[4] += rifle; x[5] += helmet
        v["CT"][6], v["T"][6] = kills["CT"], kills["T"]
        out[i] = [*v["CT"], *v["T"], planted, (t - plant_t) if planted else np.nan, t]
    f = pd.DataFrame(out, columns=STATE_COLS, index=states.index)
    f["t"] = states.elapsed.to_numpy()
    return f


def is_team_a(b: Bundle, P: pd.DataFrame) -> np.ndarray:
    g = b.games.drop_duplicates("game_id").set_index("game_id").team_a
    return P.team_id.to_numpy() == g.reindex(P.game_id).to_numpy()


def team_money(b: Bundle, game_ids) -> pd.DataFrame:
    """Per round and team A/B: money after the buy (cleaned), summed."""
    P = player_rounds(b, game_ids)
    agg = P.assign(is_a=is_team_a(b, P)).groupby(["game_id", "round", "is_a"]).money.sum().unstack("is_a")
    return pd.DataFrame({"a_mleft": agg.get(True), "b_mleft": agg.get(False)}, index=agg.index)


def post_table(b: Bundle, game_ids) -> pd.DataFrame:
    """Per round and team A/B, the economy before the buy: money, equipment kept from the previous round (survivors'
    weapons), kept rifles and AWPs, and players able to buy rifle + armor. Blank at half and overtime-half starts."""
    P = player_rounds(b, game_ids)
    mp = P.money_pre.where(P.money_pre < 20000)
    rifle = P.prim_pre.isin(RIFLES)
    P = P.assign(mp=mp, rifle_pre=rifle, awp_pre=P.prim_pre == "awp",
                 nbuy=(mp >= P.side.map(SIDE_FULL)) | (rifle & (mp >= TOPUP)), is_a=is_team_a(b, P))
    agg = P.groupby(["game_id", "round", "is_a"]).agg(mp=("mp", "sum"), eqp=("equip_pre", "sum"), rif=("rifle_pre", "sum"),
                                                     kawp=("awp_pre", "sum"), nbuy=("nbuy", "sum")).unstack("is_a")
    out = pd.DataFrame(index=agg.index)
    for lab, flag in (("a", True), ("b", False)):
        for c in ("mp", "eqp", "rif", "kawp", "nbuy"):
            out[f"{lab}_{c}"] = agg[(c, flag)] if (c, flag) in agg else np.nan
    out["d_nbuy_post"] = out.a_nbuy - out.b_nbuy
    out["d_res_post"] = out.a_mp + out.a_eqp - out.b_mp - out.b_eqp
    rounds = out.index.get_level_values("round").to_numpy()
    out.loc[resets(rounds), POST_COLS] = np.nan
    return out[POST_COLS]
