"""In-round dynamics read off the timeline: who is still alive and with what (round-start loadouts, weapons seen in
kills), how the round has gone so far (opening kill, trades, deaths in the last seconds, time since the last kill),
the plant (site, bomb time left against the 5/10-second defuse, kits alive) and whether the round is already over.

Everything at a state comes from that round's players' round-start rows (loadout, never outcomes) and the events of
the round with seq <= the state's event_seq, walked in seq order, so a bundle truncated at the state gives the same
row."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Bundle
from ..features import need

RIFLES = {"ak47", "m4a1_s", "m4a4", "galilar", "famas", "aug", "sg553"}
SNIPERS = {"awp"}
SMGS = {"mp9", "mac10", "mp7", "ump45", "mp5", "p90", "bizon", "xm1014", "mag7", "nova", "sawedoff", "ssg08",
        "negev", "m249", "g3sg1", "scar20"}
ROUND_SECONDS = 115.0
BOMB_SECONDS = 41.0
TRADE_WINDOW = 5.0

SIDE_COLS = ["alive", "eq", "awp", "kits", "rifles", "smgs", "noprim", "helm", "armor", "money", "skill"]
CT_T = {"CT": 0, "T": 1, "counter_terrorists": 0, "terrorists": 1}


def weapon_class(w) -> int:
    # 3 sniper, 2 rifle, 1 smg/heavy/scout, 0 pistol or nothing
    if w in SNIPERS:
        return 3
    if w in RIFLES:
        return 2
    if w in SMGS:
        return 1
    return 0


DYN = ([f"{s}_{c}" for s in ("ct", "t") for c in SIDE_COLS] +
       ["planted", "since_plant", "t", "site", "bomb_left", "clock_left", "defuse_kit_ok", "defuse_nokit_ok",
        "n_kills", "since_kill", "last_dead", "ct_dead_5s", "t_dead_5s", "last_trade", "open_dead", "open_t",
        "ct_dead_hs", "t_dead_hs", "ct_dead_util", "t_dead_util", "last_kill_cls", "ct_max_cls", "t_max_cls",
        "ct_pickup", "t_pickup", "over", "over_ct_won"])


def dyn_features(b: Bundle, states: pd.DataFrame, ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    gids = set(states.game_id)
    rmap = {} if ratings is None else dict(zip(zip(ratings.game_id, ratings.player_id), ratings.rating))
    P = b.players[b.players.game_id.isin(gids)]
    keys = set(zip(states.game_id, states["round"]))
    roster = {}
    for (g, r), x in P.groupby(["game_id", "round"]):
        if (g, r) not in keys:
            continue
        roster[(g, r)] = {int(pid): [CT_T.get(sd, -1), float(eq), weapon_class(pr), bool(kt), arm == "kevlar_and_helmet",
                                     arm is not None, 0.0 if mn != mn else float(mn), rmap.get((g, int(pid)), 0.0)]
                          for pid, sd, eq, pr, kt, arm, mn in zip(x.player_id, x.side, x.equip, x.prim,
                                                                  x.kit.fillna(False), x.armor, x.money)}
    E = b.events[b.events.game_id.isin(gids)].sort_values(["game_id", "seq"])
    E = E[[(g, r) in keys for g, r in zip(E.game_id, E["round"])]]
    want = {}
    for i, (g, s) in enumerate(zip(states.game_id, states.event_seq)):
        want.setdefault((g, s), []).append(i)
    out = np.full((len(states), len(DYN)), np.nan)
    E = E.assign(**{c: E[c].fillna(-1).astype("int64") for c in ("killer_id", "victim_id")},
                 **{c: E[c].eq(True).to_numpy() for c in ("headshot", "through_smoke", "attacker_blind", "flash_assist")})
    cols = [E[c].to_numpy() for c in ("game_id", "round", "seq", "type", "elapsed", "killer_id", "victim_id",
                                      "victim_side", "killer_side", "weapon", "headshot", "through_smoke",
                                      "attacker_blind", "flash_assist", "bomb_site", "winner_side")]
    cur = None
    for g, r, s, typ, el, kid, vid, vside, kside, wpn, hs, smk, blind, flash, site, wside in zip(*cols):
        if (g, r) != cur:
            cur = (g, r)
            alive = {pid: list(v) for pid, v in roster.get(cur, {}).items()}
            planted, plant_t, site_c, t = 0, np.nan, 0, 0.0
            kills = []  # (elapsed, victim side, killer id, victim id)
            dead_hs = [0, 0]; dead_util = [0, 0]; pickup = [0, 0]
            last_cls, open_dead, open_t, last_trade = np.nan, 0, np.nan, 0
            over, over_ct = 0, np.nan
        el = float(el)
        t = max(t, el)
        if typ == "kill":
            vs = CT_T.get(vside, -1)
            vid_i = None if vid < 0 else int(vid)
            kid_i = None if kid < 0 else int(kid)
            if vid_i is not None and vid_i in alive:
                vs = alive[vid_i][0]
                alive.pop(vid_i)
            if kid_i is not None and kid_i in alive:
                c = weapon_class(wpn)
                if c >= 1:
                    k = alive[kid_i]
                    if c != k[2]:
                        pickup[k[0]] += 1
                    k[2] = c
            if vs in (0, 1):
                if not kills:
                    open_dead, open_t = (1 if vs == 0 else -1), el
                last_trade = int(bool(kills) and kills[-1][1] != vs and el - kills[-1][0] <= TRADE_WINDOW
                                 and kid_i is not None and kills[-1][3] is not None and kid_i != kills[-1][2]
                                 and vid_i == kills[-1][2])
                kills.append((el, vs, kid_i, vid_i))
                dead_hs[vs] += hs
                dead_util[vs] += smk + blind + flash
                last_cls = weapon_class(wpn)
        elif typ == "bomb_planted":
            planted, plant_t = 1, el
            site_c = 1 if site == "A" else 2 if site == "B" else 0
        elif typ == "bomb_exploded":
            over, over_ct = 1, 0.0
        elif typ == "round_end":
            over = 1
            if wside in ("CT", "counter_terrorists"):
                over_ct = 1.0
            elif wside in ("T", "terrorists"):
                over_ct = 0.0
        idx = want.get((g, s))
        if idx is None:
            continue
        v = np.zeros((2, len(SIDE_COLS)))
        mx = [0, 0]
        for side, eq, cls, kit, helm, arm, money, skill in alive.values():
            if side not in (0, 1):
                continue
            x = v[side]
            x[0] += 1; x[1] += eq; x[2] += cls == 3; x[3] += kit; x[4] += cls == 2; x[5] += cls == 1
            x[6] += cls == 0; x[7] += helm; x[8] += arm; x[9] += money; x[10] += skill
            mx[side] = max(mx[side], cls)
        since_plant = t - plant_t if planted else np.nan
        bomb_left = BOMB_SECONDS - since_plant if planted else np.nan
        recent = [k for k in kills if t - k[0] <= TRADE_WINDOW]
        out[idx] = [*v[0], *v[1], planted, since_plant, t, site_c, bomb_left,
                  np.nan if planted else ROUND_SECONDS - t,
                  float(planted and bomb_left >= 5 and v[0][3] > 0) if planted else np.nan,
                  float(planted and bomb_left >= 10) if planted else np.nan,
                  len(kills), (t - kills[-1][0]) if kills else np.nan, (1 if kills[-1][1] == 0 else -1) if kills else 0,
                  sum(k[1] == 0 for k in recent), sum(k[1] == 1 for k in recent), last_trade, open_dead, open_t,
                  dead_hs[0], dead_hs[1], dead_util[0], dead_util[1], last_cls, mx[0], mx[1], pickup[0], pickup[1],
                  over, over_ct]
    return pd.DataFrame(out, columns=DYN, index=states.index)


FORM = ["a_kd_prev", "a_upsets_prev", "rounds_prev", "a_gun_wr_prev", "gun_prev"]


def form_features(b: Bundle, keys: pd.DataFrame) -> pd.DataFrame:
    """Team A's form earlier in this map, before round `round` of each (game_id, round) in keys: kill-death
    difference from the timeline, and round wins against the economy (wins with less equipment minus losses with
    more), from the score progression. Only rounds before the keyed round are read."""
    gids = set(keys.game_id)
    g = b.games[b.games.game_id.isin(gids)].set_index("game_id")
    ev = b.events[b.events.game_id.isin(gids) & (b.events.type == "kill")]
    ev = ev[ev.victim_team.notna()]
    ta = g.team_a.reindex(ev.game_id).to_numpy()
    d = pd.DataFrame(dict(game_id=ev.game_id.to_numpy(), round=ev["round"].to_numpy(),
                          kd=np.where(ev.victim_team.to_numpy(dtype=float) == ta, -1.0, 1.0)))
    kd = d.groupby(["game_id", "round"]).kd.sum()
    r = b.rounds[b.rounds.game_id.isin(gids)].sort_values(["game_id", "round"])
    r = r[["game_id", "round", "score_a", "a_equip", "b_equip"]].copy()
    nxt = r.groupby("game_id").score_a.shift(-1)
    nxt_round = r.groupby("game_id")["round"].shift(-1)
    r["won"] = np.where(nxt_round == r["round"] + 1, nxt - r.score_a, np.nan)
    rel = r.a_equip - r.b_equip
    r["upset"] = np.where(r.won.isna(), 0.0, np.where((r.won == 1) & (rel < -3000), 1.0, np.where((r.won == 0) & (rel > 3000), -1.0, 0.0)))
    gun = (r.a_equip > 15000) & (r.b_equip > 15000) & r.won.notna()
    r["gun"] = gun.astype(float)
    r["gun_win"] = np.where(gun, r.won, 0.0)
    r["kd"] = kd.reindex(pd.MultiIndex.from_frame(r[["game_id", "round"]])).fillna(0.0).to_numpy()
    r["has"] = r.won.notna().astype(float)
    c = r.groupby("game_id")[["kd", "upset", "gun", "gun_win", "has"]].cumsum() - r[["kd", "upset", "gun", "gun_win", "has"]]
    r = r.assign(a_kd_prev=c.kd, a_upsets_prev=c.upset, rounds_prev=c.has, gun_prev=c.gun,
                 a_gun_wr_prev=(c.gun_win + 1) / (c.gun + 2))
    out = keys[["game_id", "round"]].merge(r[["game_id", "round"] + FORM], on=["game_id", "round"], how="left")
    return out[FORM].set_axis(keys.index)


SERIES = ["s_maps", "s_w", "s_rd"]


def series_features(b: Bundle, game_ids) -> pd.DataFrame:
    """Earlier maps of the same series that ended before each map began: maps played, team A's maps won minus lost,
    and rounds won minus lost. The final score is read from the last round's score: the map ends on the round that
    gives one team its last needed round, so that team won it (outcome columns are not needed)."""
    g = b.games[["game_id", "match_id", "begin", "end"]]
    r = b.rounds.sort_values(["game_id", "round"]).groupby("game_id").tail(1)
    nd = np.array([need(a, c) for a, c in zip(r.score_a, r.score_b)]).reshape(-1, 2)
    a_last = np.where((nd[:, 0] == 1) & (nd[:, 1] > 1), 1.0, np.where((nd[:, 1] == 1) & (nd[:, 0] > 1), 0.0, np.nan))
    fin = pd.DataFrame(dict(game_id=r.game_id.to_numpy(), rd=(r.score_a - r.score_b).to_numpy() + 2 * a_last - 1,
                            w=2 * a_last - 1))
    done = g.merge(fin, on="game_id")
    tgt = g[g.game_id.isin(set(game_ids))]
    pairs = tgt.merge(done, on="match_id", suffixes=("", "_p"))
    pairs = pairs[(pairs.end_p < pairs.begin) & (pairs.game_id_p != pairs.game_id)]
    agg = pairs.groupby("game_id").agg(s_maps=("w", "size"), s_w=("w", "sum"), s_rd=("rd", "sum"))
    return agg.reindex(tgt.game_id).fillna(0.0).rename_axis("game_id").reset_index()


TEAM_MAP = ["a_map_rwr", "b_map_rwr", "a_all_rwr", "b_all_rwr", "a_map_n", "b_map_n", "a_ct_map", "a_t_map",
            "b_ct_map", "b_t_map"]


def round_results(b: Bundle) -> pd.DataFrame:
    """Every round's winner from the score progression alone (the map's last round goes to the team it gives its last
    needed round), so maps whose outcome columns are blanked still count."""
    r = b.rounds.sort_values(["game_id", "round"])[["game_id", "round", "score_a", "score_b", "side_a"]]
    nxt = r.groupby("game_id").score_a.shift(-1)
    nr = r.groupby("game_id")["round"].shift(-1)
    won = np.where(nr == r["round"] + 1, nxt - r.score_a, np.nan)
    nd = np.array([need(a, c) for a, c in zip(r.score_a, r.score_b)]).reshape(-1, 2)
    last = nr.isna().to_numpy()
    fin = np.where((nd[:, 0] == 1) & (nd[:, 1] > 1), 1.0, np.where((nd[:, 1] == 1) & (nd[:, 0] > 1), 0.0, np.nan))
    return r.assign(a_won=np.where(last, fin, won))


def team_map_features(b: Bundle, game_ids) -> pd.DataFrame:
    """Each team's round-win rate on this map, on each side of it, and on all maps, over its maps that ended before
    this map began (shrunk toward 0.5), and how many maps on this map that is."""
    rr = round_results(b)
    rr = rr[rr.a_won.notna()]
    g = b.games[["game_id", "team_a", "team_b", "begin", "end"]]
    mp = b.rounds.drop_duplicates("game_id")[["game_id", "map"]]
    rr = rr.merge(g, on="game_id").merge(mp, on="game_id")
    recs = []
    for t, won, ct in (("team_a", rr.a_won, rr.side_a == "CT"), ("team_b", 1 - rr.a_won, rr.side_a != "CT")):
        recs.append(pd.DataFrame(dict(team=rr[t].to_numpy(), map=rr["map"].to_numpy(), end=rr.end.to_numpy(),
                                      game_id=rr.game_id.to_numpy(), w=won.to_numpy(), n=1.0,
                                      ctw=np.where(ct, won, 0.0), ctn=ct.astype(float).to_numpy(),
                                      tw=np.where(~ct, won, 0.0), tn=(~ct).astype(float).to_numpy())))
    per = pd.concat(recs).groupby(["team", "map", "game_id", "end"], as_index=False)[["w", "n", "ctw", "ctn", "tw", "tn"]].sum()
    per["maps"] = 1.0
    per = per.sort_values("end")
    vals = ["w", "n", "ctw", "ctn", "tw", "tn", "maps"]
    cm = per.groupby(["team", "map"])[vals].cumsum().assign(team=per.team, map=per["map"], end=per.end)
    allm = per.groupby(["team", "end"], as_index=False)[["w", "n"]].sum().sort_values("end")
    ca = allm.groupby("team")[["w", "n"]].cumsum().assign(team=allm.team, end=allm.end)
    tgt = g[g.game_id.isin(set(game_ids))].merge(mp, on="game_id").sort_values("begin")
    out = tgt[["game_id"]].copy()
    k = 20.0
    for s in ("a", "b"):
        q = tgt[["game_id", "begin", "map", f"team_{s}"]].rename(columns={f"team_{s}": "team"})
        x = pd.merge_asof(q, cm.sort_values("end"), left_on="begin", right_on="end", by=["team", "map"],
                          allow_exact_matches=False).fillna(0.0)
        y = pd.merge_asof(q, ca.sort_values("end"), left_on="begin", right_on="end", by="team",
                          allow_exact_matches=False).fillna(0.0)
        out[f"{s}_map_rwr"] = ((x.w + k / 2) / (x.n + k)).to_numpy()
        out[f"{s}_all_rwr"] = ((y.w + k / 2) / (y.n + k)).to_numpy()
        out[f"{s}_map_n"] = x["maps"].to_numpy()
        out[f"{s}_ct_map"] = ((x.ctw + k / 4) / (x.ctn + k / 2)).to_numpy()
        out[f"{s}_t_map"] = ((x.tw + k / 4) / (x.tn + k / 2)).to_numpy()
    return out


def round_elo(b: Bundle, game_ids, k: float = 24.0, scale: float = 400.0) -> pd.DataFrame:
    """A team rating from round shares: after each map, both teams move by k x (round share - expected share), the
    expected share a logistic in the rating gap. Each map is priced with the ratings of maps that ended before it
    began. Returns the gap for team A in logit units of a single round."""
    rr = round_results(b)
    sh = rr.groupby("game_id").agg(w=("a_won", "sum"), n=("a_won", "count"))
    g = b.games[["game_id", "team_a", "team_b", "begin", "end"]].merge(sh, on="game_id", how="left")
    want = set(game_ids)
    ev = pd.concat([g.assign(t=g.begin, kind=0), g[g.n > 0].assign(t=g.end, kind=1)]).sort_values(["t", "kind"])
    rating = {}
    gap = {}
    ln = np.log(10.0) / scale
    for gid, ta, tb, kind, w, n in zip(ev.game_id, ev.team_a, ev.team_b, ev.kind, ev.w, ev.n):
        ra, rb = rating.get(ta, 0.0), rating.get(tb, 0.0)
        if kind == 0:
            if gid in want:
                gap[gid] = (ra - rb) * ln
            continue
        e = 1.0 / (1.0 + np.exp(-(ra - rb) * ln))
        d = k * (w / n - e) * np.sqrt(n)
        rating[ta] = ra + d
        rating[tb] = rb - d
    return pd.DataFrame(dict(game_id=list(gap), relo=list(gap.values())))


def player_ratings(b: Bundle, game_ids, k: float = 60.0) -> pd.DataFrame:
    """Each player's (kills - deaths) per round over maps that ended before each map began, shrunk toward 0 with k
    pseudo-rounds. Kills and deaths come from the timeline where a map has one, else from the end-of-round player
    rows (only maps without a timeline, all of them history)."""
    g = b.games[["game_id", "begin", "end"]]
    ev = b.events[b.events.type == "kill"]
    k_ev = ev.groupby(["game_id", "killer_id"]).size().rename("k").rename_axis(["game_id", "player_id"])
    d_ev = ev.groupby(["game_id", "victim_id"]).size().rename("d").rename_axis(["game_id", "player_id"])
    P = b.players[["game_id", "round", "player_id", "kills", "alive_end"]]
    n = P.groupby(["game_id", "player_id"]).size().rename("n")
    has_ev = set(ev.game_id)
    P2 = P[~P.game_id.isin(has_ev) & P.kills.notna() & P.alive_end.notna()]
    k_pl = P2.groupby(["game_id", "player_id"]).kills.sum().rename("k")
    d_pl = (~P2.alive_end.astype(bool)).groupby([P2.game_id, P2.player_id]).sum().rename("d")
    kd = pd.concat([pd.concat([k_ev, k_pl]), pd.concat([d_ev, d_pl])], axis=1).fillna(0.0)
    per = pd.concat([kd, n], axis=1).dropna(subset=["n"]).fillna(0.0).reset_index()
    per = per[per.player_id.notna()].astype({"player_id": "int64"})
    per = per[per.game_id.isin(set(g.game_id[g.game_id.isin(has_ev)]) | set(P2.game_id))]
    per = per.merge(g[["game_id", "end"]], on="game_id").sort_values("end")
    c = per.groupby("player_id")[["k", "d", "n"]].cumsum().assign(player_id=per.player_id, end=per.end)
    tgt = P[P.game_id.isin(set(game_ids))][["game_id", "player_id"]].drop_duplicates().merge(g[["game_id", "begin"]], on="game_id")
    tgt = tgt.astype({"player_id": "int64"}).sort_values("begin")
    x = pd.merge_asof(tgt, c.sort_values("end"), left_on="begin", right_on="end", by="player_id",
                      allow_exact_matches=False).fillna({"k": 0.0, "d": 0.0, "n": 0.0})
    return x.assign(rating=(x.k - x.d) / (x.n + k))[["game_id", "player_id", "rating"]]
