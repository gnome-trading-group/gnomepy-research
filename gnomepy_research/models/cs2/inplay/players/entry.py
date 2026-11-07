"""Benchmark round + map models with point-in-time player and team strength.

Player strength (kills, deaths, opening duels, clutches, AWP use) and team strength (round win rates by side and map)
come from maps that ended strictly before each map began (skill.py). The round model sees the strength of the players
still alive on each side, not just their count; the map model sees the two rosters' strength and each team's side
record on the map. Strength is frozen per map at its start, so pricing a state never depends on anything inside the
map beyond what the benchmark already uses."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, logit, next_round, trees, with_prior
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table, state_features
from gnomepy_research.models.cs2.inplay.players import skill as S

CFG = dict(
    a_rate=float(os.environ.get("PL_A_RATE", 40)),
    a_open=float(os.environ.get("PL_A_OPEN", 15)),
    a_clutch=float(os.environ.get("PL_A_CLUTCH", 8)),
    a_team=float(os.environ.get("PL_A_TEAM", 30)),
    half_life=float(os.environ.get("PL_HALF_LIFE", 0)) or None,
    round_feats=os.environ.get("PL_ROUND", "1") == "1",
    map_feats=os.environ.get("PL_MAP", "1") == "1",
)

PFEATS = ["net", "kpr", "dpr", "okr", "opr", "clr", "awr", "lgn"]
TCOUNTS = ["ct_n", "ct_w", "t_n", "t_w"]
ROUND_X = ["ct_net_alive", "t_net_alive", "ct_net_max", "t_net_max", "ct_net_start", "t_net_start", "ct_okr", "t_okr",
           "ct_clutch", "t_clutch", "ct_best_dead", "t_best_dead", "ct_awper_dead", "t_awper_dead", "ct_lgn", "t_lgn",
           "ct_side_map", "t_side_map", "ct_team_all", "t_team_all"]
MAP_X = ["d_net_roster", "a_net_roster", "b_net_roster", "a_ct_map", "a_t_map", "b_ct_map", "b_t_map", "a_team_all", "b_team_all", "lgn_roster"]


class History:
    """Per-map counts of the training maps plus their running totals, so pricing only adds the scored month's maps."""

    def __init__(self, b: Bundle, hl):
        self.hl = hl
        ended = b.games.game_id[b.games.end.notna()]
        self.gids = set(ended)
        self.p_cum, self.t_cum, self.ta_cum = self._cums(b, ended)

    def _cums(self, b, gids):
        pc = S.stamp(S.player_map_counts(b, gids), b.games)
        self.totals = pc[S.PCOUNTS].sum()
        tc = S.stamp(S.team_map_counts(b, gids), b.games)
        ta = tc.groupby(["game_id", "team_id", "time"])[TCOUNTS].sum().reset_index()
        return (S.cumulative(pc, ["player_id"], S.PCOUNTS, self.hl), S.cumulative(tc, ["team_id", "map"], TCOUNTS, self.hl),
                S.cumulative(ta, ["team_id"], TCOUNTS, self.hl))

    def lookup(self, b: Bundle, players_t: pd.DataFrame, teams_t: pd.DataFrame):
        """players_t: player_id, time; teams_t: team_id, map, time. Totals from maps that ended before `time`."""
        new = b.games.game_id[b.games.end.notna() & ~b.games.game_id.isin(self.gids)]
        tables = [(self.p_cum, self.t_cum, self.ta_cum)]
        if len(new):
            tables.append(self._cums(b, new))
        p = sum(S.asof(players_t, t[0], ["player_id"], S.PCOUNTS, self.hl) for t in tables)
        tm = sum(S.asof(teams_t, t[1], ["team_id", "map"], TCOUNTS, self.hl) for t in tables)
        ta = sum(S.asof(teams_t, t[2], ["team_id"], TCOUNTS, self.hl) for t in tables)
        return p, tm, ta


def player_rates(c: pd.DataFrame, pri: dict) -> pd.DataFrame:
    a, ao, ac = CFG["a_rate"], CFG["a_open"], CFG["a_clutch"]
    kpr = (c.k + a * pri["kpr"]) / (c.n + a)
    dpr = (c.d + a * pri["dpr"]) / (c.n + a)
    duels = c.ok + c.od
    return pd.DataFrame(dict(
        kpr=kpr, dpr=dpr, net=kpr - dpr,
        okr=(c.ok + ao * 0.5) / (duels + ao),
        opr=(duels + a * pri["opr"]) / (c.n_ev + a),
        clr=(c.cw + ac * pri["clr"]) / (c.ca + ac),
        awr=(c.awr + a * pri["awr"]) / (c.n + a),
        lgn=np.log1p(c.n)), index=c.index)


def priors_from(h: History) -> dict:
    s = h.totals
    return dict(kpr=s.k / s.n, dpr=s.d / s.n, opr=(s.ok + s.od) / max(s.n_ev, 1), clr=s.cw / max(s.ca, 1), awr=s.awr / s.n)


def strength(model, b: Bundle, game_ids) -> tuple[dict, pd.DataFrame]:
    """Per-map player ratings {(game_id, player_id): rates} and per-map team table, as of each map's begin."""
    gids = set(game_ids)
    g = b.games[b.games.game_id.isin(gids)][["game_id", "begin", "team_a", "team_b"]]
    P = b.players[b.players.game_id.isin(gids)][["game_id", "player_id"]].drop_duplicates()
    pt = P.merge(g[["game_id", "begin"]], on="game_id").rename(columns={"begin": "time"})
    mp = b.rounds[b.rounds.game_id.isin(gids)].groupby("game_id")["map"].first()
    tt = pd.concat([g[["game_id", "team_a", "begin"]].rename(columns={"team_a": "team_id"}),
                    g[["game_id", "team_b", "begin"]].rename(columns={"team_b": "team_id"})], ignore_index=True)
    tt = tt.assign(map=tt.game_id.map(mp), time=tt.begin)
    tt = tt[tt["map"].notna()].reset_index(drop=True)
    pc, tm, ta = model["hist"].lookup(b, pt, tt)
    rates = player_rates(pc, model["pri"])
    rates.index = pd.MultiIndex.from_frame(pt[["game_id", "player_id"]])
    at = CFG["a_team"]
    gm = tt["map"].map(model["map_ct"]).fillna(0.5).to_numpy()
    team = tt[["game_id", "team_id"]].assign(
        ct_map=(tm.ct_w + at * gm) / (tm.ct_n + at), t_map=(tm.t_w + at * (1 - gm)) / (tm.t_n + at),
        all_=(ta.ct_w + ta.t_w + 2 * at * 0.5) / (ta.ct_n + ta.t_n + 2 * at))
    return rates, team


def round_skill(b: Bundle, states: pd.DataFrame, rates: pd.DataFrame, team: pd.DataFrame) -> pd.DataFrame:
    P = b.players[b.players.game_id.isin(set(states.game_id))]
    rd = rates.to_dict("index")
    roster = {k: list(zip(x.player_id, x.side, x.team_id)) for k, x in P.groupby(["game_id", "round"])}
    ev = b.events[b.events.game_id.isin(set(states.game_id)) & (b.events.type == "kill")]
    deaths = {k: list(zip(x.seq, x.victim_id)) for k, x in ev.sort_values("seq").groupby(["game_id", "round"])}
    tix = team.set_index(["game_id", "team_id"])
    tmap = {k: v for k, v in zip(tix.index, tix[["ct_map", "t_map", "all_"]].to_numpy())}
    out = np.full((len(states), len(ROUND_X)), np.nan)
    cache = {}
    for i, (gid, rnd, seq) in enumerate(zip(states.game_id.to_numpy(), states["round"].to_numpy(), states.event_seq.to_numpy())):
        key = (gid, rnd)
        if key not in cache:
            side = {"CT": [], "T": []}
            tid = {}
            for pid, sd, t in roster.get(key, []):
                r = rd.get((gid, pid))
                if r is None:
                    continue
                side[sd].append((pid, r["net"], r["okr"], r["clr"], r["awr"], r["lgn"]))
                tid[sd] = t
            cache[key] = (side, tid)
        side, tid = cache[key]
        dead = {v for s, v in deaths.get(key, []) if s <= seq}
        row = []
        stats = {}
        for sd in ("CT", "T"):
            allp = side[sd]
            alive = [p for p in allp if p[0] not in dead]
            net_alive = sum(p[1] for p in alive)
            net_max = max((p[1] for p in alive), default=np.nan)
            net_start = sum(p[1] for p in allp) if allp else np.nan
            okr = np.mean([p[2] for p in alive]) if alive else np.nan
            clutch = alive[0][3] if len(alive) == 1 else np.nan
            best = max(allp, key=lambda p: p[1])[0] if allp else None
            awper = max(allp, key=lambda p: p[4]) if allp else None
            best_dead = float(best in dead) if best is not None else np.nan
            awper_dead = float(awper[0] in dead) if awper is not None and awper[4] > 0.25 else np.nan
            lgn = np.mean([p[5] for p in allp]) if allp else np.nan
            stats[sd] = (net_alive, net_max, net_start, okr, clutch, best_dead, awper_dead, lgn)
        tct = tmap.get((gid, tid.get("CT")), (np.nan,) * 3)
        tt = tmap.get((gid, tid.get("T")), (np.nan,) * 3)
        c, t = stats["CT"], stats["T"]
        out[i] = [c[0], t[0], c[1], t[1], c[2], t[2], c[3], t[3], c[4], t[4], c[5], t[5], c[6], t[6], c[7], t[7],
                  tct[0], tt[1], tct[2], tt[2]]
    return pd.DataFrame(out, columns=ROUND_X, index=states.index)


def map_skill(b: Bundle, rt: pd.DataFrame, rates: pd.DataFrame, team: pd.DataFrame) -> pd.DataFrame:
    """Roster strength of each team in each round (its players' net rating summed) and each team's side record."""
    P = b.players[b.players.game_id.isin(set(rt.game_id))][["game_id", "round", "player_id", "team_id"]]
    P = P.join(rates[["net", "lgn"]], on=["game_id", "player_id"])
    g = b.games.set_index("game_id")
    P = P.assign(is_a=P.team_id.to_numpy() == g.team_a.reindex(P.game_id).to_numpy())
    agg = P.groupby(["game_id", "round", "is_a"]).net.sum().unstack()
    lg = P.groupby(["game_id", "round"]).lgn.mean()
    x = rt[["game_id", "round"]].copy()
    k = pd.MultiIndex.from_frame(x)
    x["a_net_roster"] = agg.get(True, pd.Series(dtype=float)).reindex(k).to_numpy()
    x["b_net_roster"] = agg.get(False, pd.Series(dtype=float)).reindex(k).to_numpy()
    x["d_net_roster"] = x.a_net_roster - x.b_net_roster
    x["lgn_roster"] = lg.reindex(k).to_numpy()
    tix = team.set_index(["game_id", "team_id"])
    for t in ("a", "b"):
        tk = pd.MultiIndex.from_arrays([x.game_id.to_numpy(), g[f"team_{t}"].reindex(x.game_id).to_numpy()])
        v = tix.reindex(tk)
        x[f"{t}_ct_map"], x[f"{t}_t_map"], x[f"{t}_team_all"] = v.ct_map.to_numpy(), v.t_map.to_numpy(), v.all_.to_numpy()
    return x[MAP_X].set_axis(rt.index)


def round_cols():
    return ROUND + (ROUND_X if CFG["round_feats"] else [])


def map_cols():
    return MAPCOLS + (MAP_X if CFG["map_feats"] else [])


def rows(model, b: Bundle, states: pd.DataFrame):
    rt = with_prior(round_table(b.subset(states.game_id)))
    rates, team = strength(model, b, rt.game_id.unique())
    rt = pd.concat([rt, map_skill(b, rt, rates, team)], axis=1).set_index(["game_id", "round"])
    from_bench = states.join(state_features(b, states))
    x = from_bench.join(rt[["side_a", "a_equip", "b_equip", "need_a", "need_b", "map_code"]], on=["game_id", "round"])
    a_ct = x.side_a == "CT"
    x["ct_eq_start"] = np.where(a_ct, x.a_equip, x.b_equip); x["t_eq_start"] = np.where(a_ct, x.b_equip, x.a_equip)
    x["ct_need"] = np.where(a_ct, x.need_a, x.need_b); x["t_need"] = np.where(a_ct, x.need_b, x.need_a)
    x = x.join(round_skill(b, states, rates, team))
    return x, rt


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    hist = History(train, CFG["half_life"])
    model = dict(hist=hist, pri=priors_from(hist))
    r = train.rounds[train.rounds.a_wins.notna()]
    ct_won = np.where(r.side_a == "CT", r.a_wins, 1 - r.a_wins)
    model["map_ct"] = pd.Series(ct_won).groupby(r["map"].to_numpy()).mean().to_dict()
    legal = legal_maps(train)
    sub = train.subset(legal)
    rt = with_prior(round_table(sub))
    rates, team = strength(model, sub, rt.game_id.unique())
    rt = pd.concat([rt, map_skill(sub, rt, rates, team)], axis=1)
    rt = rt[rt.map_a_wins.notna()]
    model["map_model"] = trees(seed).fit(rt[map_cols()], rt.map_a_wins)
    st = map_states(train, legal)
    x, rti = rows(model, train, st)
    x = x.join(rti[["a_wins"]], on=["game_id", "round"])
    y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    model["round_model"] = trees(seed, max_depth=6, learning_rate=0.05).fit(x[round_cols()], y)
    return model


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    x, rt = rows(model, ctx, states)
    p_ct = model["round_model"].predict_proba(x[round_cols()])[:, 1]
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rr = rt.loc[keys.unique()].reset_index()
    m = {}
    for a_won in (1, 0):
        X, done = next_round(rr, a_won)
        X = pd.concat([X, rr[MAP_X]], axis=1)
        p = model["map_model"].predict_proba(X[map_cols()])[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
    p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()
