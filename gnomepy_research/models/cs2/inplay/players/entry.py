"""Benchmark round + map models with point-in-time player and team strength.

Player strength (kills, deaths, opening duels, clutches, AWP use) and team strength (round win rates by side and map)
come from maps that ended strictly before each map began (skill.py). The round model sees the strength of the players
still alive on each side, not just their count; the map model sees the two rosters' strength and each team's side
record on the map. Strength is frozen per map at its start, so pricing a state never depends on anything inside the
map beyond what the benchmark already uses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, logit, next_round, trees, with_prior
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table, state_features
from gnomepy_research.models.cs2.inplay.players import skill as S
from gnomepy_research.models.cs2.inplay.players.dp import dp_value

CFG = dict(
    a_rate=40,
    a_open=15,
    a_clutch=8,
    a_team=10,
    half_life=None,
    elo_k=0.3,
    elo_k_map=0.1,
    pm="full2",
    round_feats=False,
    round_prior=False,
    map_x=[],
    map_prior="roster",
    stack="memap",
    map_nohas=True,
    map_mono=True,
    dp=False,
    map_bag=1,
    map_kw=dict(max_depth=6, min_child_weight=300, learning_rate=0.05),
)

PFEATS = ["net", "kpr", "dpr", "okr", "opr", "clr", "awr", "lgn"]
TCOUNTS = ["ct_n", "ct_w", "t_n", "t_w"]
ROUND_X = ["ct_net_alive", "t_net_alive", "ct_net_max", "t_net_max", "ct_net_start", "t_net_start", "ct_okr", "t_okr",
           "ct_clutch", "t_clutch", "ct_best_dead", "t_best_dead", "ct_awper_dead", "t_awper_dead", "ct_lgn", "t_lgn",
           "ct_side_map", "t_side_map", "ct_team_all", "t_team_all"]
MAP_X = ["d_net_roster", "a_net_roster", "b_net_roster", "a_ct_map", "a_t_map", "b_ct_map", "b_t_map", "a_team_all", "b_team_all", "lgn_roster",
         "side_adv_ct", "side_adv_t", "kd_rate"]


class History:
    """Per-map counts of the training maps plus their running totals, so pricing only adds the scored month's maps."""

    def __init__(self, b: Bundle, hl, until: pd.Timestamp):
        self.hl = hl
        ended = b.games.game_id[b.games.end.notna()]
        self.gids = set(ended)
        self.cums = self._cums(b, ended)
        res = S.map_results(b, ended)
        self.elo = S.PlayerElo(CFG["elo_k"], CFG["elo_k_map"])
        self.pelo = self.elo.run(res, res).set_axis(res.game_id)
        self.apm = S.PlusMinus(res, until) if "d_apm" in pm_cols() else None

    def player_elo(self, b: Bundle, game_ids) -> pd.DataFrame:
        """Player-Elo roster means as of each map's begin: stored for training maps; for others the ratings at the end
        of training are carried forward through the maps that have ended since (results read from their timelines)."""
        gids = pd.Index(sorted(set(game_ids)))
        known = gids[gids.isin(self.pelo.index)]
        out = [self.pelo.loc[known]]
        rest = gids[~gids.isin(self.pelo.index)]
        if len(rest):
            new = b.games.game_id[b.games.end.notna() & ~b.games.game_id.isin(self.gids)]
            tg = S.map_results(b, rest)
            out.append(self.elo.copy().run(S.map_results(b, new), tg).set_axis(tg.game_id))
        return pd.concat(out).reindex(gids)

    def _cums(self, b, gids):
        pc = S.stamp(S.player_map_counts(b, gids), b.games)
        self.totals = pc[S.PCOUNTS].sum()
        tc = S.stamp(S.team_map_counts(b, gids), b.games)
        ta = tc.groupby(["game_id", "team_id", "time"])[TCOUNTS].sum().reset_index()
        return (S.cumulative(pc, ["player_id"], S.PCOUNTS, self.hl), S.cumulative(tc, ["team_id", "map"], TCOUNTS, self.hl),
                S.cumulative(ta, ["team_id"], TCOUNTS, self.hl), S.cumulative(pc.assign(maps=1.0), ["player_id", "team_id"], ["maps"]))

    def lookup(self, b: Bundle, players_t: pd.DataFrame, teams_t: pd.DataFrame):
        """players_t: player_id, team_id, time; teams_t: team_id, map, time. Totals from maps that ended before `time`;
        the last is the number of maps each player has played for this team."""
        new = b.games.game_id[b.games.end.notna() & ~b.games.game_id.isin(self.gids)]
        tables = [self.cums]
        if len(new):
            tables.append(self._cums(b, new))
        p = sum(S.asof(players_t, t[0], ["player_id"], S.PCOUNTS, self.hl) for t in tables)
        tm = sum(S.asof(teams_t, t[1], ["team_id", "map"], TCOUNTS, self.hl) for t in tables)
        ta = sum(S.asof(teams_t, t[2], ["team_id"], TCOUNTS, self.hl) for t in tables)
        pt = sum(S.asof(players_t, t[3], ["player_id", "team_id"], ["maps"]) for t in tables)
        return p, tm, ta, pt


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
    P = b.players[b.players.game_id.isin(gids)][["game_id", "player_id", "team_id"]].drop_duplicates(["game_id", "player_id"])
    pt = P.merge(g[["game_id", "begin"]], on="game_id").rename(columns={"begin": "time"})
    mp = b.rounds[b.rounds.game_id.isin(gids)].groupby("game_id")["map"].first()
    tt = pd.concat([g[["game_id", "team_a", "begin"]].rename(columns={"team_a": "team_id"}),
                    g[["game_id", "team_b", "begin"]].rename(columns={"team_b": "team_id"})], ignore_index=True)
    tt = tt.assign(map=tt.game_id.map(mp), time=tt.begin)
    tt = tt[tt["map"].notna()].reset_index(drop=True)
    pc, tm, ta, ptm = model["hist"].lookup(b, pt, tt)
    rates = player_rates(pc, model["pri"]).assign(lstab=np.log1p(ptm.maps.to_numpy()))
    rates.index = pd.MultiIndex.from_frame(pt[["game_id", "player_id"]])
    at = CFG["a_team"]
    gm = tt["map"].map(model["map_ct"]).fillna(0.5).to_numpy()
    team = tt[["game_id", "team_id"]].assign(
        ct_map=(tm.ct_w + at * gm) / (tm.ct_n + at), t_map=(tm.t_w + at * (1 - gm)) / (tm.t_n + at),
        all_=(ta.ct_w + ta.t_w + 2 * at * 0.5) / (ta.ct_n + ta.t_n + 2 * at))
    team["map_all"] = (tm.ct_w + tm.t_w + at * team.all_) / (tm.ct_n + tm.t_n + at)
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


def kill_diff(b: Bundle, x: pd.DataFrame) -> np.ndarray:
    """Team A's kills minus team B's over the map's completed rounds, per player-round: in-map evidence of strength
    beyond the score (a 13-5 with many survivors is not a scrappy 13-5). Only rounds before each row's round count,
    and only maps with a timeline (NaN otherwise)."""
    ev = b.events[b.events.game_id.isin(set(x.game_id)) & b.events.type.isin(["kill", "round_start"])]
    ta = b.games.set_index("game_id").team_a
    k = ev[ev.type == "kill"]
    a_dead = (k.victim_team.to_numpy(dtype=float) == ta.reindex(k.game_id).to_numpy())
    per = pd.DataFrame(dict(game_id=k.game_id.to_numpy(), round=k["round"].to_numpy(), d=np.where(a_dead, -1.0, 1.0)))
    per = per.groupby(["game_id", "round"]).d.sum()
    started = ev[ev.type == "round_start"][["game_id", "round"]].drop_duplicates()
    r = started.join(per, on=["game_id", "round"]).fillna({"d": 0.0}).sort_values(["game_id", "round"])
    r["cum"] = r.groupby("game_id").d.cumsum() - r.d
    r["n"] = r.groupby("game_id").cumcount()
    r["rate"] = np.where(r.n > 0, r.cum / (5.0 * r.n.clip(lower=1)), np.nan)
    return r.set_index(["game_id", "round"]).rate.reindex(pd.MultiIndex.from_frame(x[["game_id", "round"]])).to_numpy()


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
    x["kd_rate"] = kill_diff(b, x)
    x["side_adv_ct"] = logit(x.a_ct_map) - logit(x.b_t_map)
    x["side_adv_t"] = logit(x.a_t_map) - logit(x.b_ct_map)
    return x[MAP_X].set_axis(rt.index)


PM0 = ["elo_logit", "d_net_mean", "d_net_max", "d_net_min", "d_okr_mean", "d_clr_mean", "d_kpr_mean", "d_dpr_mean", "d_all", "d_lgn_mean"]


PM_SETS = {"base": [], "pelo": ["d_pelo", "d_pn"], "full": ["d_pelo", "d_pn", "d_pmap", "d_share"], "share": ["d_share"],
           "full2": ["d_pelo", "d_pn", "d_pmap", "d_share", "d_lstab_mean", "d_lstab_min", "d_teammap"],
           "full_stab": ["d_pelo", "d_pn", "d_pmap", "d_share", "d_lstab_mean", "d_lstab_min"],
           "full_tm": ["d_pelo", "d_pn", "d_pmap", "d_share", "d_teammap"],
           "full3": ["d_pelo", "d_pn", "d_pmap", "d_share", "d_lstab_mean", "d_lstab_min", "d_teammap", "d_apm"]}


def pm_cols():
    return PM0 + PM_SETS[CFG["pm"]]


def premap(model, b: Bundle, game_ids, rates: pd.DataFrame, team: pd.DataFrame) -> pd.DataFrame:
    """Per map: team A minus team B over the roster of the map's first round with player rows (mean/max/min net rating,
    opening, clutch, kills, deaths, experience) and overall round win rate. A scored state's round always has player
    rows, so its map's first such round is never after it."""
    gids = set(game_ids)
    P = b.players[b.players.game_id.isin(gids)][["game_id", "round", "player_id", "team_id"]]
    first = P.groupby("game_id")["round"].transform("min")
    P = P[P["round"] == first].join(rates, on=["game_id", "player_id"])
    g = b.games.set_index("game_id")
    P = P.assign(is_a=P.team_id.to_numpy() == g.team_a.reindex(P.game_id).to_numpy())
    agg = P.groupby(["game_id", "is_a"])[["net", "okr", "clr", "kpr", "dpr", "lgn", "lstab"]].agg(["mean", "max", "min"])
    agg.columns = [f"{a}_{c}" for a, c in agg.columns]
    agg = agg.unstack("is_a")
    out = pd.DataFrame(index=sorted(gids))
    out.index.name = "game_id"
    for c in ["net_mean", "net_max", "net_min", "okr_mean", "clr_mean", "kpr_mean", "dpr_mean", "lgn_mean", "lstab_mean", "lstab_min"]:
        a = agg.get((c, True), pd.Series(dtype=float)).reindex(out.index)
        bb = agg.get((c, False), pd.Series(dtype=float)).reindex(out.index)
        out[f"d_{c}"] = a - bb
    tix = team.set_index(["game_id", "team_id"]).all_
    za = logit(tix.reindex(pd.MultiIndex.from_arrays([out.index, g.team_a.reindex(out.index)])).to_numpy())
    zb = logit(tix.reindex(pd.MultiIndex.from_arrays([out.index, g.team_b.reindex(out.index)])).to_numpy())
    out["d_all"] = za - zb
    tmall = team.set_index(["game_id", "team_id"]).map_all
    if model["hist"].apm is not None:
        out["d_apm"] = model["hist"].apm.rate(S.map_results(b, out.index).set_index("game_id")).reindex(out.index)
    out["d_teammap"] = (logit(tmall.reindex(pd.MultiIndex.from_arrays([out.index, g.team_a.reindex(out.index)])).to_numpy())
                        - logit(tmall.reindex(pd.MultiIndex.from_arrays([out.index, g.team_b.reindex(out.index)])).to_numpy()))
    out["elo_logit"] = b.elo.set_index("game_id").elo_logit.reindex(out.index).fillna(0.0).to_numpy()
    pe = model["hist"].player_elo(b, out.index).reindex(out.index)
    out["d_pelo"] = pe.pelo_a - pe.pelo_b
    out["d_pn"] = np.log1p(pe.pelo_na) - np.log1p(pe.pelo_nb)
    out["d_pmap"] = pe.pmap_a - pe.pmap_b
    out["d_share"] = np.log(pe.pshare_a) - np.log(pe.pshare_b)
    return out


def fit_premap(pm: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    cols = pm_cols()
    ok = pm[cols].notna().all(axis=1) & y.reindex(pm.index).notna()
    return LogisticRegression(C=1.0, max_iter=2000).fit(pm.loc[ok, cols], y.reindex(pm.index)[ok])


def premap_z(pm: pd.DataFrame, lr) -> pd.Series:
    x = pm[pm_cols()]
    ok = x.notna().all(axis=1)
    z = pd.Series(np.nan, index=pm.index)
    if ok.any():
        z[ok] = lr.decision_function(x[ok])
    return z


STACK = {"mz": ["m", "z"], "mez": ["m", "e", "z"], "me": ["m", "e"], "memap": ["m", "e", "pmap", "share", "teammap"]}


def stack_x(d: pd.DataFrame, z, pm: pd.DataFrame) -> pd.DataFrame:
    """Stack inputs per row of d (rows with a market price): the market and e12 logits, the roster prior z, and the
    map-specific roster terms, which a series price cannot know (which map is played says whose pick it is)."""
    p = pm.reindex(d.game_id)
    x = pd.DataFrame(dict(m=logit(d.p_market), e=logit(d.p_e12), z=np.asarray(z, float), pmap=p.d_pmap.to_numpy(),
                          share=p.d_share.to_numpy(), teammap=p.d_teammap.to_numpy()), index=d.index)
    return x[STACK[CFG["stack"]]]


def fit_stack(train: Bundle, z: pd.Series, pm: pd.DataFrame):
    """The kickoff market (a series price turned into a map price, underconfident on its own) combined with the
    roster prior; needs a few hundred market-listed maps to learn from, else the market is used as is."""
    if CFG["stack"] not in STACK:
        return None
    d = train.priors.merge(train.games[["game_id", "map_a_wins"]], on="game_id")
    d = d.assign(z=z.reindex(d.game_id).to_numpy()).dropna(subset=["p_market", "p_e12", "map_a_wins", "z"])
    x = stack_x(d, d.z, pm)
    ok = x.notna().all(axis=1).to_numpy()
    if ok.sum() < 300:
        return None
    return LogisticRegression(C=1.0).fit(x[ok], d.map_a_wins[ok])


def with_prior2(rt: pd.DataFrame, pm: pd.DataFrame, lr, stack=None) -> pd.DataFrame:
    """Benchmark prior (kickoff market, else Elo) with Elo replaced by the roster-strength prior where the map has one,
    and the market stacked with the roster prior once there is enough market history."""
    rt = with_prior(rt)
    if CFG["map_prior"] != "roster":
        return rt
    zz = premap_z(pm, lr).reindex(rt.game_id).to_numpy()
    if stack is not None:
        x = stack_x(rt, zz, pm)
        ok = (rt.p_market.notna() & rt.p_e12.notna()).to_numpy() & ~np.isnan(zz) & x.notna().all(axis=1).to_numpy()
        if ok.any():
            z = rt.prior_logit.to_numpy().copy()
            z[ok] = stack.decision_function(x[ok])
            rt = rt.assign(prior_logit=z)
    use = (rt.has_prior.to_numpy() == 0) & ~np.isnan(zz)
    return rt.assign(prior_logit=np.where(use, zz, rt.prior_logit))


def round_cols():
    return ROUND + (ROUND_X if CFG["round_feats"] else []) + (["ct_prior", "has_prior"] if CFG["round_prior"] else [])


def map_cols():
    base = [c for c in MAPCOLS if not (CFG["map_nohas"] and c == "has_prior")]
    return base + list(CFG["map_x"]) + (["dp"] if CFG["dp"] else [])


def with_dp(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(dp=dp_value(X.need_a, X.need_b, X.ot, X.prior_logit))


class Bag:
    """Several tree models on different seeds, probabilities averaged: the map model's per-round rows share one label
    per map, so single fits carry a lot of seed noise."""

    def __init__(self, make, n, seed):
        self.models = [make(seed * 1000 + i) for i in range(n)]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


TIERS = {"s": 0, "a": 1, "b": 2, "c": 3, "d": 4}

MONO = {"need_a": -1, "need_b": 1, "prior_logit": 1, "side_adv_ct": 1, "side_adv_t": 1, "kd_rate": 1, "dp": 1}


def map_trees(seed):
    # Once the stacked prior is calibrated on outcomes either way, more rounds needed and a weaker prior can only hurt
    # team A; the constraint keeps the per-round trees from fitting map-level noise in the opposite direction.
    kw = dict(CFG["map_kw"])
    if CFG["map_mono"]:
        kw["monotone_constraints"] = "(" + ",".join(str(MONO.get(c, 0)) for c in map_cols()) + ")"
    return trees(seed, **kw)


def rows(model, b: Bundle, states: pd.DataFrame, rates=None, team=None, pm=None):
    sub = b.subset(states.game_id)
    rt = round_table(sub)
    if rates is None:
        rates, team = strength(model, b, rt.game_id.unique())
        pm = premap(model, b, rt.game_id.unique(), rates, team)
    rt = with_prior2(rt, pm, model.get("premap"), model.get("stack"))
    rt = pd.concat([rt, map_skill(sub, rt, rates, team)], axis=1).set_index(["game_id", "round"])
    x = states.join(state_features(sub, states))
    x = x.join(rt[["side_a", "a_equip", "b_equip", "need_a", "need_b", "map_code", "prior_logit", "has_prior"]], on=["game_id", "round"])
    a_ct = x.side_a == "CT"
    x["ct_eq_start"] = np.where(a_ct, x.a_equip, x.b_equip); x["t_eq_start"] = np.where(a_ct, x.b_equip, x.a_equip)
    x["ct_need"] = np.where(a_ct, x.need_a, x.need_b); x["t_need"] = np.where(a_ct, x.need_b, x.need_a)
    x["ct_prior"] = np.where(a_ct, x.prior_logit, -x.prior_logit)
    x = x.join(round_skill(sub, states, rates, team))
    return x, rt


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    hist = History(train, CFG["half_life"], asof)
    model = dict(hist=hist, pri=priors_from(hist))
    r = train.rounds[train.rounds.a_wins.notna()]
    ct_won = np.where(r.side_a == "CT", r.a_wins, 1 - r.a_wins)
    model["map_ct"] = pd.Series(ct_won).groupby(r["map"].to_numpy()).mean().to_dict()
    done = train.games[train.games.map_a_wins.notna()]
    rates, team = strength(model, train, done.game_id)
    pm = premap(model, train, done.game_id, rates, team)
    model["premap"] = fit_premap(pm, done.set_index("game_id").map_a_wins)
    model["stack"] = fit_stack(train, premap_z(pm, model["premap"]), pm)
    legal = legal_maps(train)
    sub = train.subset(legal)
    rt = with_prior2(round_table(sub), pm, model["premap"], model["stack"])
    rt = pd.concat([rt, map_skill(sub, rt, rates, team)], axis=1)
    rt = rt[rt.map_a_wins.notna()]
    model["map_model"] = Bag(map_trees, CFG["map_bag"], seed).fit(with_dp(rt)[map_cols()], rt.map_a_wins)
    st = map_states(train, legal)
    x, rti = rows(model, sub, st, rates, team, pm)
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
        X = with_dp(pd.concat([X, rr[MAP_X]], axis=1))
        p = model["map_model"].predict_proba(X[map_cols()])[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
    p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()
