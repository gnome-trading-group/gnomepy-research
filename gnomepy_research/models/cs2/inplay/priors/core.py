"""Benchmark round and map models with a better pre-map prior and a per-round team-strength edge.

The prior is a logistic model on point-in-time ratings (ratings.py: result Elo and round-share Elo for teams, teams on
this map and rosters of players, plus the provided Elo), the series so far (maps of the same match that ended before
this one began), tier and format. Where the series was market-listed, a second logistic model combines the kickoff
market, e12 and pool prices with that rating prior.

The edge is the same information turned into P(A wins a round): a logistic model on rounds. The benchmark's round model
is CT vs T with no notion of who is stronger, so the edge (oriented to the CT side) is given to it, and to the map
model next to the prior, so strength keeps acting through the whole map rather than only through the start price."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, logit, next_round, trees
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table, state_features
from gnomepy_research.models.cs2.inplay.priors import ratings

TIERS = ["s", "a", "b", "c", "d"]
NM = ["elo_logit", "r_team", "r_map", "r_player", "q_team", "q_map", "q_player", "s_diff", "s_rd", "bo1"] + [f"tier_{t}" for t in TIERS]
MK = ["l_market", "l_e12", "l_pool", "nm_logit", "s_diff", "s_rd"]
EDGE = NM + ["l_market", "l_e12", "has_m"]
MIN_MARKET = 300


@dataclass(frozen=True)
class Config:
    prior_from: str = "2025-04-01"
    edge_round: bool = False
    edge_map: bool = False
    market_prior: bool = True


def map_frame(info: pd.DataFrame, feats: pd.DataFrame, b: Bundle) -> pd.DataFrame:
    g = info.merge(feats, on="game_id").merge(b.elo[["game_id", "elo_logit"]], on="game_id", how="left")
    g = g.merge(b.priors, on="game_id", how="left")
    g["elo_logit"] = g.elo_logit.fillna(0.0)
    g["has_m"] = g.p_market.notna().astype(int)
    for c, k in (("p_market", "l_market"), ("p_e12", "l_e12"), ("p_pool", "l_pool")):
        g[k] = np.where(g[c].notna(), logit(g[c].fillna(0.5)), 0.0)
    g["s_diff"] = g.s_a_wins - g.s_b_wins
    g["bo1"] = (g.bo == 1).astype(int)
    for t in TIERS:
        g[f"tier_{t}"] = (g.tier == t).astype(int)
    return g


class Model:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def prior_table(self, g: pd.DataFrame) -> pd.DataFrame:
        nm = self.nm.decision_function(g[NM])
        if self.mk is not None and self.cfg.market_prior:
            mk = self.mk.decision_function(g[MK].assign(nm_logit=nm))
        else:
            mk = g.l_market.to_numpy()
        out = pd.DataFrame(dict(game_id=g.game_id.to_numpy(), my_prior=np.where(g.has_m == 1, mk, nm), has_m=g.has_m.to_numpy()))
        out["edge"] = self.edge.decision_function(g[EDGE]) if self.edge is not None else 0.0
        return out

    def fit_prior(self, g: pd.DataFrame, train: Bundle):
        tr = g[g.result.notna() & (g.begin.fillna(g.match_begin) >= pd.Timestamp(self.cfg.prior_from, tz="UTC"))]
        self.nm = LogisticRegression(C=1.0, max_iter=2000).fit(tr[NM], tr.result)
        tm = tr[tr.has_m == 1]
        self.mk = None
        if len(tm) >= MIN_MARKET:
            self.mk = LogisticRegression(C=1.0, max_iter=2000).fit(tm[MK].assign(nm_logit=self.nm.decision_function(tm[NM])), tm.result)
        self.edge = None
        if self.cfg.edge_round or self.cfg.edge_map:
            r = train.rounds[train.rounds.game_id.isin(set(tr.game_id))][["game_id", "a_wins"]].merge(tr[["game_id"] + EDGE], on="game_id")
            self.edge = LogisticRegression(C=1.0, max_iter=2000).fit(r[EDGE], r.a_wins)

    def map_cols(self):
        return MAPCOLS + (["edge"] if self.cfg.edge_map else [])

    def round_cols(self):
        return ROUND + (["edge_ct"] if self.cfg.edge_round else [])

    def round_rows(self, b: Bundle, states: pd.DataFrame, prior: pd.DataFrame):
        rt = with_prior(round_table(b.subset(states.game_id)), prior).set_index(["game_id", "round"])
        f = state_features(b, states)
        x = states.join(f).join(rt[["side_a", "a_equip", "b_equip", "need_a", "need_b", "map_code", "edge"]], on=["game_id", "round"])
        a_ct = x.side_a == "CT"
        x["ct_eq_start"] = np.where(a_ct, x.a_equip, x.b_equip); x["t_eq_start"] = np.where(a_ct, x.b_equip, x.a_equip)
        x["ct_need"] = np.where(a_ct, x.need_a, x.need_b); x["t_need"] = np.where(a_ct, x.need_b, x.need_a)
        x["edge_ct"] = np.where(a_ct, x.edge, -x.edge)
        return x, rt

    def fit(self, train: Bundle, asof: pd.Timestamp, seed: int):
        self.params = ratings.Params()
        self.asof = asof
        info = ratings.game_info(train)
        feats, self.state = ratings.run(info, self.params)
        g = map_frame(info, feats, train)
        self.fit_prior(g, train)
        prior = self.prior_table(g)
        legal = legal_maps(train)
        rt = with_prior(round_table(train.subset(legal)), prior)
        rt = rt[rt.map_a_wins.notna()]
        self.map_model = trees(seed).fit(rt[self.map_cols()], rt.map_a_wins)
        sts = map_states(train, legal)
        x, rti = self.round_rows(train, sts, prior)
        x = x.join(rti[["a_wins"]], on=["game_id", "round"])
        y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
        self.round_model = trees(seed, max_depth=6, learning_rate=0.05).fit(x[self.round_cols()], y)
        return self

    def predict(self, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
        st = self.state.copy()
        info = ratings.game_info(ctx, set(ctx.games.game_id) - st.done)
        feats, _ = ratings.run(info, self.params, st, floor=self.asof)
        g = map_frame(info, feats, ctx)
        prior = self.prior_table(g[g.game_id.isin(set(states.game_id))])
        x, rt = self.round_rows(ctx, states, prior)
        p_ct = self.round_model.predict_proba(x[self.round_cols()])[:, 1]
        keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
        rr = rt.loc[keys.unique()].reset_index()
        m = {}
        for a_won in (1, 0):
            X, done = next_round(rr, a_won)
            if self.cfg.edge_map:
                X = X.assign(edge=rr.edge.to_numpy())
            p = self.map_model.predict_proba(X[self.map_cols()])[:, 1]
            m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
        p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
        return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()


def with_prior(rt: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    rt = rt.merge(prior, on="game_id", how="left")
    return rt.assign(prior_logit=rt.my_prior.fillna(0.0), has_prior=rt.has_m.fillna(0).astype(int), edge=rt.edge.fillna(0.0))


def entry(cfg: Config):
    def fit(train: Bundle, asof: pd.Timestamp, seed: int):
        return Model(cfg).fit(train, asof, seed)

    def predict(model: Model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
        return model.predict(ctx, states)

    return fit, predict
