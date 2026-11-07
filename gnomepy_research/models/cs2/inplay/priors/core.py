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
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, logit, next_round, trees
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table, state_features
from gnomepy_research.models.cs2.inplay.priors import ratings

TIERS = ["s", "a", "b", "c", "d"]
RAT = ["elo_logit", "r_team", "r_map", "r_player", "q_team", "q_map", "q_player"]
BASIC = RAT + ["bo1"] + [f"tier_{t}" for t in TIERS]
NM_SETS = {
    "v1": BASIC + ["s_diff", "s_rd"],
    # Ratings of players with few maps are noise, so their weight scales with the roster's experience.
    "p2": BASIC + [f"{c}_p" for c in RAT] + ["npmin", "rest_a", "rest_b"],
    "p2s": BASIC + [f"{c}_p" for c in RAT] + ["npmin", "rest_a", "rest_b", "s_diff", "s_rd"],
    "x1": BASIC + ["npmin", "nmin", "rest_a", "rest_b", "np_a", "np_b", "n_a", "n_b", "roster_known"],
    "x2": BASIC + ["npmin", "nmin", "rest_a", "rest_b", "np_a", "np_b", "n_a", "n_b", "roster_known", "s_diff", "s_rd"],
}
MK = ["l_market", "l_e12", "l_pool", "nm_logit", "s_diff", "s_rd"]
# A series price does not know which map is played; team-on-map ratings carry the map pick.
MK_MAP = ["l_avg", "q_map", "r_map"]
EDGE = NM_SETS["v1"] + ["l_market", "l_e12", "has_m"]
MIN_MARKET = 300


@dataclass(frozen=True)
class Config:
    prior_from: str = "2025-04-01"
    edge_round: bool = False
    edge_map: bool = False
    edge_from_prior: bool = False
    map_extra: tuple = ()
    mono: bool = False
    map_kw: tuple = ()
    market_prior: bool = True
    market_mode: str = "lr"
    nm_set: str = "v1"
    nm_kind: str = "lr"
    folds: int = 5
    k: tuple = ()


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
    g["nmin"] = np.log1p(np.minimum(g.n_a, g.n_b))
    g["npmin"] = np.log1p(np.minimum(g.np_a, g.np_b))
    for c in RAT:
        g[f"{c}_p"] = g[c] * g.npmin
    for c in ("rest_a", "rest_b"):
        g[c] = np.log1p(g[c].fillna(60).clip(0, 60))
    return g


class Model:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def nm_logit(self, g: pd.DataFrame) -> np.ndarray:
        if self.cfg.nm_kind == "blend":
            return (self.nm.decision_function(g[NM_SETS[self.cfg.nm_set]]) + self.nm2.predict(g[NM_SETS["x1"]], output_margin=True)) / 2
        x = g[NM_SETS[self.cfg.nm_set]]
        if self.cfg.nm_kind == "xgb":
            return self.nm.predict(x, output_margin=True)
        return self.nm.decision_function(x)

    def nm_model(self, seed=0):
        if self.cfg.nm_kind == "xgb":
            return xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                     min_child_weight=50, n_jobs=8, random_state=seed)
        return LogisticRegression(C=1.0, max_iter=3000)

    def prior_table(self, g: pd.DataFrame, nm: np.ndarray | None = None) -> pd.DataFrame:
        nm = self.nm_logit(g) if nm is None else nm
        if self.cfg.market_mode == "avg":
            # The kickoff market and e12 err differently; their logit average beats either and needs no fitting,
            # so it also covers the months before any market-listed map is in the training history.
            mk = ((g.l_market + g.l_e12) / 2).to_numpy()
        elif self.cfg.market_mode == "avgmap":
            mk = ((g.l_market + g.l_e12) / 2).to_numpy()
            if self.mk is not None:
                mk = self.mk.decision_function(g.assign(l_avg=mk)[MK_MAP])
        elif self.cfg.market_mode == "avg3":
            mk = ((g.l_market + g.l_e12 + g.l_pool) / 3).to_numpy()
        elif self.cfg.market_mode == "e12w":
            mk = (0.4 * g.l_market + 0.6 * g.l_e12).to_numpy()
        elif self.cfg.market_mode == "avgnm":
            mk = (0.85 * (g.l_market + g.l_e12) / 2 + 0.15 * nm).to_numpy()
        elif self.mk is not None and self.cfg.market_prior:
            mk = self.mk.decision_function(g.assign(nm_logit=nm)[MK])
        else:
            mk = g.l_market.to_numpy()
        out = pd.DataFrame(dict(game_id=g.game_id.to_numpy(), my_prior=np.where(g.has_m == 1, mk, nm), has_m=g.has_m.to_numpy()))
        if self.cfg.edge_from_prior:
            out["edge"] = out.my_prior
        else:
            out["edge"] = self.edge.decision_function(g[EDGE]) if self.edge is not None else 0.0
        out["tier_c"] = g.tier.map({t: i for i, t in enumerate(TIERS)}).fillna(4).to_numpy()
        out["bo1"] = g.bo1.to_numpy()
        out["npmin"] = g.npmin.to_numpy()
        return out

    def fit_prior(self, g: pd.DataFrame, train: Bundle, seed: int = 0) -> np.ndarray:
        """Fits the prior models and returns the rating prior for every row of g: out-of-fold (by match) for the
        rows it was fitted on, so the map model learns how much to trust a prior it has not seen the answer of."""
        use = (g.result.notna() & (g.begin.fillna(g.match_begin) >= pd.Timestamp(self.cfg.prior_from, tz="UTC"))).to_numpy()
        tr = g[use]
        cols = NM_SETS[self.cfg.nm_set]
        self.nm = self.nm_model(seed).fit(tr[cols], tr.result)
        if self.cfg.nm_kind == "blend":
            self.nm2 = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                         min_child_weight=50, n_jobs=8, random_state=seed).fit(tr[NM_SETS["x1"]], tr.result)
        nm = self.nm_logit(g)
        if self.cfg.nm_kind == "xgb" and self.cfg.folds > 1:
            fold = (pd.util.hash_array(tr.match_id.to_numpy()) % self.cfg.folds).astype(int)
            oof = np.zeros(len(tr))
            for k in range(self.cfg.folds):
                m = self.nm_model(seed).fit(tr[cols][fold != k], tr.result[fold != k])
                oof[fold == k] = m.predict(tr[cols][fold == k], output_margin=True)
            nm[use] = oof
        tm = tr.assign(nm_logit=nm[use])[tr.has_m.to_numpy() == 1]
        self.mk = None
        if len(tm) >= MIN_MARKET:
            if self.cfg.market_mode == "avgmap":
                self.mk = LogisticRegression(C=1.0, max_iter=2000).fit(tm.assign(l_avg=(tm.l_market + tm.l_e12) / 2)[MK_MAP], tm.result)
            else:
                self.mk = LogisticRegression(C=1.0, max_iter=2000).fit(tm[MK], tm.result)
        self.edge = None
        if (self.cfg.edge_round or self.cfg.edge_map) and not self.cfg.edge_from_prior:
            r = train.rounds[train.rounds.game_id.isin(set(tr.game_id))][["game_id", "a_wins"]].merge(tr[["game_id"] + EDGE], on="game_id")
            self.edge = LogisticRegression(C=1.0, max_iter=2000).fit(r[EDGE], r.a_wins)
        return nm

    def map_cols(self):
        return MAPCOLS + (["edge"] if self.cfg.edge_map else []) + list(self.cfg.map_extra)

    def map_trees(self, seed):
        kw = dict(self.cfg.map_kw)
        if not self.cfg.mono:
            return trees(seed, **kw)
        # A stronger prior or fewer rounds still needed can only help team A; forcing that removes tree noise
        # where data is thin (overtime, lopsided priors).
        sign = {"prior_logit": 1, "need_a": -1, "need_b": 1}
        return trees(seed, **kw, monotone_constraints="(" + ",".join(str(sign.get(c, 0)) for c in self.map_cols()) + ")")

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
        self.params = ratings.Params(*self.cfg.k)
        self.asof = asof
        info = ratings.game_info(train)
        feats, self.state = ratings.run(info, self.params)
        g = map_frame(info, feats, train)
        prior = self.prior_table(g, self.fit_prior(g, train, seed))
        legal = legal_maps(train)
        rt = with_prior(round_table(train.subset(legal)), prior)
        rt = rt[rt.map_a_wins.notna()]
        self.map_model = self.map_trees(seed).fit(rt[self.map_cols()], rt.map_a_wins)
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
            X = X.assign(**{c: rr[c].to_numpy() for c in self.map_cols() if c not in MAPCOLS})
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
