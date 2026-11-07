"""Benchmark round and map models with a better pre-map prior.

The prior is a logistic model on point-in-time ratings (ratings.py: result Elo and round-share Elo for teams, teams on
this map and rosters of players, plus the provided Elo), the series so far (maps of the same match that ended before
this one began), tier and format. Where the series was market-listed, a second logistic model combines the kickoff
market, e12 and pool prices with that rating prior. The map model is the benchmark's, fed this prior."""
from __future__ import annotations

import os

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
PRIOR_FROM = pd.Timestamp(os.environ.get("PRIORS_PRIOR_FROM", "2025-04-01"), tz="UTC")
MIN_MARKET = 300


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


def prior_logits(pm: dict, g: pd.DataFrame) -> np.ndarray:
    nm = pm["nm"].decision_function(g[NM])
    g = g.assign(nm_logit=nm)
    if pm["mk"] is not None:
        mk = pm["mk"].decision_function(g[MK])
    else:
        mk = g.l_market.to_numpy()
    return np.where(g.has_m == 1, mk, nm)


def fit_prior(g: pd.DataFrame) -> dict:
    tr = g[g.result.notna() & (g.begin.fillna(g.match_begin) >= PRIOR_FROM)]
    nm = LogisticRegression(C=1.0, max_iter=2000).fit(tr[NM], tr.result)
    tm = tr[tr.has_m == 1].assign(nm_logit=nm.decision_function(tr[tr.has_m == 1][NM]))
    mk = LogisticRegression(C=1.0, max_iter=2000).fit(tm[MK], tm.result) if len(tm) >= MIN_MARKET else None
    return dict(nm=nm, mk=mk)


def with_prior(rt: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    rt = rt.merge(prior, on="game_id", how="left")
    return rt.assign(prior_logit=rt.my_prior.fillna(0.0), has_prior=rt.has_m.fillna(0).astype(int))


def round_rows(b: Bundle, states: pd.DataFrame, prior: pd.DataFrame):
    rt = with_prior(round_table(b.subset(states.game_id)), prior).set_index(["game_id", "round"])
    f = state_features(b, states)
    x = states.join(f).join(rt[["side_a", "a_equip", "b_equip", "need_a", "need_b", "map_code"]], on=["game_id", "round"])
    a_ct = x.side_a == "CT"
    x["ct_eq_start"] = np.where(a_ct, x.a_equip, x.b_equip); x["t_eq_start"] = np.where(a_ct, x.b_equip, x.a_equip)
    x["ct_need"] = np.where(a_ct, x.need_a, x.need_b); x["t_need"] = np.where(a_ct, x.need_b, x.need_a)
    return x, rt


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    prm = ratings.Params()
    info = ratings.game_info(train)
    feats, st = ratings.run(info, prm)
    g = map_frame(info, feats, train)
    pm = fit_prior(g)
    prior = pd.DataFrame(dict(game_id=g.game_id, my_prior=prior_logits(pm, g), has_m=g.has_m))
    legal = legal_maps(train)
    rt = with_prior(round_table(train.subset(legal)), prior)
    rt = rt[rt.map_a_wins.notna()]
    map_model = trees(seed).fit(rt[MAPCOLS], rt.map_a_wins)
    sts = map_states(train, legal)
    x, rti = round_rows(train, sts, prior)
    x = x.join(rti[["a_wins"]], on=["game_id", "round"])
    y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    round_model = trees(seed, max_depth=6, learning_rate=0.05).fit(x[ROUND], y)
    return dict(map_model=map_model, round_model=round_model, prior_model=pm, state=st, params=prm, asof=asof)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    st = model["state"].copy()
    new = set(ctx.games.game_id) - st.done
    info = ratings.game_info(ctx, new)
    feats, _ = ratings.run(info, model["params"], st, floor=model["asof"])
    g = map_frame(info, feats, ctx)
    g = g[g.game_id.isin(set(states.game_id))]
    prior = pd.DataFrame(dict(game_id=g.game_id, my_prior=prior_logits(model["prior_model"], g), has_m=g.has_m))
    x, rt = round_rows(ctx, states, prior)
    p_ct = model["round_model"].predict_proba(x[ROUND])[:, 1]
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rr = rt.loc[keys.unique()].reset_index()
    m = {}
    for a_won in (1, 0):
        X, done = next_round(rr, a_won)
        p = model["map_model"].predict_proba(X)[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
    p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()
