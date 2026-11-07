"""The benchmark entry. Two gradient-boosted tree models:

1. Round model: P(CT wins this round) from the state at round start and after each kill/plant (alive, round-start
   equipment still alive, AWPs and kits alive, plant, seconds elapsed and since the plant), each side's round-start
   equipment, rounds each side still needs, map. Framed CT vs T so the team order carries nothing.
2. Map model: P(team A wins the map) at the start of a round from the score state alone (rounds needed, side,
   overtime, round in half, pistol, loss-bonus counters, map) and a prior: the kickoff market's map price where the
   map's series was market-listed, Elo otherwise (flagged).

Price at a state: P(A wins round) * M(next score if A wins) + P(A loses round) * M(next score if A loses), with M = 1/0
when the round ends the map. M has no economy, because the next round's buy isn't known yet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .data import Bundle
from .features import legal_maps, loss_step, map_states, need, round_table, side_swaps_at, state_features

MAPCOLS = ["need_a", "need_b", "ct_a", "ot", "round_in_half", "pistol", "map_code", "a_loss_ctr", "b_loss_ctr", "prior_logit", "has_prior"]
ROUND = ["ct_alive", "t_alive", "ct_eq", "t_eq", "ct_awp", "t_awp", "ct_kits", "planted", "t", "since_plant",
         "ct_eq_start", "t_eq_start", "ct_need", "t_need", "map_code"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def with_prior(rt: pd.DataFrame) -> pd.DataFrame:
    has = rt.p_market.notna() if "p_market" in rt else pd.Series(False, index=rt.index)
    return rt.assign(has_prior=has.astype(int), prior_logit=np.where(has, logit(rt.get("p_market", 0.5)), rt.elo_logit))


def round_rows(b: Bundle, states: pd.DataFrame) -> pd.DataFrame:
    """State features joined to their round's start-of-round info, in CT-vs-T terms."""
    rt = with_prior(round_table(b.subset(states.game_id))).set_index(["game_id", "round"])
    f = state_features(b, states)
    x = states.join(f).join(rt[["side_a", "a_equip", "b_equip", "need_a", "need_b", "map_code"]], on=["game_id", "round"])
    a_ct = x.side_a == "CT"
    x["ct_eq_start"] = np.where(a_ct, x.a_equip, x.b_equip); x["t_eq_start"] = np.where(a_ct, x.b_equip, x.a_equip)
    x["ct_need"] = np.where(a_ct, x.need_a, x.need_b); x["t_need"] = np.where(a_ct, x.need_b, x.need_a)
    return x, rt


def next_round(rt: pd.DataFrame, a_won: int):
    """Map-model features of the round after each row, given who won it; done is 1/0 where this round ends the map."""
    sa, sb = rt.score_a + a_won, rt.score_b + 1 - a_won
    n = rt["round"] + 1
    nd = np.array([need(a, c) for a, c in zip(sa, sb)])
    ot = (np.minimum(sa, sb) >= 12).astype(int)
    la, lb = loss_step(rt.a_loss_ctr.to_numpy(), rt.b_loss_ctr.to_numpy(), np.full(len(rt), a_won), n.to_numpy())
    x = pd.DataFrame(dict(need_a=nd[:, 0], need_b=nd[:, 1], ct_a=np.where(side_swaps_at(n), 1 - rt.ct_a, rt.ct_a), ot=ot,
                          round_in_half=np.where(ot == 1, (n - 25) % 3, (n - 1) % 12), pistol=n.isin([1, 13]).astype(int),
                          map_code=rt.map_code, a_loss_ctr=la, b_loss_ctr=lb, prior_logit=rt.prior_logit, has_prior=rt.has_prior), index=rt.index)
    done = np.where(sa >= rt.score_a + rt.need_a, 1.0, np.where(sb >= rt.score_b + rt.need_b, 0.0, np.nan))
    return x[MAPCOLS], done


def trees(seed, **kw):
    params = dict(n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                  reg_lambda=5, n_jobs=8, random_state=seed)
    params.update(kw)
    return xgb.XGBClassifier(**params)


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    legal = legal_maps(train)
    rt = with_prior(round_table(train.subset(legal)))
    rt = rt[rt.map_a_wins.notna()]
    map_model = trees(seed).fit(rt[MAPCOLS], rt.map_a_wins)
    st = map_states(train, legal)
    x, rti = round_rows(train, st)
    x = x.join(rti[["a_wins"]], on=["game_id", "round"])
    y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    round_model = trees(seed, max_depth=6, learning_rate=0.05).fit(x[ROUND], y)
    return dict(map_model=map_model, round_model=round_model)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    x, rt = round_rows(ctx, states)
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
