"""The structure entry: the benchmark's round model x map value split, with the map value taken from a dynamic
programme over round-win probabilities (dp.py) blended with a regression of the map outcome on the score and the
economy carried into the next round."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from ..baseline import MAPCOLS, ROUND
from ..data import Bundle
from ..features import legal_maps, map_states, resets
from . import dp, feats

CARRY = ["a_carry", "b_carry", "a_eq_prev", "b_eq_prev"]
MC = MAPCOLS + ["a_carry", "b_carry", "a_eq_prev", "b_eq_prev", "a_won_last"]
RC = ROUND + ["ct_prior", "since_kill", "last_dead_ct", "first_dead_ct", "site_a"]


@dataclass
class Config:
    w_dp: float = 0.7
    rcols: list = field(default_factory=lambda: list(RC))
    round_kw: dict = field(default_factory=lambda: dict(max_depth=6, learning_rate=0.05))
    map_kw: dict = field(default_factory=lambda: dict(min_child_weight=200))
    q_kw: dict = field(default_factory=dict)
    lookahead: bool = True
    q1_kw: dict = field(default_factory=dict)
    recency_half_life: float | None = None
    n_jobs: int = 8


def trees(seed, n_jobs, **kw):
    params = dict(n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                  reg_lambda=5, n_jobs=n_jobs, random_state=seed, tree_method="hist")
    params.update(kw)
    return xgb.XGBClassifier(**params)


def rounds(b: Bundle, game_ids) -> pd.DataFrame:
    rt = feats.rounds_frame(b, game_ids)
    rt["streak"] = dp.streaks(rt)
    cur = feats.current_round_map_features(rt)
    return pd.concat([rt, cur.add_prefix("m_")], axis=1)


def state_rows(b: Bundle, states: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    ef = feats.event_features(b, set(states.game_id))
    x = states[["game_id", "round", "event_seq"]].merge(
        ef.drop(columns=["round", "type", "elapsed"]).rename(columns={"seq": "event_seq"}), on=["game_id", "event_seq"], how="left")
    cols = ["side_a", "a_equip", "b_equip", "need_a", "need_b", "map_code", "prior_logit", "a_wins", "a_money", "b_money",
            "a_loss_ctr", "b_loss_ctr", "ot", "streak", "q_a"]
    x = x.join(rt.set_index(["game_id", "round"])[[c for c in cols if c in rt]], on=["game_id", "round"])
    a_ct = (x.side_a == "CT").to_numpy()
    x["ct_eq_start"] = np.where(a_ct, x.a_equip, x.b_equip); x["t_eq_start"] = np.where(a_ct, x.b_equip, x.a_equip)
    x["ct_need"] = np.where(a_ct, x.need_a, x.need_b); x["t_need"] = np.where(a_ct, x.need_b, x.need_a)
    x["ct_prior"] = np.where(a_ct, x.prior_logit, -x.prior_logit)
    x["a_ct"] = a_ct
    x["ct_money"] = np.where(a_ct, x.a_money, x.b_money); x["t_money"] = np.where(a_ct, x.b_money, x.a_money)
    x["ct_loss"] = np.where(a_ct, x.a_loss_ctr, x.b_loss_ctr); x["t_loss"] = np.where(a_ct, x.b_loss_ctr, x.a_loss_ctr)
    x["ct_streak"] = np.where(a_ct, x.streak, -x.streak)
    if "q_a" in x:
        x["ct_q"] = np.where(a_ct, x.q_a, 1 - x.q_a)
    return x


def fit(train: Bundle, asof: pd.Timestamp, seed: int, cfg: Config):
    legal = legal_maps(train)
    rt = rounds(train, legal).merge(train.games[["game_id", "end"]], on="game_id")
    rt = rt[rt.map_a_wins.notna()]
    qkw = dict(cfg.q_kw)
    if cfg.recency_half_life:
        qkw["weight"] = 0.5 ** ((asof - rt.end).dt.days.to_numpy() / cfg.recency_half_life)
    q = dp.fit_q(rt, seed, n_jobs=cfg.n_jobs, **qkw)
    q1 = None
    if cfg.lookahead:
        # The next round's win chance also depends on the money and equipment carried out of this round.
        X1 = rt[["m_" + c for c in CARRY]].set_axis(CARRY, axis=1).join(rt[dp.QCOLS])
        q1 = trees(seed, cfg.n_jobs, **dict(dict(n_estimators=400, learning_rate=0.05, min_child_weight=100), **cfg.q1_kw)).fit(X1, rt.a_wins)
    X = rt[["m_" + c for c in MC]].set_axis(MC, axis=1)
    mm = trees(seed, cfg.n_jobs, **cfg.map_kw).fit(X, rt.map_a_wins)
    rt["q_a"] = q.predict_proba(rt[dp.QCOLS])[:, 1]
    st = map_states(train, legal)
    x = state_rows(train, st, rt)
    y = np.where(x.a_ct, x.a_wins, 1 - x.a_wins)
    rm = trees(seed, cfg.n_jobs, **cfg.round_kw).fit(x[cfg.rcols], y)
    return dict(q=q, q1=q1, map_model=mm, round_model=rm, cfg=cfg)


def components(model, ctx: Bundle, states: pd.DataFrame) -> pd.DataFrame:
    """Per state: P(A wins this round), and the map value after an A win / loss from the DP (d1, d0) and the
    regression (r1, r0)."""
    cfg = model["cfg"]
    rt = rounds(ctx, set(states.game_id))
    statics = rt.groupby("game_id").head(1).reset_index(drop=True)
    V = dp.values(model["q"], statics)
    mix = pd.Series(np.arange(len(statics)), index=statics.game_id)
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rr = rt.set_index(["game_id", "round"]).loc[keys].reset_index()
    d1, d0 = dp.next_values(V, mix.loc[rr.game_id].to_numpy(), rr.score_a.to_numpy(), rr.score_b.to_numpy(),
                            rr.streak.to_numpy(), rr["round"].to_numpy())
    mi = mix.loc[rr.game_id].to_numpy()
    n = rr["round"].to_numpy()
    rx, ex = {}, {}
    for a_won in (1, 0):
        X, done = feats.next_round_features(rr, a_won)
        p = model["map_model"].predict_proba(X[MC])[:, 1]
        rx[a_won] = np.where(np.isnan(done), p, done)
        if model.get("q1") is not None:
            k2 = np.where(resets(n + 1), 0, dp.step_streak(rr.streak.to_numpy(), a_won == 1))
            X["streak"] = k2
            p1 = model["q1"].predict_proba(X[CARRY + dp.QCOLS])[:, 1]
            sa2, sb2 = rr.score_a.to_numpy() + a_won, rr.score_b.to_numpy() + 1 - a_won
            ok = np.isnan(done)
            v1, v0 = dp.next_values(V, mi, np.where(ok, sa2, 0), np.where(ok, sb2, 0), np.where(ok, k2, 0), n + 1)
            ex[a_won] = np.where(ok, p1 * v1 + (1 - p1) * v0, done)
    if ex:
        d1, d0 = ex[1], ex[0]
    rt["q_a"] = model["q"].predict_proba(rt[dp.QCOLS])[:, 1]
    x = state_rows(ctx, states, rt)
    p_ct = model["round_model"].predict_proba(x[cfg.rcols])[:, 1]
    p_a = np.where(x.a_ct, p_ct, 1 - p_ct)
    return pd.DataFrame(dict(p_a=p_a, d1=d1, d0=d0, r1=rx[1], r0=rx[0]), index=states.index)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    c = components(model, ctx, states)
    w = model["cfg"].w_dp
    m1 = w * c.d1 + (1 - w) * c.r1
    m0 = w * c.d0 + (1 - w) * c.r0
    return (c.p_a * m1 + (1 - c.p_a) * m0).to_numpy()
