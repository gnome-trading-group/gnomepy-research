"""Economy-aware in-play map price.

Price at a state s of round r: P(A wins r | s) * M(s, A wins) + P(A loses r | s) * M(s, A loses).

The round model is the benchmark's. The benchmark's M only sees the next score; here M(s, o) is that score-state map
model (trained on every map since 2025) used as a base margin, plus a boosted correction learned on in-round states of
the maps with timelines, each conditioned on its realised round outcome o. Given o, who is alive now (survivors keep
their weapons), the money each team holds after this round's buy, the kill rewards earned so far, the loss counters,
each team's projected pre-buy money and how many players could then buy rifle + armor describe the next round's
economy. Rows are drawn from P(s, o), so a model fitted on them estimates P(map | s, o) without forecasting who
survives. Tier is a feature as well: how much economy and score are worth differs by tier.

The correction's target is partly distilled: 30% of it is M given the next round's actual pre-buy economy (fitted on
every map with player data), which removes the noise of the rounds after the next; 70% is the map result."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..baseline import MAPCOLS, ROUND, next_round, trees, with_prior
from ..data import Bundle
from ..features import legal_maps, map_states, resets, round_table
from .econ_features import POST_COLS, SIDE_COLS, loss_counters, post_table, state_table, team_money

TIERS = {"s": 0, "a": 1, "b": 2, "c": 3, "d": 4}
SCORE = ["o", "n_need_a", "n_need_b", "n_ct_a"]
ECON = ["a_alive", "b_alive", "a_eq", "b_eq", "a_rifles", "b_rifles", "a_awp", "b_awp", "a_kills", "b_kills", "a_mleft",
        "b_mleft", "a_equip", "b_equip", "n_la", "n_lb", "proj_a", "proj_b", "res_a", "res_b", "d_res", "planted", "t",
        "nbuy_a", "nbuy_b", "nfull_a", "nfull_b", "d_nbuy"]
FEATS = SCORE + ECON + ["tier"]

DEFAULT = dict(soft=0.3, eco_trees=dict(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.7,
                                        colsample_bytree=0.8, min_child_weight=300, reg_lambda=20))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def tier_codes(b: Bundle, game_ids) -> pd.Series:
    return pd.Series(game_ids).map(b.games.drop_duplicates("game_id").set_index("game_id").tier.map(TIERS)).to_numpy()


def ab_view(x: pd.DataFrame) -> pd.DataFrame:
    a_ct = (x.side_a == "CT").to_numpy()
    out = {}
    for c in SIDE_COLS:
        ct, t = x[f"ct_{c}"].to_numpy(), x[f"t_{c}"].to_numpy()
        out[f"a_{c}"] = np.where(a_ct, ct, t)
        out[f"b_{c}"] = np.where(a_ct, t, ct)
    return pd.DataFrame(out, index=x.index)


def round_info(b: Bundle, game_ids) -> pd.DataFrame:
    """Round-start info per (game_id, round): the benchmark's round table plus money left after the buy, loss counters
    and tier."""
    rt = with_prior(round_table(b.subset(game_ids)))
    rt = rt.join(team_money(b, game_ids), on=["game_id", "round"])
    rt["a_ctr"], rt["b_ctr"] = loss_counters(rt)
    rt["tier"] = tier_codes(b, rt.game_id)
    return rt.set_index(["game_id", "round"])


def state_rows(b: Bundle, states: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    r = rt.reindex(pd.MultiIndex.from_frame(states[["game_id", "round"]]))
    a_ct = (r.side_a == "CT").to_numpy()
    f = state_table(b, states, np.where(a_ct, r.a_ctr, r.b_ctr), np.where(a_ct, r.b_ctr, r.a_ctr))
    cols = ["side_a", "a_equip", "b_equip", "a_mleft", "b_mleft", "need_a", "need_b", "map_code"]
    x = states.join(f).join(rt[cols], on=["game_id", "round"])
    a_ct = x.side_a == "CT"
    sw = lambda a, b: (np.where(a_ct, x[a], x[b]), np.where(a_ct, x[b], x[a]))
    x["ct_eq_start"], x["t_eq_start"] = sw("a_equip", "b_equip")
    x["ct_need"], x["t_need"] = sw("need_a", "need_b")
    return x.join(ab_view(x))


def eco_rows(x: pd.DataFrame, rt: pd.DataFrame, o: int):
    """Correction features of each state given round outcome o (1 = A wins the round), the score-state map features
    of the next round, and done (1/0 where o ends the map)."""
    r = rt.reindex(pd.MultiIndex.from_frame(x[["game_id", "round"]])).reset_index()
    nxt, done = next_round(r, o)
    wa, wb = ("w", "l") if o == 1 else ("l", "w")
    la, lb = r.a_ctr.to_numpy(), r.b_ctr.to_numpy()
    e = pd.DataFrame(index=x.index)
    e["o"] = o
    e["n_need_a"], e["n_need_b"], e["n_ct_a"] = nxt.need_a.to_numpy(), nxt.need_b.to_numpy(), nxt.ct_a.to_numpy()
    for c in ("alive", "eq", "rifles", "awp", "kills"):
        e[f"a_{c}"], e[f"b_{c}"] = x[f"a_{c}"].to_numpy(), x[f"b_{c}"].to_numpy()
    for c in ("a_mleft", "b_mleft", "a_equip", "b_equip"):
        e[c] = x[c].to_numpy()
    e["n_la"] = np.where(o == 1, np.maximum(la - 1, 0), np.minimum(la + 1, 4))
    e["n_lb"] = np.where(o == 1, np.minimum(lb + 1, 4), np.maximum(lb - 1, 0))
    e["proj_a"], e["proj_b"] = x[f"a_proj_{wa}"].to_numpy(), x[f"b_proj_{wb}"].to_numpy()
    e["res_a"] = e.proj_a + x.a_eq.to_numpy()
    e["res_b"] = e.proj_b + x.b_eq.to_numpy()
    e["d_res"] = e.res_a - e.res_b
    e["planted"], e["t"] = x.planted.to_numpy(), x.t.to_numpy()
    e["nbuy_a"], e["nbuy_b"] = x[f"a_nbuy_{wa}"].to_numpy(), x[f"b_nbuy_{wb}"].to_numpy()
    e["nfull_a"], e["nfull_b"] = x[f"a_nfull_{wa}"].to_numpy(), x[f"b_nfull_{wb}"].to_numpy()
    e["d_nbuy"] = e.nbuy_a - e.nbuy_b
    # At a half or overtime-half start money resets, so this round's economy says nothing about the next.
    e.loc[resets((r["round"] + 1).to_numpy()), ECON] = np.nan
    e["tier"] = r.tier.to_numpy()
    return e[FEATS], nxt[MAPCOLS], done


class Model:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = seed

    def fit(self, train: Bundle):
        legal = legal_maps(train)
        st = map_states(train, legal)
        rt = round_info(train, set(st.game_id))
        return self.fit_rows(train, legal, state_rows(train, st, rt), rt)

    def fit_rows(self, train: Bundle, legal, x: pd.DataFrame, rt: pd.DataFrame, map_model=None, round_model=None):
        if map_model is None:
            ra = with_prior(round_table(train.subset(legal)))
            ra = ra[ra.map_a_wins.notna()]
            map_model = trees(self.seed).fit(ra[MAPCOLS], ra.map_a_wins)
        self.map_model = map_model
        keys = pd.MultiIndex.from_frame(x[["game_id", "round"]])
        a_wins = rt.a_wins.reindex(keys).to_numpy()
        if round_model is None:
            y = np.where(x.side_a == "CT", a_wins, 1 - a_wins)
            round_model = trees(self.seed, max_depth=6, learning_rate=0.05).fit(x[ROUND], y)
        self.round_model = round_model

        E, base, y, nxt_keys = [], [], [], []
        map_wins = rt.map_a_wins.reindex(keys).to_numpy()
        for o in (1, 0):
            m = a_wins == o
            e, nx, done = eco_rows(x[m], rt, o)
            keep = np.isnan(done)
            E.append(e[keep])
            base.append(logit(self.map_model.predict_proba(nx[keep])[:, 1]))
            y.append(map_wins[m][keep])
            nxt_keys.append((x.game_id.to_numpy()[m][keep], x["round"].to_numpy()[m][keep] + 1))
        E, base, y = pd.concat(E), np.concatenate(base), np.concatenate(y)
        if self.cfg["soft"]:
            post = self.post_values(train, legal, set(x.game_id))
            gid, rnd = (np.concatenate([k[i] for k in nxt_keys]) for i in (0, 1))
            pv = post.reindex(pd.MultiIndex.from_arrays([gid, rnd])).to_numpy()
            ok = ~np.isnan(pv)
            E, base, y = E[ok], base[ok], self.cfg["soft"] * pv[ok] + (1 - self.cfg["soft"]) * y[ok]
        params = dict(objective="binary:logistic", n_jobs=8, random_state=self.seed, **self.cfg["eco_trees"])
        # A regressor wrapper, because the partly distilled targets are probabilities rather than classes.
        self.eco_model = xgb.XGBRegressor(**params).fit(E, y, base_margin=base)
        return self

    def post_values(self, train: Bundle, legal, game_ids) -> pd.Series:
        """M given the score and the actual pre-buy economy of each round, fitted on every map with player data; its
        values on the timeline maps' rounds, indexed by (game_id, round)."""
        ra = with_prior(round_table(train.subset(legal)))
        ra = ra[ra.map_a_wins.notna()]
        ra["tier"] = tier_codes(train, ra.game_id)
        ra = ra.join(post_table(train, legal), on=["game_id", "round"])
        cols = MAPCOLS + ["tier"] + POST_COLS
        m = trees(self.seed).fit(ra[cols], ra.map_a_wins)
        ev = ra[ra.game_id.isin(game_ids)]
        return pd.Series(m.predict_proba(ev[cols])[:, 1], index=pd.MultiIndex.from_frame(ev[["game_id", "round"]]))

    def m_value(self, x, rt, o):
        e, nx, done = eco_rows(x, rt, o)
        base = logit(self.map_model.predict_proba(nx)[:, 1])
        return np.where(np.isnan(done), self.eco_model.predict(e, base_margin=base), done)

    def predict(self, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
        rt = round_info(ctx, set(states.game_id))
        return self.predict_rows(state_rows(ctx, states, rt), rt)

    def predict_rows(self, x: pd.DataFrame, rt: pd.DataFrame) -> np.ndarray:
        p_ct = self.round_model.predict_proba(x[ROUND])[:, 1]
        p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
        return p_a * self.m_value(x, rt, 1) + (1 - p_a) * self.m_value(x, rt, 0)


def make(**over):
    cfg = {**DEFAULT, **over}

    def fit(train: Bundle, asof: pd.Timestamp, seed: int):
        return Model(cfg, seed).fit(train)

    def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
        return model.predict(ctx, states)

    return fit, predict
