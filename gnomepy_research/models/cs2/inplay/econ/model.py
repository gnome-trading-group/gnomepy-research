"""Economy-aware in-play map price.

Price at a state s of round r: P(A wins r | s) * M(s, A wins) + P(A loses r | s) * M(s, A loses).

The benchmark's M only sees the next score. Here M(s, o) is the benchmark score-state map model (trained on every
map since 2025) used as a base margin, plus a boosted correction learned on in-round states of maps with timelines,
conditioned on the actual round outcome o: given o, who is alive now (an upper bound on who keeps their weapons), the
money each team holds after this round's buy, kills made so far (kill rewards), the loss-bonus counters and the
projected pre-buy money of the next round describe the next round's economy. Rows are drawn from P(s, o), so a model
fitted on rows with their realised outcome estimates P(map | s, o) directly, without forecasting survivors.

The round model is the benchmark's plus economy within the round: money left after the buy, rifles and helmets alive."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..baseline import MAPCOLS, ROUND, next_round, trees, with_prior
from ..data import Bundle
from ..features import legal_maps, map_states, resets, round_table
from .econ_features import SIDE_COLS, loss_counters, state_table, team_money

ROUND_EXTRA = ["ct_rifles", "t_rifles", "ct_helmets", "t_helmets", "ct_mleft", "t_mleft", "ct_money_pre", "t_money_pre"]
SCORE = ["o", "n_need_a", "n_need_b", "n_ct_a"]
V1 = SCORE + ["a_alive", "b_alive", "a_eq", "b_eq", "a_rifles", "b_rifles", "a_awp", "b_awp", "a_kills", "b_kills", "a_mleft",
              "b_mleft", "a_equip", "b_equip", "n_la", "n_lb", "proj_a", "proj_b", "res_a", "res_b", "d_res", "planted", "t"]
ECO_SETS = {
    "v1": V1,
    "v2": V1 + ["nbuy_a", "nbuy_b", "nfull_a", "nfull_b", "d_nbuy"],
    "score": SCORE,
    "score_alive": SCORE + ["a_alive", "b_alive", "a_eq", "b_eq"],
    "money_only": SCORE + ["a_mleft", "b_mleft", "a_equip", "b_equip", "n_la", "n_lb", "proj_a", "proj_b"],
    "lean": SCORE + ["a_alive", "b_alive", "a_eq", "b_eq", "n_la", "n_lb", "proj_a", "proj_b", "nbuy_a", "nbuy_b", "d_res", "d_nbuy"],
}

DEFAULT = dict(round_extra=False, eco=True, eco_feats="v2", aug_weight=0,
               eco_trees=dict(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.7, colsample_bytree=0.8,
                              min_child_weight=300, reg_lambda=20))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def ab_view(x: pd.DataFrame) -> pd.DataFrame:
    a_ct = (x.side_a == "CT").to_numpy()
    out = {}
    for c in SIDE_COLS:
        ct, t = x[f"ct_{c}"].to_numpy(), x[f"t_{c}"].to_numpy()
        out[f"a_{c}"] = np.where(a_ct, ct, t)
        out[f"b_{c}"] = np.where(a_ct, t, ct)
    return pd.DataFrame(out, index=x.index)


def round_info(b: Bundle, game_ids) -> pd.DataFrame:
    rt = with_prior(round_table(b.subset(game_ids)))
    tm = team_money(b, game_ids)
    rt = rt.join(tm, on=["game_id", "round"])
    rt["a_ctr"], rt["b_ctr"] = loss_counters(rt)
    return rt.set_index(["game_id", "round"])


def state_rows(b: Bundle, states: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    r = rt.reindex(pd.MultiIndex.from_frame(states[["game_id", "round"]]))
    a_ct = (r.side_a == "CT").to_numpy()
    f = state_table(b, states, np.where(a_ct, r.a_ctr, r.b_ctr), np.where(a_ct, r.b_ctr, r.a_ctr))
    cols = ["side_a", "a_equip", "b_equip", "a_money_pre", "b_money_pre", "a_mleft", "b_mleft", "need_a", "need_b", "map_code"]
    x = states.join(f).join(rt[cols], on=["game_id", "round"])
    a_ct = x.side_a == "CT"
    sw = lambda a, b: (np.where(a_ct, x[a], x[b]), np.where(a_ct, x[b], x[a]))
    x["ct_eq_start"], x["t_eq_start"] = sw("a_equip", "b_equip")
    x["ct_need"], x["t_need"] = sw("need_a", "need_b")
    x["ct_mleft"], x["t_mleft"] = sw("a_mleft", "b_mleft")
    x["ct_money_pre"], x["t_money_pre"] = sw("a_money_pre", "b_money_pre")
    return x.join(ab_view(x))


def start_rows(b: Bundle, game_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Round-start states of maps without timelines: everyone alive, nothing happened yet. They carry the economy
    between rounds for every map with player data, not only the 2026 maps with timelines."""
    r = b.rounds[b.rounds.game_id.isin(set(game_ids)) & b.rounds.has_players]
    st = pd.DataFrame(dict(game_id=r.game_id.to_numpy(), round=r["round"].to_numpy(), step=0, kind="start", event_seq=-1,
                           ingame_ms=0, elapsed=0))
    rt = round_info(b, set(st.game_id))
    return state_rows(b, st, rt), rt


def eco_rows(x: pd.DataFrame, rt: pd.DataFrame, o: int, feats=V1):
    """M-correction features of each state given round outcome o (1 = A wins the round), the score-state map features
    of the next round and done (1/0 where o ends the map)."""
    keys = pd.MultiIndex.from_frame(x[["game_id", "round"]])
    r = rt.reindex(keys).reset_index()
    nxt, done = next_round(r, o)
    reset = resets((r["round"] + 1).to_numpy())
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
    e["nbuy_a"], e["nbuy_b"] = x[f"a_nbuy_{wa}"].to_numpy(), x[f"b_nbuy_{wb}"].to_numpy()
    e["nfull_a"], e["nfull_b"] = x[f"a_nfull_{wa}"].to_numpy(), x[f"b_nfull_{wb}"].to_numpy()
    e["d_nbuy"] = e.nbuy_a - e.nbuy_b
    e["res_a"] = e.proj_a + x.a_eq.to_numpy()
    e["res_b"] = e.proj_b + x.b_eq.to_numpy()
    e["d_res"] = e.res_a - e.res_b
    e["planted"], e["t"] = x.planted.to_numpy(), x.t.to_numpy()
    econ = [c for c in e.columns if c not in SCORE]
    # At a half or overtime-half start money resets, so this round's economy says nothing about the next.
    e.loc[reset, econ] = np.nan
    return e[feats], nxt[MAPCOLS], done


class Model:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = seed

    def round_cols(self):
        return ROUND + (ROUND_EXTRA if self.cfg["round_extra"] else [])

    def fit(self, train: Bundle):
        legal = legal_maps(train)
        st = map_states(train, legal)
        rt = round_info(train, set(st.game_id))
        aug = start_rows(train, legal - set(st.game_id)) if self.cfg["aug_weight"] else None
        return self.fit_rows(train, legal, state_rows(train, st, rt), rt, aug=aug)

    def eco_part(self, x, rt, w):
        a_wins = rt.a_wins.reindex(pd.MultiIndex.from_frame(x[["game_id", "round"]])).to_numpy()
        out = []
        for o in (1, 0):
            m = a_wins == o
            e, nx, done = eco_rows(x[m], rt, o, ECO_SETS[self.cfg["eco_feats"]])
            keep = np.isnan(done)
            base = logit(self.map_model.predict_proba(nx[keep])[:, 1])
            mw = rt.map_a_wins.reindex(pd.MultiIndex.from_frame(x[m][["game_id", "round"]])).to_numpy()[keep]
            out.append((e[keep], base, mw, np.full(keep.sum(), w)))
        return out

    def fit_rows(self, train: Bundle, legal, x: pd.DataFrame, rt: pd.DataFrame, map_model=None, round_model=None, aug=None):
        if map_model is None:
            rt_all = with_prior(round_table(train.subset(legal)))
            rt_all = rt_all[rt_all.map_a_wins.notna()]
            map_model = trees(self.seed).fit(rt_all[MAPCOLS], rt_all.map_a_wins)
        self.map_model = map_model
        a_wins = rt.a_wins.reindex(pd.MultiIndex.from_frame(x[["game_id", "round"]])).to_numpy()
        y = np.where(x.side_a == "CT", a_wins, 1 - a_wins)
        if round_model is None:
            round_model = trees(self.seed, max_depth=6, learning_rate=0.05).fit(x[self.round_cols()], y)
        self.round_model = round_model
        self.eco_model = None
        if self.cfg["eco"]:
            parts = self.eco_part(x, rt, 1.0)
            if aug is not None and self.cfg["aug_weight"]:
                parts += self.eco_part(*aug, self.cfg["aug_weight"])
            E = pd.concat([p[0] for p in parts]); base = np.concatenate([p[1] for p in parts]); yy = np.concatenate([p[2] for p in parts])
            wt = np.concatenate([p[3] for p in parts])
            params = dict(objective="binary:logistic", n_jobs=8, random_state=self.seed, **self.cfg["eco_trees"])
            self.eco_model = xgb.XGBClassifier(**params)
            self.eco_model.fit(E, yy, base_margin=base, sample_weight=wt)
        return self

    def m_value(self, x, rt, o):
        e, nx, done = eco_rows(x, rt, o, ECO_SETS[self.cfg["eco_feats"]])
        base = logit(self.map_model.predict_proba(nx)[:, 1])
        if self.eco_model is None:
            p = 1 / (1 + np.exp(-base))
        else:
            p = self.eco_model.predict_proba(e, base_margin=base)[:, 1]
        return np.where(np.isnan(done), p, done)

    def predict(self, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
        rt = round_info(ctx, set(states.game_id))
        return self.predict_rows(state_rows(ctx, states, rt), rt)

    def predict_rows(self, x: pd.DataFrame, rt: pd.DataFrame) -> np.ndarray:
        p_ct = self.round_model.predict_proba(x[self.round_cols()])[:, 1]
        p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
        return p_a * self.m_value(x, rt, 1) + (1 - p_a) * self.m_value(x, rt, 0)


def make(**over):
    cfg = {**DEFAULT, **over}

    def fit(train: Bundle, asof: pd.Timestamp, seed: int):
        return Model(cfg, seed).fit(train)

    def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
        return model.predict(ctx, states)

    return fit, predict
