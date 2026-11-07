"""rounddyn: the benchmark's decomposition P(A wins round) x M(score if A wins) + P(A loses round) x M(score if A
loses), with a richer round model and an economy-aware continuation.

Round model: P(CT wins the round) from the benchmark's state (alive, equipment alive, AWPs, kits, plant, timing,
round-start equipment, rounds needed, map) plus what the timeline says about the round so far (feats.dyn_features), the
pre-map prior turned to the CT team, and the CT team's form earlier in the map (feats.form_features). A round already
decided on the timeline (round_end or bomb_exploded at or before the state) is priced from its known winner.

Continuation: M is learned from round n's rows with their actual outcome (the next score state as the benchmark builds
it, plus round n's start-of-round economy and team A's form), so it is E[map win | round n's start, who won round n]
rather than a score-only round-start model: a team that wins a round with $9k left over is better placed than one
that won it on a force buy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states
from gnomepy_research.models.cs2.inplay.rounddyn.feats import FORM, dyn_features, form_features

ROUND = baseline.ROUND + ["ct_rifles", "t_rifles", "ct_smgs", "t_smgs", "ct_helm", "t_helm", "ct_armor", "t_armor",
                          "site", "bomb_left", "clock_left", "defuse_kit_ok", "defuse_nokit_ok", "since_kill",
                          "last_dead", "ct_dead_5s", "t_dead_5s", "last_trade", "open_dead", "ct_max_cls", "t_max_cls",
                          "ct_prior", "has_prior", "ct_kd_prev", "ct_upsets_prev", "ct_gun_wr_prev", "rounds_prev",
                          "gun_prev", "ct_resid_prev"]
ROUND_PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05)
ECON = ["a_money", "b_money", "a_equip", "b_equip", "a_money_pre", "b_money_pre", "a_awp", "b_awp", "a_rifles", "b_rifles"]
CONT = baseline.MAPCOLS + ECON + ["a_cash_next", "b_cash_next", "a_resid_prev"]
SURV = ["a_alive_end", "b_alive_end"]
SURV_PARAMS = dict(n_estimators=120, max_depth=5, learning_rate=0.15, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=50, n_jobs=8)
EXPECT = ["a_equip", "b_equip", "ct_a", "map_code", "pistol", "a_money_pre", "b_money_pre"]


def round_info(b: Bundle) -> pd.DataFrame:
    rt = baseline.with_prior(baseline.round_table(b))
    return rt.join(form_features(b, rt[["game_id", "round"]]))


def round_won(rt: pd.DataFrame) -> np.ndarray:
    # Known from the next round's score; NaN for a map's latest visible round.
    nxt = rt.groupby("game_id").score_a.shift(-1)
    nr = rt.groupby("game_id")["round"].shift(-1)
    return np.where(nr == rt["round"] + 1, nxt - rt.score_a, np.nan)


def add_resid(rt: pd.DataFrame, expect) -> pd.DataFrame:
    """Team A's round wins earlier in this map beyond what both teams' buys predicted: in-map strength."""
    won = round_won(rt)
    p = expect.predict_proba(rt[EXPECT])[:, 1]
    res = pd.Series(np.where(np.isnan(won), 0.0, won - p), index=rt.index)
    return rt.assign(a_resid_prev=res.groupby(rt.game_id).cumsum() - res)


def rows(b: Bundle, states: pd.DataFrame, expect):
    sub = b.subset(states.game_id)
    x, _ = baseline.round_rows(sub, states)
    rt = add_resid(round_info(sub), expect).set_index(["game_id", "round"])
    d = dyn_features(sub, states)
    x = x.join(d.drop(columns=[c for c in d.columns if c in x.columns]))
    x = x.join(rt[["prior_logit", "has_prior", "a_resid_prev"] + FORM], on=["game_id", "round"])
    sg = np.where(x.side_a == "CT", 1.0, -1.0)
    x["ct_prior"] = sg * x.prior_logit
    x["ct_kd_prev"] = sg * x.a_kd_prev
    x["ct_upsets_prev"] = sg * x.a_upsets_prev
    x["ct_gun_wr_prev"] = np.where(sg > 0, x.a_gun_wr_prev, 1 - x.a_gun_wr_prev)
    x["ct_resid_prev"] = sg * x.a_resid_prev
    return x, rt


def income(ctr, won):
    # Team income for the next round: the win reward, or the loss bonus at the counter after this loss.
    return np.where(won, 3250.0, 1400.0 + 500.0 * np.clip(ctr - 1, 0, 4)) * 5


def cont_rows(rt: pd.DataFrame, a_won: int):
    X, done = baseline.next_round(rt, a_won)
    hp = rt.has_players.to_numpy(bool)
    # Rounds without player data carry zeros for the economy: missing, not broke.
    X = X.assign(**{c: np.where(hp, rt[c].to_numpy(float), np.nan) for c in ECON}, a_resid_prev=rt.a_resid_prev.to_numpy())
    X["a_cash_next"] = X.a_money + income(X.a_loss_ctr.to_numpy(), a_won == 1)
    X["b_cash_next"] = X.b_money + income(X.b_loss_ctr.to_numpy(), a_won == 0)
    return X, done


def survivor_models(x: pd.DataFrame, rti: pd.DataFrame, y_ct: np.ndarray, seed: int):
    """P(survivors of each side at the round's end | state now, who wins the round): multiclass 0..5."""
    end = x[["game_id", "round", "side_a"]].join(rti[["a_alive_end", "b_alive_end"]], on=["game_id", "round"])
    a_ct = (end.side_a == "CT").to_numpy()
    labels = {"ct": np.where(a_ct, end.a_alive_end, end.b_alive_end), "t": np.where(a_ct, end.b_alive_end, end.a_alive_end)}
    X = x[ROUND].assign(ct_won=y_ct)
    out = {}
    for side, lab in labels.items():
        ok = ~np.isnan(lab.astype(float))
        out[side] = xgb.XGBClassifier(objective="multi:softprob", num_class=6, random_state=seed, **SURV_PARAMS).fit(
            X[ok], np.clip(lab[ok], 0, 5).astype(int))
    return out


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    legal = legal_maps(train)
    rt = round_info(train.subset(legal))
    rt = rt[rt.map_a_wins.notna() & rt.a_wins.notna()].reset_index(drop=True)
    expect = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=8, random_state=seed)
    expect.fit(rt[EXPECT], rt.a_wins)
    rt = add_resid(rt, expect)
    Xs, ys = [], []
    for w in (1, 0):
        X, done = cont_rows(rt, w)
        keep = ((rt.a_wins == w) & np.isnan(done)).to_numpy()
        Xs.append(X[keep].assign(a_alive_end=rt.a_alive_end[keep].to_numpy(), b_alive_end=rt.b_alive_end[keep].to_numpy()))
        ys.append(rt.map_a_wins[keep])
    X, y = pd.concat(Xs), pd.concat(ys)
    cont_model = baseline.trees(seed).fit(X[CONT], y)
    grid_model = baseline.trees(seed).fit(X[CONT + SURV], y)
    st = map_states(train, legal)
    x, rti = rows(train, st, expect)
    x = x.join(rti[["a_wins"]], on=["game_id", "round"])
    y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    round_model = baseline.trees(seed, **ROUND_PARAMS).fit(x[ROUND], y)
    surv = survivor_models(x, rti, y, seed)
    return dict(cont_model=cont_model, grid_model=grid_model, round_model=round_model, expect=expect, surv=surv)


def start_states(ctx: Bundle, states: pd.DataFrame) -> pd.DataFrame:
    """The round-start state of every round in states (its round_start event), priced alongside to anchor the
    survivor adjustment."""
    ev = ctx.events[ctx.events.game_id.isin(set(states.game_id)) & (ctx.events.type == "round_start")]
    ev = ev.merge(states[["game_id", "round"]].drop_duplicates(), on=["game_id", "round"])
    ev = ev.sort_values("seq").groupby(["game_id", "round"]).head(1)
    return pd.DataFrame(dict(game_id=ev.game_id.to_numpy(), round=ev["round"].to_numpy(), step=0, kind="start",
                             event_seq=ev.seq.to_numpy(), ingame_ms=ev.ingame_ms.to_numpy(), elapsed=ev.elapsed.to_numpy()))


def survivor_ev(model, x: pd.DataFrame, grid: np.ndarray, pos: np.ndarray, a_won: int) -> np.ndarray:
    """E[M(survivors)] under P(survivors | state, A wins = a_won), survivors capped at those alive now."""
    a_ct = (x.side_a == "CT").to_numpy()
    X = x[ROUND].assign(ct_won=np.where(a_ct, a_won, 1 - a_won))
    ks = np.arange(6)[None, :]
    dist = {}
    for side in ("ct", "t"):
        p = model["surv"][side].predict_proba(X)
        p = np.where(ks <= x[f"{side}_alive"].to_numpy()[:, None], p, 0.0)
        dist[side] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    pa = np.where(a_ct[:, None], dist["ct"], dist["t"]); pb = np.where(a_ct[:, None], dist["t"], dist["ct"])
    return np.einsum("nk,nkj,nj->n", pa, grid[pos], pb)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    states = states.reset_index(drop=True)
    st0 = start_states(ctx, states)
    both = pd.concat([states, st0], ignore_index=True)
    xb, rt = rows(ctx, both, model["expect"])
    x, x0 = xb.iloc[:len(states)], xb.iloc[len(states):]
    p_ct = model["round_model"].predict_proba(x[ROUND])[:, 1]
    known = ((x.over == 1) & x.over_ct_won.notna()).to_numpy()
    p_ct = np.where(known, x.over_ct_won.to_numpy(), p_ct)
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rk = keys.unique()
    rr = rt.loc[rk].reset_index()
    pos = rk.get_indexer(keys)
    pos0 = rk.get_indexer(pd.MultiIndex.from_frame(x0[["game_id", "round"]]))
    m = {}
    for a_won in (1, 0):
        X, done = cont_rows(rr, a_won)
        base = model["cont_model"].predict_proba(X[CONT])[:, 1]
        grid = np.stack([np.stack([model["grid_model"].predict_proba(X.assign(a_alive_end=k, b_alive_end=j)[CONT + SURV])[:, 1]
                                   for j in range(6)], axis=1) for k in range(6)], axis=1)
        ev = survivor_ev(model, x, grid, pos, a_won)
        ev0 = np.full(len(rk), np.nan)
        ev0[pos0] = survivor_ev(model, x0, grid, pos0, a_won)
        # The survivor term only moves M by how far this state's survivor outlook is from the round start's.
        adj = np.where(np.isnan(ev0[pos]), 0.0, ev - ev0[pos])
        mm = np.clip(base[pos] + adj, 1e-4, 1 - 1e-4)
        m[a_won] = np.where(np.isnan(done[pos]), mm, done[pos])
    p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1] + (1 - p_a) * m[0]
