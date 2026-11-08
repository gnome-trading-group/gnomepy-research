"""Combination of the three competition-3 winners.

P(A wins the map) = P(A wins this round | state) * M(A wins) + P(A loses) * M(A loses), with
- the prior: the players entry's roster / stacked-market prior in place of the kickoff market or Elo, everywhere the
  structure entry reads a prior (round-win chain, regression, round model);
- the map value M: the structure entry's round-win chain blended with its regression (one-step economy lookahead);
- optionally the econ entry's boosted economy correction on top of M, its base margin now the structure map value
  after each round outcome instead of the benchmark's score-only map value."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from ..data import Bundle
from ..econ import model as E
from ..features import legal_maps, map_states
from ..structure import core
from ..structure.variants import VARIANTS
from .prior import RosterPrior


@dataclass
class Combo:
    structure: core.Config = field(default_factory=lambda: VARIANTS["slow-all"])
    roster_prior: bool = True
    econ: bool = False
    map_price: bool = False
    econ_cfg: dict = field(default_factory=lambda: dict(E.DEFAULT))


def map_values(model, c: pd.DataFrame) -> dict:
    w = model["cfg"].w_dp
    return {1: (w * c.d1 + (1 - w) * c.r1).to_numpy(), 0: (w * c.d0 + (1 - w) * c.r0).to_numpy()}


def fit_econ(train: Bundle, smodel, seed: int, cfg: dict):
    """The econ correction trained on every in-round state of the training maps, conditioned on the realised round
    outcome, on top of the structure map value for that outcome."""
    legal = legal_maps(train)
    st = map_states(train, legal)
    rt = E.round_info(train, set(st.game_id))
    x = E.state_rows(train, st, rt)
    m = map_values(smodel, core.components(smodel, train, st))
    keys = pd.MultiIndex.from_frame(x[["game_id", "round"]])
    a_wins = rt.a_wins.reindex(keys).to_numpy()
    map_wins = rt.map_a_wins.reindex(keys).to_numpy()
    F, base, y, nk = [], [], [], []
    for o in (1, 0):
        sel = a_wins == o
        e, _, done = E.eco_rows(x[sel], rt, o)
        keep = np.isnan(done)
        F.append(e[keep])
        base.append(E.logit(m[o][sel][keep]))
        y.append(map_wins[sel][keep])
        nk.append((x.game_id.to_numpy()[sel][keep], x["round"].to_numpy()[sel][keep] + 1))
    F, base, y = pd.concat(F), np.concatenate(base), np.concatenate(y)
    if cfg["soft"]:
        post = E.Model(cfg, seed).post_values(train, legal, set(x.game_id))
        gid, rnd = (np.concatenate([k[i] for k in nk]) for i in (0, 1))
        pv = post.reindex(pd.MultiIndex.from_arrays([gid, rnd])).to_numpy()
        ok = ~np.isnan(pv) & ~np.isnan(y)
        F, base, y = F[ok], base[ok], cfg["soft"] * pv[ok] + (1 - cfg["soft"]) * y[ok]
    params = dict(objective="binary:logistic", n_jobs=8, random_state=seed, **cfg["eco_trees"])
    return xgb.XGBRegressor(**params).fit(F, y, base_margin=base)


def fit(train: Bundle, asof: pd.Timestamp, seed: int, cfg: Combo):
    prior = RosterPrior(train, asof, cfg.map_price) if cfg.roster_prior else None
    smodel = core.fit(train, asof, seed, cfg.structure, prior)
    eco = fit_econ(train, smodel, seed, cfg.econ_cfg) if cfg.econ else None
    if prior is not None:
        prior.release_training()
    return dict(s=smodel, eco=eco)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    s = model["s"]
    c = core.components(s, ctx, states)
    m = map_values(s, c)
    if model["eco"] is not None:
        rt = E.round_info(ctx, set(states.game_id))
        x = E.state_rows(ctx, states, rt)
        for o in (1, 0):
            e, _, done = E.eco_rows(x, rt, o)
            m[o] = np.where(np.isnan(done), model["eco"].predict(e, base_margin=E.logit(m[o])), done)
    p_a = c.p_a.to_numpy()
    return p_a * m[1] + (1 - p_a) * m[0]
