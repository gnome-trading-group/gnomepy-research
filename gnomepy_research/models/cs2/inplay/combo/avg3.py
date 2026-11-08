"""Reference blend: the three winning entries fitted separately, their prices averaged on the logit scale."""
import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.inplay.econ.model import make
from gnomepy_research.models.cs2.inplay.players import entry as players
from gnomepy_research.models.cs2.inplay.structure import core
from gnomepy_research.models.cs2.inplay.structure.variants import VARIANTS

econ_fit, econ_predict = make()


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(train, asof: pd.Timestamp, seed: int):
    return dict(p=players.fit(train, asof, seed), s=core.fit(train, asof, seed, VARIANTS["slow-all"]), e=econ_fit(train, asof, seed))


def predict(model, ctx, states):
    z = (logit(players.predict(model["p"], ctx, states)) + logit(core.predict(model["s"], ctx, states))
         + logit(econ_predict(model["e"], ctx, states))) / 3
    return 1 / (1 + np.exp(-z))
