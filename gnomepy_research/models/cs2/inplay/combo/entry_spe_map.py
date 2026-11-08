"""Combination variant spe-map (variants.py)."""
import pandas as pd

from gnomepy_research.models.cs2.inplay.combo import model
from gnomepy_research.models.cs2.inplay.combo.variants import VARIANTS

CFG = VARIANTS["spe-map"]


def fit(train, asof: pd.Timestamp, seed: int):
    return model.fit(train, asof, seed, CFG)


def predict(m, ctx, states):
    return model.predict(m, ctx, states)
