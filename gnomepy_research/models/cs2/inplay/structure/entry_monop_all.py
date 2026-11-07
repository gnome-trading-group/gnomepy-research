"""Structure variant 'monop-all' (see variants.py and NOTES.md)."""
import pandas as pd

from gnomepy_research.models.cs2.inplay.structure import core
from gnomepy_research.models.cs2.inplay.structure.variants import VARIANTS

CFG = VARIANTS["monop-all"]


def fit(train, asof: pd.Timestamp, seed: int):
    return core.fit(train, asof, seed, CFG)


def predict(model, ctx, states):
    return core.predict(model, ctx, states)
