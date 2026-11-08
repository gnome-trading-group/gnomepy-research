"""Structure entry: round model x map value, the map value from a round-win chain (see NOTES.md).

The configuration is variants.VARIANTS["slow-all"]; every other scored configuration is listed there and in NOTES.md."""
import pandas as pd

from gnomepy_research.models.cs2.inplay.structure import core
from gnomepy_research.models.cs2.inplay.structure.variants import VARIANTS

CFG = VARIANTS["slow-all"]


def fit(train, asof: pd.Timestamp, seed: int):
    return core.fit(train, asof, seed, CFG)


def predict(model, ctx, states):
    return core.predict(model, ctx, states)
