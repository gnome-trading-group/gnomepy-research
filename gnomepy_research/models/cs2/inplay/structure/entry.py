"""Structure entry: round model x (DP map value blended with a regression map value). See NOTES.md."""
import pandas as pd

from gnomepy_research.models.cs2.inplay.structure import core

CFG = core.Config()


def fit(train, asof: pd.Timestamp, seed: int):
    return core.fit(train, asof, seed, CFG)


def predict(model, ctx, states):
    return core.predict(model, ctx, states)
