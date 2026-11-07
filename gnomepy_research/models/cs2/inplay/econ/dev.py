"""Development helper: build and cache the train/test state rows of one walk-forward month so model variants can be
compared in seconds instead of re-running the harness. Not used by the entry."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import harness
from ..data import Bundle
from ..features import legal_maps, map_states
from .model import round_info, state_rows

DATA = "/Users/mprey/gnome-research-work/cs2/comp3/data"


def build(month: str, out: Path):
    b = Bundle.load(DATA)
    asof, train, ctx, states, y = harness.month_split(b, month)
    legal = legal_maps(train)
    st = map_states(train, legal)
    rt_tr = round_info(train, set(st.game_id))
    x_tr = state_rows(train, st, rt_tr)
    rt_te = round_info(ctx, set(states.game_id))
    x_te = state_rows(ctx, states, rt_te)
    with open(out, "wb") as fh:
        pickle.dump(dict(train=train, legal=legal, x_tr=x_tr, rt_tr=rt_tr, x_te=x_te, rt_te=rt_te, y=y, states=states), fh)


if __name__ == "__main__":
    build(sys.argv[1], Path(sys.argv[2]))
