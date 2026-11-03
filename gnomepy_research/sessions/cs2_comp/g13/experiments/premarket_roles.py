"""
Does a five missing its usual IGL or AWPer, or the brief's thin flag, add to e12's HLTV price on the year before the
market data starts? Month-held-out logistic on e12's HLTV-only price alone against the price plus one column.

    python -m gnomepy_research.sessions.cs2_comp.g13.experiments.premarket_roles --data <bundle>
        --first 2025-03-31 --last 2026-03-23
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g13.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g13.e12.features import feature_rows
from gnomepy_research.sessions.cs2_comp.g13.e12.model import series_price
from gnomepy_research.sessions.cs2_comp.g13.e12.predict import CONFIG_HLTV
from gnomepy_research.sessions.cs2_comp.g13.experiments.evalkit import ci, fit, fmt, ll
from gnomepy_research.sessions.cs2_comp.g13.experiments.features import context_rows
from gnomepy_research.sessions.cs2_comp.g13.rows import WEEK, history_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    a = ap.parse_args()
    cutoffs = list(pd.date_range(a.first, a.last, freq=WEEK))
    until = cutoffs[-1] + WEEK
    inputs = Inputs(a.data, until, faceit=False)
    frame = history_frame(inputs, cutoffs)
    h = feature_rows(inputs, frame, CONFIG_HLTV, False)
    ctx = context_rows(a.data, frame, until)
    p = np.clip(series_price(h.gap.to_numpy()), 1e-4, 1 - 1e-4)
    zh = np.log(p / (1 - p))
    y = h.y.to_numpy(float)
    months = pd.to_datetime(h.time).dt.to_period("M").astype(str).to_numpy()
    ones = np.ones(len(y))

    def held_out(X: np.ndarray) -> np.ndarray:
        out = np.empty(len(y))
        for m in np.unique(months):
            tr = months != m
            out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fit(X[tr], y[tr])))
        return ll(out, y)

    base = held_out(np.column_stack([zh, ones]))
    print(f"n={len(y)} series over {len(np.unique(months))} months; e12 HLTV price recalibrated: log loss {base.mean():.4f}")
    for name, col in {"igl_missing": ctx.igl_missing.to_numpy(), "awp_missing": ctx.awp_missing.to_numpy(),
                      "thin_brief x price": zh * ctx.thin_brief.to_numpy()}.items():
        hh = held_out(np.column_stack([zh, col, ones]))
        affected = col != 0
        on = f"{(hh - base)[affected].mean():+.4f}" if affected.sum() > 20 else "n/a"
        print(f"+{name:20s} vs price alone {fmt(ci(hh - base))}; on the {affected.sum()} affected series {on}")


if __name__ == "__main__":
    main()
