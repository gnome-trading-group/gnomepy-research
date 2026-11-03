"""e12 prices for every HLTV bo3 series of the weeks in [start, end), with the fives announced for it, so that
corrections can be weighted against e12 itself on pre-dev outcomes. python -m ...g10.dev.pree12 --data D --start S --end E"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g10.e12_data import Inputs
from gnomepy_research.sessions.cs2_comp.g10.e12_predict import LAYER_FACEIT, design, rows
from gnomepy_research.sessions.cs2_comp.g10.rows import monday


def main() -> None:
    threadpool_limits(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    inputs = Inputs(a.data, end, True)
    s = inputs.series
    s = s[(s.bo_type == 3) & (s.time >= start) & (s.time < end)].copy()
    t = pd.DataFrame({"match_id": s.match_id.to_numpy(), "team_a": s.team_a.to_numpy(), "team_b": s.team_b.to_numpy(),
                      "week_cutoff": monday(s.time).to_numpy(), "lineup_a": s.lineup_a.to_numpy(), "lineup_b": s.lineup_b.to_numpy(),
                      "sub_a": 0, "sub_b": 0})
    h, f = rows(inputs, t, True, True)
    q = expit(design(h, f)[list(LAYER_FACEIT)].to_numpy() @ np.array(list(LAYER_FACEIT.values())))
    pd.DataFrame({"match_id": h.match_id.to_numpy(), "q_e12": q}).to_parquet(
        Path(os.environ.get("OUTPUT_DIR", ".")) / f"pree12_{a.start}.parquet", index=False)


if __name__ == "__main__":
    main()
