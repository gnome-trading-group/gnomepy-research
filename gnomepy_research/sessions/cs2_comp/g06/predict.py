"""
g06: e12's player-skill price, corrected by the announced fives' FACEIT matchmaking form.

    python -m gnomepy_research.sessions.cs2_comp.g06.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13]

    logit(q) = logit(e12) + GAMMA * (form_A - form_B)

e12 is refitted at each week_cutoff (HLTV and HLTV+FACEIT skill fits and its fixed logistic layer) and prices each side
with its announced five. A five's form is the mean over its players of a shrunk, recency-weighted win rate in their own
FACEIT matchmaking and hub games before the cutoff (`mm_form`). GAMMA is the form's weight relative to the skill price
in a logistic regression on every HLTV bo3 series of Jan 2025 - Feb 2026, all of it before the first tune cutoff.

q carries no market information: the harness pools it with the kickoff price, and pre-kickoff price paths added
nothing once a skill price was pooled (see the README).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g06.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g06.e12.predict import LAYER_FACEIT, LAYER_HLTV, design, rows
from gnomepy_research.sessions.cs2_comp.g06.history import logit, naive
from gnomepy_research.sessions.cs2_comp.g06.mm_form import MatchmakingForm

GAMMA = 0.925


def targets_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The e12 target shape: teams, cutoff, announced fives and stand-in counts."""
    t = pd.DataFrame({"match_id": df.match_id.to_numpy(), "team_a": df.team_a_id.astype(np.int64).to_numpy(),
                      "team_b": df.team_b_id.astype(np.int64).to_numpy(), "week_cutoff": df.week_cutoff.to_numpy()})
    for side in ("a", "b"):
        t[f"lineup_{side}"] = df[f"team_{side}_lineup_ids"].to_numpy()
        t[f"sub_{side}"] = [0 if v is None else len(v) for v in df[f"team_{side}_standin_ids"]]
    return t


def e12_rows(data_dir: str | Path, series: pd.DataFrame, faceit: bool = True) -> pd.DataFrame:
    """e12's prices (HLTV-only, and with FACEIT the two-fit layer) and per-side context for each series, from the fits
    at its own week_cutoff."""
    t = targets_frame(series)
    inputs = Inputs(data_dir, t.week_cutoff.max(), faceit)
    h, f = rows(inputs, t, faceit)
    Xh = design(h, None)[list(LAYER_HLTV)]
    out = pd.DataFrame({"match_id": h.match_id.to_numpy(),
                        "e12h": expit(Xh.to_numpy() @ np.array(list(LAYER_HLTV.values()))),
                        "gap": h.gap.to_numpy(float)})
    for c in ("thin", "new", "idle", "rank", "standin"):
        out[f"{c}_a"] = h[f"{c}_a"].to_numpy(float)
        out[f"{c}_b"] = h[f"{c}_b"].to_numpy(float)
    if faceit:
        Xf = design(h, f)[list(LAYER_FACEIT)]
        out["e12f"] = expit(Xf.to_numpy() @ np.array(list(LAYER_FACEIT.values())))
        out["gap_f"] = f.gap.to_numpy(float)
        out["linked_a"] = f.linked_a.to_numpy(float)
        out["linked_b"] = f.linked_b.to_numpy(float)
    return out


def predict(data_dir: str | Path, targets: pd.DataFrame) -> pd.DataFrame:
    e = e12_rows(data_dir, targets)
    form = MatchmakingForm(data_dir, targets.week_cutoff.max())
    d = targets[["match_id", "week_cutoff", "team_a_lineup_ids", "team_b_lineup_ids"]].merge(e, on="match_id")
    d["form"] = form.diff(d.team_a_lineup_ids, d.team_b_lineup_ids, d.week_cutoff)
    q = expit(logit(d.e12f) + GAMMA * d.form.to_numpy())
    return pd.DataFrame({"match_id": d.match_id.to_numpy(), "q": q, "fit_cutoff": d.week_cutoff.to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets.reset_index(drop=True))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
