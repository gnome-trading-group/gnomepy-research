"""
g12 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g12.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--no-faceit]

Cheap online raters (raters.py) are replayed over the whole history once; each week's stack (stack.py) is fitted on
the HLTV series known before that week's cutoff, each priced by the raters as they stood at its own week's cutoff, and
then prices the week's targets from their announced fives and their own price path up to kickoff.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import five_tuple, load_inputs, load_target_market, naive
from gnomepy_research.sessions.cs2_comp.g12.raters import EXTRA, RaterConfig, Replay
from gnomepy_research.sessions.cs2_comp.g12.stack import Stack

STACK = dict(use1=("gauss",) + tuple(name for name, _ in EXTRA), model_only=True)

HIST_COLS = ["match_id", "team_a", "team_b", "week_cutoff", "five_a", "five_b", "sub_a", "sub_b", "bo_type", "is_lan",
             "tier", "y", "known", "time"]


def target_rows(targets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": targets.match_id.to_numpy(), "team_a": targets.team_a_id.astype("int64").to_numpy(),
        "team_b": targets.team_b_id.astype("int64").to_numpy(), "week_cutoff": naive(targets.week_cutoff).to_numpy(),
        "five_a": [five_tuple(v) for v in targets.team_a_lineup_ids], "five_b": [five_tuple(v) for v in targets.team_b_lineup_ids],
        "sub_a": [0 if v is None else len(v) for v in targets.team_a_standin_ids],
        "sub_b": [0 if v is None else len(v) for v in targets.team_b_standin_ids],
        "bo_type": targets.bo_type.to_numpy(), "is_lan": targets.is_lan.to_numpy(), "tier": targets.event_tier.to_numpy(),
        "time": naive(targets.match_time).to_numpy(), "target": True})


def panel(data_dir: str | Path, targets: pd.DataFrame, faceit: bool = True, cfg: RaterConfig = RaterConfig(extra=EXTRA)) -> pd.DataFrame:
    until = pd.Timestamp(naive(targets.week_cutoff).max())
    inp = load_inputs(data_dir, until, faceit)
    hist = inp.series[inp.series.week_cutoff <= until][HIST_COLS].assign(target=False)
    tr = target_rows(targets)
    p = Replay(inp, cfg).run(pd.concat([hist, tr], ignore_index=True))
    hm = inp.market.drop(columns=["team_a", "team_b"])
    tm = load_target_market(data_dir, targets.match_id)
    return pd.concat([p[~p.target].merge(hm, on="match_id", how="left"), p[p.target].merge(tm, on="match_id", how="left")],
                     ignore_index=True)


def predict(data_dir: str | Path, targets: pd.DataFrame, faceit: bool = True, verbose: bool = False) -> pd.DataFrame:
    p = panel(data_dir, targets, faceit)
    hist, tg = p[~p.target], p[p.target]
    out = []
    for c in sorted(tg.week_cutoff.unique()):
        c = pd.Timestamp(c)
        st = Stack(**STACK).fit(hist, c)
        d = tg[tg.week_cutoff == c]
        out.append(pd.DataFrame({"match_id": d.match_id.to_numpy(), "q": st.predict(d), "fit_cutoff": c}))
        if verbose:
            print(f"{c:%Y-%m-%d} train {st.n_train} series {len(d)}", flush=True)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    if a.out.startswith("@"):
        a.out = str(Path(os.environ["OUTPUT_DIR"]) / a.out[1:])
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, a.faceit, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
