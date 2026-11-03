"""g08 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g08.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--no-e12]

Per distinct week_cutoff: the recurrent skill state (HLTV + FACEIT maps and past series' kickoff prices) read out at
the cutoff, a logistic layer fitted on past series, and an even logit average with the frozen e12 price for the same
week (e12's own weekly skill fits, run unchanged). Each side is priced with its announced five."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g08.e12ref.predict import predict as e12_predict
from gnomepy_research.sessions.cs2_comp.g08.model import combine, state_prices


def predict(data_dir: str | Path, targets: pd.DataFrame, use_e12: bool = True, verbose: bool = False) -> pd.DataFrame:
    s = state_prices(data_dir, targets, verbose=verbose)
    if use_e12:
        e = e12_predict(data_dir, targets, faceit=True, verbose=verbose)[["match_id", "q"]].rename(columns={"q": "q_e12"})
        s = s.merge(e, on="match_id", how="left")
        s["q"] = combine(s.q_state.to_numpy(), s.q_e12.fillna(s.q_state).to_numpy())
    else:
        s["q"] = s.q_state
    s["fit_cutoff"] = pd.to_datetime(s.week_cutoff)
    return s.drop(columns=["week_cutoff"])


def read_targets(path: str | Path, weeks: str | None = None) -> pd.DataFrame:
    t = pd.read_parquet(path)
    wc = pd.to_datetime(t.week_cutoff)
    t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if weeks:
        t = t[t.week_cutoff.isin(pd.to_datetime(weeks.split(",")))]
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--no-e12", dest="e12", action="store_false")
    a = ap.parse_args()
    out = predict(a.data, read_targets(a.targets, a.weeks), use_e12=a.e12, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
