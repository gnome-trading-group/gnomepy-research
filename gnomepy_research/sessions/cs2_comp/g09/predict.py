"""
g09 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g09.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13]

Per distinct week_cutoff: one chronological pass builds point-in-time features for every past series with a market
(snapshotted at its own week start) and for the targets (at the cutoff); a ridge logistic layer fitted on the past
series then prices the targets. See layer.py for the features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g09.core import EloConfig, build_rows, naive
from gnomepy_research.sessions.cs2_comp.g09.layer import fit_predict

# Three ratings averaged, all tuned by tune_elo.py on HLTV bo3 series from Jun 2025 to Mar 2026 (before the first dev
# cutoff): HLTV maps only; HLTV and FACEIT maps; and the same fully re-tuned. The full re-tune scores best on the
# tuning window but not on Polymarket series, so no single one is trusted.
MAIN_ELO = EloConfig(k_faceit=1.0, faceit_stats=True)
EXTRA_ELO = {
    "h": EloConfig(),
    "t": EloConfig(k_new=1.6072, n_settle=48.0, new_gap=1.16, rank_scale=3.35, margin=0.9, k_faceit=0.4444, faceit_offset=1.5,
                   ind_weight=13.5, ind_alpha=0.0029333, faceit_stats=True),
}


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    out = []
    for cutoff, t in targets.groupby("week_cutoff"):
        train, tgt = build_rows(data_dir, pd.Timestamp(cutoff), t, elo_cfg=MAIN_ELO, extra_elo=EXTRA_ELO)
        q = fit_predict(train, tgt)
        out.append(pd.DataFrame({"match_id": tgt.match_id.to_numpy(), "q": q, "fit_cutoff": pd.Timestamp(cutoff)}))
        if verbose:
            print(f"{pd.Timestamp(cutoff).date()}: {len(train)} past series, {len(tgt)} targets", flush=True)
    return pd.concat(out, ignore_index=True)


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
    out = predict(a.data, targets, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
