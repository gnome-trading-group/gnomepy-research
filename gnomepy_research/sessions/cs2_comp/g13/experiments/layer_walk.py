"""
Walk-forward evaluation of layer specs on dev rows: the layer refit at each dev week on earlier priced series, the
dev targets scored with the harness pool and stack (the copied e12 price stands in for the frozen baseline).

    python -m gnomepy_research.sessions.cs2_comp.g13.experiments.layer_walk --rows <dev_rows parquets...>
        --targets <targets parquet> [--features <features parquet>] [--first-dev-week 2026-03-30]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g13.experiments.evalkit import ci, fmt, ll, pool
from gnomepy_research.sessions.cs2_comp.g13.market import PATH_COLUMNS, path_features
from gnomepy_research.sessions.cs2_comp.g13.layer import Layer, LayerSpec, fallback, thin_market

GATE = ("lm", "z", "lm_thin", "z_thin")
SPECS = {
    "fallback only": None,
    "z only": LayerSpec(columns=("z",), offset=None),
    "lm+z (1h)": LayerSpec(price="m_last1h", columns=("lm", "z"), offset=None),
    "lm0+z": LayerSpec(price="m0", columns=("lm", "z"), offset=None),
    "lm+z+gate (1h)": LayerSpec(price="m_last1h", columns=GATE, offset=None),
    "lm+z+gate (2h)": LayerSpec(columns=GATE, offset=None),
    "lm+z+gate (2h) l2=5": LayerSpec(columns=GATE, offset=None, l2=5.0),
    "lm+z+gate (2h) l2=20": LayerSpec(columns=GATE, offset=None, l2=20.0),
    "lm+z+gate (6h)": LayerSpec(price="m_last6h", columns=GATE, offset=None),
    "+zh+zf instead of z": LayerSpec(columns=("lm", "zh", "zf", "lm_thin", "z_thin"), offset=None),
    "+drift6": LayerSpec(offset=None, columns=GATE + ("drift6",)),
    "+drift24": LayerSpec(offset=None, columns=GATE + ("drift24",)),
    "+lan": LayerSpec(offset=None, columns=GATE + ("z_lan", "lm_lan")),
    "+standin+idle": LayerSpec(offset=None, columns=GATE + ("standin", "idle")),
    "+z_unk+z_new": LayerSpec(offset=None, columns=GATE + ("z_unk", "z_new")),
    "+log_rank": LayerSpec(offset=None, columns=GATE + ("log_rank",)),
    "+ekf": LayerSpec(offset=None, columns=GATE + ("z_ekf",)),
    "+ekf+ekf_sd": LayerSpec(offset=None, columns=GATE + ("z_ekf", "z_sd", "lm_sd")),
    "ekf instead of z": LayerSpec(columns=("lm", "z_ekf", "lm_thin", "zekf_thin"), offset=None),
    "+chem": LayerSpec(offset=None, columns=GATE + ("ekf_chem",)),
    "e12 offset + gate, submitted": LayerSpec(),
    "e12 offset + gate l2=5": LayerSpec(columns=("lm_thin", "z_thin"), offset="z", l2=5.0),
    "e12 offset + lm + gate": LayerSpec(columns=("lm", "lm_thin", "z_thin"), offset="z"),
    "e12 offset + z_thin only": LayerSpec(columns=("z_thin",), offset="z"),
    "e12 offset + intercept only": LayerSpec(columns=(), offset="z"),
}


def refresh_paths(rows: pd.DataFrame, data_dir: str | None) -> pd.DataFrame:
    """Path features as `market.py` computes them now (rows saved earlier may predate a column)."""
    if data_dir is None:
        return rows
    paths = load(data_dir, "market/series_paths")
    rows = rows.drop(columns=[c for c in PATH_COLUMNS if c in rows])
    return rows.merge(path_features(paths[paths.match_id.isin(rows.match_id)]), on="match_id", how="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--data", help="bundle dir: recompute the path features of the rows from market/series_paths")
    ap.add_argument("--features")
    ap.add_argument("--first-dev-week", default="2026-03-30")
    ap.add_argument("--specs", help="'|'-separated names to run (default: all)")
    a = ap.parse_args()
    rows = pd.concat([pd.read_parquet(p) for p in a.rows], ignore_index=True)
    rows = rows[rows.y.notna()].reset_index(drop=True)
    rows["week_cutoff"], rows["known"] = pd.to_datetime(rows.week_cutoff), pd.to_datetime(rows.known)
    rows = refresh_paths(rows, a.data)
    if a.features:
        feats = pd.read_parquet(a.features)
        feats["week_cutoff"] = pd.to_datetime(feats.week_cutoff)
        rows = rows.merge(feats, on=["match_id", "week_cutoff"], how="left")
    targets = pd.read_parquet(a.targets)
    dev = rows[(rows.week_cutoff >= pd.Timestamp(a.first_dev_week)) & rows.match_id.isin(targets.match_id)].copy()
    months = pd.to_datetime(dev.time).dt.to_period("M").astype(str).to_numpy()
    y, m0 = dev.y.to_numpy(float), dev.m0.to_numpy()
    e12 = 1 / (1 + np.exp(-dev.z_e12.to_numpy()))
    base_m, base_pool = ll(m0, y), ll(pool(m0, e12, y, months), y)
    thin = thin_market(dev) > 0
    print(f"dev rows {len(dev)}  market LL {base_m.mean():.4f}  e12 alone-market {fmt(ci(ll(e12, y) - base_m))}  "
          f"pool(e12)-market {fmt(ci(base_pool - base_m))}")
    specs = SPECS if not a.specs else {k: v for k, v in SPECS.items() if k in a.specs.split("|")}
    for name, spec in specs.items():
        if spec is not None and any(c in ("z_ekf", "z_sd", "lm_sd", "zekf_thin", "ekf_chem") for c in spec.columns) and "z_ekf" not in dev:
            continue
        q = np.empty(len(dev))
        for c in sorted(dev.week_cutoff.unique()):
            week = (dev.week_cutoff == c).to_numpy()
            if spec is None:
                q[week] = fallback(dev[week], LayerSpec())
            else:
                q[week] = Layer.fit(rows[(rows.known < c) & (rows.week_cutoff < c)], spec).predict(dev[week])
        p, s = ll(pool(m0, q, y, months), y), ll(pool(m0, q, y, months, [e12]), y)
        per = "  ".join(f"{m}:{(p - base_m)[months == m].mean():+.4f}/{(base_pool - base_m)[months == m].mean():+.4f}" for m in np.unique(months))
        print(f"{name:28s} q-m {fmt(ci(ll(q, y) - base_m))}  pool-m {fmt(ci(p - base_m))}  stack-bpool {fmt(ci(s - base_pool))}  "
              f"thin pool-m {(p - base_m)[thin].mean():+.4f} (baseline {(base_pool - base_m)[thin].mean():+.4f}, n={thin.sum()})")
        print(f"{'':28s} per month layer/baseline: {per}")


if __name__ == "__main__":
    main()
