"""
Ceiling check: the harness pool (month-held-out, on dev) with a candidate added as an extra input next to
logit(m0) and logit(e12). What a walk-forward layer could deliver at best from that candidate.

    python -m gnomepy_research.sessions.cs2_comp.g13.experiments.oracle --rows <dev_rows parquets...>
        --features <features parquet> --targets <targets parquet>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g13.experiments.evalkit import ci, fit, fmt, ll, logit
from gnomepy_research.sessions.cs2_comp.g13.market import PATH_COLUMNS, path_features

CANDIDATES = [
    ["lm2"], ["lm1"], ["lm6"], ["lm0_thin", "z_thin"], ["lm2", "lm2_thin", "z_thin"], ["ze"], ["z_sd"], ["chem"],
    ["z_s20"], ["z_s45"], ["z_s90"], ["z_nr300"], ["z_s45_d"], ["z_s90_d"],
    ["mp_edge"], ["mp_pick"], ["mp_opp"], ["mp_decider"], ["mp_depth"], ["mp_h2h"], ["mp_h2h_n"],
    ["drift6"], ["drift24"], ["z_lan", "lm0_lan"], ["log_rank"], ["standin", "idle"], ["z_unk"], ["z_new"], ["z_linked"],
    ["lm0_thinb", "z_thinb"], ["z_thinb"], ["lm0_thinb"], ["z_days"], ["lm0_days"], ["z_days", "lm0_days"],
    ["z_tier0", "lm0_tier0"], ["igl_missing"], ["awp_missing"], ["igl_missing", "awp_missing"],
]


def held_out(X: np.ndarray, y: np.ndarray, months: np.ndarray) -> np.ndarray:
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fit(X[tr], y[tr])))
    return ll(out, y)


def columns(d: pd.DataFrame) -> dict[str, np.ndarray]:
    thin = ((d.n_prints < 50) | (d.flat > 0)).to_numpy(float)
    z = d.z_e12.to_numpy()
    lm0, lm2 = logit(d.m0), logit(d.m_last2h)
    lan = d.is_lan.to_numpy(float)
    cols = {"lm0": lm0, "z": z, "lm1": logit(d.m_last1h), "lm2": lm2, "lm6": logit(d.m_last6h), "ze": d.z_ekf.to_numpy(),
            "lm0_thin": lm0 * thin, "z_thin": z * thin, "lm2_thin": lm2 * thin, "z_sd": z * d.ekf_sd.to_numpy(),
            "chem": d.ekf_chem.to_numpy(), "drift6": lm2 - logit(d.m_at6h), "drift24": lm2 - logit(d.m_at24h),
            "z_lan": z * lan, "lm0_lan": lm0 * lan, "log_rank": d.log_rank.to_numpy(), "standin": d.standin.to_numpy(),
            "idle": d.idle.to_numpy(), "z_unk": z * d.thin.to_numpy(float), "z_new": z * d.new.to_numpy(float),
            "z_linked": z * d.linked.to_numpy(float)}
    for k in ("z_s20", "z_s45", "z_s90", "z_nr300"):
        cols[k] = d[k].to_numpy()
        cols[k + "_d"] = d[k].to_numpy() - z
    for k in [c for c in d.columns if c.startswith("mp_")]:
        cols[k] = d[k].fillna(0).to_numpy()
    if "thin_brief" in d:
        thin_b = d.thin_brief.to_numpy(float)
        days = (pd.to_datetime(d.time) - pd.to_datetime(d.week_cutoff)).dt.total_seconds().to_numpy() / 86400
        days = days - days.mean()
        tier0 = (d.tier == 0).to_numpy(float)
        cols.update({"lm0_thinb": lm0 * thin_b, "z_thinb": z * thin_b, "z_days": z * days, "lm0_days": lm0 * days,
                     "z_tier0": z * tier0, "lm0_tier0": lm0 * tier0, "igl_missing": d.igl_missing.fillna(0).to_numpy(),
                     "awp_missing": d.awp_missing.fillna(0).to_numpy()})
    return cols


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
    ap.add_argument("--features", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--data", help="bundle dir: recompute the path features of the rows from market/series_paths")
    a = ap.parse_args()
    rows = pd.concat([pd.read_parquet(p) for p in a.rows], ignore_index=True)
    rows["week_cutoff"] = pd.to_datetime(rows.week_cutoff)
    rows = refresh_paths(rows, a.data)
    feats = pd.read_parquet(a.features)
    feats["week_cutoff"] = pd.to_datetime(feats.week_cutoff)
    rows = rows.merge(feats, on=["match_id", "week_cutoff"], how="left")
    targets = pd.read_parquet(a.targets)
    d = rows[rows.match_id.isin(targets.match_id) & rows.y.notna()].copy()
    y, months = d.y.to_numpy(float), pd.to_datetime(d.time).dt.to_period("M").astype(str).to_numpy()
    cols = columns(d)
    base_m = ll(d.m0, y)
    ones = np.ones(len(y))
    base = held_out(np.column_stack([cols["lm0"], cols["z"], ones]), y, months)
    print(f"n={len(d)} market {base_m.mean():.4f} baseline pool-m {fmt(ci(base - base_m))}")
    for names in CANDIDATES:
        if not all(n in cols for n in names):
            continue
        h = held_out(np.column_stack([cols["lm0"], cols["z"]] + [cols[n] for n in names] + [ones]), y, months)
        print(f"+{'+'.join(names):28s} pool-m {fmt(ci(h - base_m))}  vs baseline pool {fmt(ci(h - base))}")


if __name__ == "__main__":
    main()
