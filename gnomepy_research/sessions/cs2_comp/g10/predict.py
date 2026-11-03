"""
g10 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g10.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--parts <dir>]

The e12 skill price (HLTV and FACEIT player-skill fits, frozen logistic layer) moved by three team-level corrections
the player fit cannot see, each a shrunken round residual against the HLTV-only skill fit at the same cutoff:

  map pool    per (team, map) residuals, priced through a bo3 veto in which each side bans its worst maps and picks
              its best;
  setting     per (team, LAN or online) residuals, for the setting the series is played in;
  form        per team residuals over the last few weeks (21-day half-life).

    logit q = logit(q_e12) + sum_k w_k * (logit(q_k) - logit(q_hltv))

The weights w_k were fitted on 11,172 bo3 series from Jan 2025 to Mar 2026 (weekly point-in-time fits, outcomes
only, no market), so they are point-in-time for every dev and confirmation target. The market's price path is not an
input: on the dev months none of its summaries (two-hour average, opening price, liquidity, staleness, market age)
added anything once the e12 skill price was pooled with the kickoff price (see README).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g10.e12_data import Inputs
from gnomepy_research.sessions.cs2_comp.g10.e12_model import SkillFit, series_price
from gnomepy_research.sessions.cs2_comp.g10.e12_predict import CONFIG_HLTV, LAYER_FACEIT, design, rows
from gnomepy_research.sessions.cs2_comp.g10.mapveto import MapConfig, MapModel, load_maps

MAP_CONFIG = MapConfig(half_life_days=240.0, prior_rounds=240.0)
SETTING_CONFIG = MapConfig(half_life_days=240.0, prior_rounds=120.0)
WEIGHTS = {"map": 0.166, "setting": 0.09, "form": 0.13}


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def components(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Per target: q_e12, the HLTV-only skill price and the three corrected prices."""
    t = targets.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b"})[["match_id", "team_a", "team_b", "week_cutoff", "is_lan"]].copy()
    for side in ("a", "b"):
        t[f"lineup_{side}"] = targets[f"team_{side}_lineup_ids"].to_numpy()
        t[f"sub_{side}"] = [0 if v is None else len(v) for v in targets[f"team_{side}_standin_ids"]]
    until = pd.Timestamp(t.week_cutoff.max())
    inputs = Inputs(data_dir, until, True)
    h, f = rows(inputs, t.drop(columns="is_lan"), True, verbose)
    q_e12 = expit(design(h, f)[list(LAYER_FACEIT)].to_numpy() @ np.array(list(LAYER_FACEIT.values())))
    out = pd.DataFrame({"match_id": h.match_id.to_numpy(), "week_cutoff": pd.to_datetime(h.week_cutoff).to_numpy(),
                        "gap": h.gap.to_numpy(), "q_e12": q_e12})
    out = out.merge(t[["match_id", "team_a", "team_b", "is_lan"]], on="match_id")
    maps = load_maps(data_dir, until)
    parts = []
    for c in sorted(out.week_cutoff.unique()):
        c = pd.Timestamp(c)
        t0 = time.time()
        wk = out[out.week_cutoff == c].copy()
        fit = SkillFit.fit(inputs, c, CONFIG_HLTV, False)
        mm = MapModel(maps, fit, c, MAP_CONFIG)
        ms = MapModel(maps, fit, c, SETTING_CONFIG)
        trip = list(zip(wk.gap, wk.team_a.astype(int), wk.team_b.astype(int), wk.is_lan.astype(int)))
        wk["q_hltv"] = series_price(wk.gap.to_numpy())
        wk["q_map"] = [mm.series_price(g, a, b) for g, a, b, _ in trip]
        wk["q_setting"] = series_price(np.array([ms.lan_gap(g, a, b, lan) for g, a, b, lan in trip]))
        wk["q_form"] = series_price(np.array([ms.offset_gap(g, a, b, "form") for g, a, b, _ in trip]))
        parts.append(wk)
        if verbose:
            print(f"{c:%Y-%m-%d} corrections {time.time() - t0:.1f}s series {len(wk)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def combine(comp: pd.DataFrame, weights: dict[str, float] = WEIGHTS) -> np.ndarray:
    z = _logit(comp.q_e12)
    base = _logit(comp.q_hltv)
    for k, w in weights.items():
        z = z + w * (_logit(comp[f"q_{k}"]) - base)
    return expit(z)


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False, parts_dir: str | None = None) -> pd.DataFrame:
    comp = components(data_dir, targets, verbose)
    if parts_dir:
        Path(parts_dir).mkdir(parents=True, exist_ok=True)
        comp.to_parquet(Path(parts_dir) / f"components_{comp.week_cutoff.min():%Y%m%d}.parquet", index=False)
    return pd.DataFrame({"match_id": comp.match_id.to_numpy(), "q": combine(comp), "fit_cutoff": comp.week_cutoff.to_numpy()})


def main() -> None:
    # Batch containers see every host core but get a share of them; one BLAS thread per process avoids thrashing.
    threadpool_limits(1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--parts", help="directory to also save the per-target components in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, verbose=True, parts_dir=a.parts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
