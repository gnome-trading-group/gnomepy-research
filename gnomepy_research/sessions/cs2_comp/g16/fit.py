"""
Fit skill-filter hyperparameters by one-step-ahead log loss on HLTV maps, before the first target week.

    python -m gnomepy_research.sessions.cs2_comp.g16.fit --data <bundle> --until 2026-03-30 --score-from 2025-01-01
        --params sigma0,tau,rho,faceit_weight,off_low,off_mid,off_high [--init '<json>'] [--maxfev 150] [--out <dir>]
    python -m gnomepy_research.sessions.cs2_comp.g16.fit ... --evaluate '[{"name": ..., "sigma0": ...}, ...]'

`--evaluate` scores each listed set of values once instead of optimising, so a grid can be spread over Batch tasks.

Every map before --until is consumed; only HLTV maps from --score-from count in the loss, each priced with the fives
that played, from the state just before it. The targets are HLTV series, so HLTV maps are the right domain; FACEIT
maps only earn their weight through what they add there. Positive parameters are optimised on the log scale.

Division offsets are grouped: low = entry, open; mid = intermediate, main; high = advanced, pro, qualifier, other.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig, map_win_prob
from gnomepy_research.sessions.cs2_comp.g16.data import Inputs
from gnomepy_research.sessions.cs2_comp.g16.skill import SkillConfig, make_filter, map_arrays, observe

POSITIVE = {"sigma0", "tau", "rho", "faceit_weight", "lam", "kappa_max", "c0", "delta", "rating_gamma", "rating_sd"}
GROUPS = {"off_low": ("entry", "open"), "off_mid": ("intermediate", "main"), "off_high": ("advanced", "pro", "qualifier", "other")}
FILTER_PARAMS = {"sigma0", "tau", "rho", "lam", "kappa_max", "c0", "delta", "a", "b"}


def make_config(values: dict) -> SkillConfig:
    offsets = {}
    for g, divs in GROUPS.items():
        if g in values:
            offsets.update({d: values[g] for d in divs})
    return SkillConfig(replace(PlayerSkillConfig(), **{k: v for k, v in values.items() if k in FILTER_PARAMS}),
                       values.get("faceit_weight", 1.0), values.get("rating_gamma", 0.0), values.get("rating_sd", 0.31), offsets)


def map_loss(inputs: Inputs, cfg: SkillConfig, score_from: pd.Timestamp, won_a: np.ndarray) -> float:
    f = make_filter(inputs, cfg)
    cols = map_arrays(inputs, cfg)
    score = (cols["source"] == "hltv") & (cols["day"] >= np.datetime64(score_from)) & ~np.isnan(won_a)
    total, n = 0.0, 0
    for i in range(len(cols["known"])):
        if score[i]:
            day = pd.Timestamp(cols["day"][i])
            a = f.lineup(cols["five_a"][i], int(cols["team_a"][i]), 0.0, day)
            b = f.lineup(cols["five_b"][i], int(cols["team_b"][i]), 0.0, day)
            d, s2, _, _ = f.gap(a, b)
            p = min(max(map_win_prob(d, s2), 1e-4), 1 - 1e-4)
            total -= math.log(p) if won_a[i] > 0.5 else math.log(1 - p)
            n += 1
        observe(f, cols, i, cfg)
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--score-from", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--init", default="{}")
    ap.add_argument("--evaluate")
    ap.add_argument("--maxfev", type=int, default=150)
    ap.add_argument("--name", default="fit")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(a.data, pd.Timestamp(a.until))
    base = PlayerSkillConfig()
    values = {"sigma0": base.sigma0, "tau": base.tau, "rho": base.rho, "lam": base.lam, "faceit_weight": 1.0,
              "off_low": 0.0, "off_mid": 0.0, "off_high": 0.0, "rating_gamma": 0.0, "rating_sd": 0.31}
    values.update(json.loads(a.init))
    params = a.params.split(",")
    won = inputs.maps.won_a.to_numpy(float)
    trace = []
    if a.evaluate:
        for point in json.loads(a.evaluate):
            v = dict(values)
            v.update({k: x for k, x in point.items() if k != "name"})
            t0 = time.time()
            loss = map_loss(inputs, make_config(v), pd.Timestamp(a.score_from), won)
            trace.append({"name": point.get("name"), "loss": loss, "values": v, "seconds": time.time() - t0})
            print(f"{point.get('name')} {loss:.5f} {time.time() - t0:.0f}s", flush=True)
            (out / f"{a.name}.json").write_text(json.dumps(trace, indent=1))
        return

    def unpack(x: np.ndarray) -> dict:
        v = dict(values)
        for k, xi in zip(params, x):
            v[k] = math.exp(xi) if k in POSITIVE else xi
        return v

    def objective(x: np.ndarray) -> float:
        v = unpack(x)
        t0 = time.time()
        loss = map_loss(inputs, make_config(v), pd.Timestamp(a.score_from), won)
        trace.append({"loss": loss, "values": {k: v[k] for k in params}, "seconds": time.time() - t0})
        print(f"{len(trace):4d} {loss:.5f} {json.dumps({k: round(v[k], 5) for k in params})} {time.time() - t0:.0f}s", flush=True)
        (out / f"{a.name}.json").write_text(json.dumps({"best": min(trace, key=lambda r: r["loss"]), "trace": trace}, indent=1))
        return loss

    x0 = np.array([math.log(values[k]) if k in POSITIVE else values[k] for k in params])
    step = np.array([0.3 if k in POSITIVE else 0.15 for k in params])
    simplex = np.vstack([x0] + [x0 + step * np.eye(len(params))[i] for i in range(len(params))])
    minimize(objective, x0, method="Nelder-Mead", options={"maxfev": a.maxfev, "initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-5})


if __name__ == "__main__":
    main()
