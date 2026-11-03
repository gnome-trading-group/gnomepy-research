"""Development walk-forward for g07: price every HLTV bo3 series online (from maps known before it starts) and score it.

    python -m gnomepy_research.sessions.cs2_comp.g07.dev --data <bundle> --out <parquet> [--config k=v,...]

The online price uses slightly more data than a weekly refit would (maps of the same week), which is fine for ranking
configurations against each other."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g07.data import FIVE, Inputs
from gnomepy_research.sessions.cs2_comp.g07.model import Config, Filter, bo3_price

EPOCH = np.datetime64("2023-01-01")
DEV_START = pd.Timestamp("2025-01-01")


def series_fives(inputs: Inputs) -> pd.DataFrame:
    s = inputs.series
    s = s[(s.bo_type == 3) & (s.time >= DEV_START)].copy()
    first = inputs.maps[inputs.maps.src == 0].drop_duplicates("match_id").set_index("match_id")
    for side in ("a", "b"):
        out = []
        for mid, lu in zip(s.match_id.astype(str), s[f"lineup_{side}"]):
            five = tuple(int(x) for x in lu) if lu is not None and len(lu) == FIVE and len(set(lu)) == FIVE else None
            if five is None and mid in first.index:
                five = first.at[mid, f"five_{side}"]
            out.append(five)
        s[f"five_{side}"] = out
    return s[s.five_a.notna() & s.five_b.notna()].sort_values("time").reset_index(drop=True)


def online(inputs: Inputs, cfg: Config, series: pd.DataFrame) -> pd.DataFrame:
    f = Filter(cfg)
    rows = []
    for r in series.itertuples(index=False):
        f.advance(inputs.maps, np.datetime64(r.time), EPOCH)
        now = float((np.datetime64(r.time) - EPOCH) / np.timedelta64(1, "D"))
        ma, va, ua, na = f.side(r.five_a, int(r.team_a), now)
        mb, vb, ub, nb = f.side(r.five_b, int(r.team_b), now)
        rg, rv = f.region_gap(r.five_a, r.five_b, int(r.team_a), int(r.team_b), now)
        rows.append((r.match_id, r.time, r.y, r.is_lan, r.tier, ma - mb + rg, va + vb + rv, ua, ub, na, nb))
    out = pd.DataFrame(rows, columns=["match_id", "time", "y", "is_lan", "tier", "gap", "gvar", "unk_a", "unk_b", "n_a", "n_b"])
    out["q"] = bo3_price(out.gap.to_numpy(), out.gvar.to_numpy())
    out["q_point"] = bo3_price(out.gap.to_numpy(), np.zeros(len(out)))
    return out


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fit(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def held_out(cols, y, groups):
    X = np.column_stack(cols + [np.ones(len(y))])
    out = np.empty(len(y))
    for g in np.unique(groups):
        tr = groups != g
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ _fit(X[tr], y[tr])))
    return out


def kickoff_prices(data: str) -> pd.DataFrame:
    """m0 per market series: the last print within 30 minutes of kickoff, else the last print."""
    p = load(data, "market/series_paths").sort_values(["match_id", "t"])
    last = p.groupby("match_id").tail(1)
    return pd.DataFrame({"match_id": last.match_id.astype(str).to_numpy(), "m0": last.p_a.to_numpy(),
                         "kickoff": last.kickoff.dt.tz_convert(None).to_numpy()})


def summary(res: pd.DataFrame, market: pd.DataFrame) -> dict:
    y = res.y.to_numpy(float)
    pre = (res.time < pd.Timestamp("2026-03-01")).to_numpy()
    rec = held_out([_logit(res.q)], y, res.time.dt.to_period("Q").astype(str).to_numpy())
    j = res.assign(match_id=res.match_id.astype(str)).merge(market, on="match_id")
    yy, mo = j.y.to_numpy(float), j.kickoff.dt.month.to_numpy()
    d = _ll(held_out([_logit(j.m0), _logit(j.q)], yy, mo), yy) - _ll(j.m0, yy)
    return {"ll_pre": float(_ll(res.q, y)[pre].mean()), "ll_pre_recal": float(_ll(rec, y)[pre].mean()),
            "pool_mkt_all": float(d.mean()), "pool_mkt_dev": float(d[mo >= 4].mean()), "n_mkt": int(len(j))}


def load_inputs(data: str, cache: str | None) -> Inputs:
    if cache and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    inputs = Inputs(data, pd.Timestamp("2026-08-01"))
    if cache:
        Path(cache).write_bytes(pickle.dumps(inputs))
    return inputs


def parse_config(text: str | None) -> Config:
    if not text:
        return Config()
    kw = {k: (None if v == "none" else float(v)) for k, v in (p.split("=") for p in text.split(","))}
    return dataclasses.replace(Config(), **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--config", action="append", help="k=v,... (repeatable: one run per config)")
    ap.add_argument("--configs-json", help="a JSON list of config strings, appended to --config")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    if a.configs_json:
        a.config = (a.config or []) + json.loads(a.configs_json)
    t0 = time.time()
    inputs = load_inputs(a.data, a.cache)
    series = series_fives(inputs)
    print(f"inputs {time.time() - t0:.0f}s maps {len(inputs.maps)} series {len(series)}", flush=True)
    out_dir = Path(a.out or os.environ.get("OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    market = kickoff_prices(a.data)
    summaries = []
    for i, text in enumerate(a.config or [""]):
        t0 = time.time()
        cfg = parse_config(text)
        res = online(inputs, cfg, series)
        res["config"] = text
        if a.save:
            res.to_parquet(out_dir / f"dev_{a.tag}{i}.parquet", index=False)
        sm = {"config": text, **summary(res, market)}
        summaries.append(sm)
        print(f"[{i}] {time.time() - t0:.0f}s {json.dumps(sm)}", flush=True)
    (out_dir / f"summary_{a.tag}.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
