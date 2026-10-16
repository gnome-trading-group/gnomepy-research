"""Pre-registered candidate selection on the 2021-24 rolling backtests (the Task 51/53 rule).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.candidate_selection \
        --label p20 --ref Np --candidates Npv Npvl [--nested Npvl:Npv]

A candidate passes against the reference if, pooled over 2021-24, it improves log loss by at least ADOPT_GAIN with
paired-bootstrap P(worse) <= MAX_P_WORSE and no season is worse by more than MAX_SEASON_LOSS. Among passing
candidates the lowest pooled log loss wins, except that a nested candidate (A:B, A extends B) beats B only with a
further NESTED_GAIN. 2025 is never read.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import paired_bootstrap

TAGS = ("2021_2022", "2023_2024")
ADOPT_GAIN = 0.001
MAX_P_WORSE = 0.10
MAX_SEASON_LOSS = 0.003
NESTED_GAIN = 0.0005


def log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load(models: list[str], p_open: pd.Series) -> pd.DataFrame:
    frames = {m: pd.concat([pd.read_parquet(RESULTS / f"predictions_{t}_{m}.parquet") for t in TAGS])
              .drop_duplicates("game_id").set_index("game_id") for m in models}
    base = frames[models[0]][["season", "week", "season_type", "home_win"]]
    d = base.join(pd.DataFrame({m: f["p_home"] for m, f in frames.items()}), how="inner").join(p_open, how="inner")
    return d.dropna()


def compare(d: pd.DataFrame, cand: str, ref: str) -> dict:
    y = d["home_win"].to_numpy()
    boot = paired_bootstrap(y, d[cand].to_numpy(), d[ref].to_numpy())
    per = {int(s): float(log_loss(g["home_win"].to_numpy(), g[cand].to_numpy()).mean()
                         - log_loss(g["home_win"].to_numpy(), g[ref].to_numpy()).mean()) for s, g in d.groupby("season")}
    ok = boot["diff"] <= -ADOPT_GAIN and boot["p_a_worse"] <= MAX_P_WORSE and max(per.values()) <= MAX_SEASON_LOSS
    return {"diff": float(boot["diff"]), "lo95": float(boot["lo95"]), "hi95": float(boot["hi95"]),
            "p_worse": float(boot["p_a_worse"]), "per_season": per, "passes": bool(ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ref", default="Np")
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--nested", nargs="*", default=[], help="A:B pairs where A extends B")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024],
                    help="seasons pooled by the rule (a pre-registered subset when a candidate cannot differ in some season)")
    args = ap.parse_args()
    models = [args.ref, *args.candidates]
    d = load(models, market_probs(DatasetStore().load("cfb_lines")).set_index("game_id")["p_open"])
    d = d[d["season"].isin(args.seasons)]
    y = d["home_win"].to_numpy()
    phase = np.where(d["season_type"] == "postseason", "post", np.where(d["week"] <= 4, "wk1-4", np.where(d["week"] <= 9, "wk5-9", "wk10+")))
    cols = (*models, "p_open")
    cuts = pd.DataFrame({m: {"all": log_loss(y, d[m].to_numpy()).mean(),
                             **{ph: log_loss(y[phase == ph], d[m].to_numpy()[phase == ph]).mean() for ph in ("wk1-4", "wk5-9", "wk10+", "post")}}
                         for m in cols}).rename(columns={"p_open": "open"})
    seasons = pd.DataFrame({m: {int(s): log_loss(g["home_win"].to_numpy(), g[m].to_numpy()).mean() for s, g in d.groupby("season")}
                            for m in cols}).rename(columns={"p_open": "open"})
    res = {c: compare(d, c, args.ref) for c in args.candidates}
    nested = dict(p.split(":") for p in args.nested)
    for a, b in nested.items():
        res[f"{a} vs {b}"] = compare(d, a, b)
    passing = [c for c in args.candidates if res[c]["passes"]]
    for a, b in nested.items():
        if a in passing and b in passing and res[f"{a} vs {b}"]["diff"] > -NESTED_GAIN:
            passing.remove(a)
    chosen = min(passing, key=lambda c: res[c]["diff"]) if passing else args.ref
    text = "\n\n".join([
        f"{args.label} selection vs {args.ref}: pooled {min(args.seasons)}-{str(max(args.seasons))[2:]} rolling backtests, "
        f"{len(d)} FBS games with an opening line; 2025 not read",
        "log loss by phase:\n" + cuts.round(4).to_string(),
        "log loss by season:\n" + seasons.round(4).to_string(),
        "\n".join(f"{k}: diff {v['diff']:+.4f} [{v['lo95']:+.4f}, {v['hi95']:+.4f}] P(worse) {v['p_worse']:.2f}; per season "
                  + ", ".join(f"{s} {x:+.4f}" for s, x in v["per_season"].items()) + f"; passes: {v['passes']}" for k, v in res.items()),
        f"chosen: {chosen}"])
    (RESULTS / f"{args.label}_selection.txt").write_text(text + "\n")
    (RESULTS / f"{args.label}_selection.json").write_text(json.dumps({"chosen": chosen, "results": res}, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
