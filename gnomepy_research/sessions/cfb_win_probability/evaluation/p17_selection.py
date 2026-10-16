"""Phase 17 selection, applying the rule pre-registered in the plan (Task 48) to the 2021-24 rolling backtests.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.p17_selection

Npm is adopted over Np if, pooled over 2021-24, it improves log loss by at least ADOPT_GAIN with paired-bootstrap
P(worse) <= MAX_P_WORSE, no season is worse than Np by more than MAX_SEASON_LOSS, and the big-favorite calibration
gap (actual - mean predicted on open > 80% favorites) shrinks. Npmt replaces Npm only with a further TIER_GAIN.
2025 is never read.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import paired_bootstrap

TAGS = ("2021_2022", "2023_2024")
MODELS = ("Np", "Npm", "Npmt", "K", "Km")
ADOPT_GAIN = 0.001
MAX_P_WORSE = 0.10
MAX_SEASON_LOSS = 0.003
TIER_GAIN = 0.0005
BIG_FAVORITE = 0.8
REPORT_TXT = RESULTS / "p17_selection.txt"
SELECTION_JSON = RESULTS / "P17_selection.json"


def load(p_open: pd.Series) -> pd.DataFrame:
    frames = {m: pd.concat([pd.read_parquet(RESULTS / f"predictions_{t}_{m}.parquet") for t in TAGS])
              .drop_duplicates("game_id").set_index("game_id") for m in MODELS}
    base = frames["Np"][["season", "week", "season_type", "home_win"]]
    d = base.join(pd.DataFrame({m: f["p_home"] for m, f in frames.items()}), how="inner").join(p_open, how="inner")
    return d.dropna()


def log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def favorite_gap(d: pd.DataFrame, col: str) -> float:
    """Actual minus mean predicted win rate of open > BIG_FAVORITE favorites (positive = under-confident)."""
    fh = d["p_open"] >= 0.5
    big = np.where(fh, d["p_open"], 1 - d["p_open"]) > BIG_FAVORITE
    y = np.where(fh, d["home_win"], 1 - d["home_win"])[big]
    p = np.where(fh, d[col], 1 - d[col])[big]
    return float(y.mean() - p.mean())


def evaluate(d: pd.DataFrame, cand: str, ref: str) -> dict:
    y = d["home_win"].to_numpy()
    boot = paired_bootstrap(y, d[cand].to_numpy(), d[ref].to_numpy())
    per = {int(s): float(log_loss(g["home_win"].to_numpy(), g[cand].to_numpy()).mean()
                         - log_loss(g["home_win"].to_numpy(), g[ref].to_numpy()).mean()) for s, g in d.groupby("season")}
    gap_c, gap_r = favorite_gap(d, cand), favorite_gap(d, ref)
    ok = (boot["diff"] <= -ADOPT_GAIN and boot["p_a_worse"] <= MAX_P_WORSE and max(per.values()) <= MAX_SEASON_LOSS
          and abs(gap_c) < abs(gap_r))
    return {"diff": float(boot["diff"]), "lo95": float(boot["lo95"]), "hi95": float(boot["hi95"]),
            "p_worse": float(boot["p_a_worse"]), "per_season": per, "fav_gap": gap_c, "fav_gap_ref": gap_r, "passes": bool(ok)}


def main() -> None:
    p_open = market_probs(DatasetStore().load("cfb_lines")).set_index("game_id")["p_open"]
    d = load(p_open)
    y = d["home_win"].to_numpy()
    fh = d["p_open"] >= 0.5
    big = np.where(fh, d["p_open"], 1 - d["p_open"]) > BIG_FAVORITE
    phase = np.where(d["season_type"] == "postseason", "post", np.where(d["week"] <= 4, "wk1-4", np.where(d["week"] <= 9, "wk5-9", "wk10+")))
    table = pd.DataFrame({m: {"all": log_loss(y, d[m].to_numpy()).mean(),
                              "big favorites": log_loss(y[big], d[m].to_numpy()[big]).mean(),
                              "other games": log_loss(y[~big], d[m].to_numpy()[~big]).mean(),
                              **{ph: log_loss(y[phase == ph], d[m].to_numpy()[phase == ph]).mean() for ph in ("wk1-4", "wk5-9", "wk10+", "post")},
                              "big-fav gap (actual - predicted)": favorite_gap(d, m)}
                          for m in (*MODELS, "p_open")}).rename(columns={"p_open": "open"})
    per_season = pd.DataFrame({m: d.groupby("season").apply(lambda g: log_loss(g["home_win"].to_numpy(), g[m].to_numpy()).mean())
                               for m in (*MODELS, "p_open")}).rename(columns={"p_open": "open"})
    res = {"Npm vs Np": evaluate(d, "Npm", "Np"), "Npmt vs Np": evaluate(d, "Npmt", "Np"),
           "Npmt vs Npm": evaluate(d, "Npmt", "Npm"), "Km vs K": evaluate(d, "Km", "K")}
    chosen = "Np"
    if res["Npm vs Np"]["passes"]:
        chosen = "Npm"
        if res["Npmt vs Np"]["passes"] and res["Npmt vs Npm"]["diff"] <= -TIER_GAIN:
            chosen = "Npmt"
    elif res["Npmt vs Np"]["passes"] and res["Npmt vs Np"]["diff"] <= res["Npm vs Np"]["diff"] - TIER_GAIN:
        chosen = "Npmt"
    text = "\n\n".join([
        f"Phase 17 selection: pooled 2021-24 rolling backtests, {len(d)} FBS games with an opening line; 2025 not read",
        "log loss by cut:\n" + table.round(4).to_string(),
        "log loss by season:\n" + per_season.round(4).to_string(),
        "\n".join(f"{k}: diff {v['diff']:+.4f} [{v['lo95']:+.4f}, {v['hi95']:+.4f}] P(worse) {v['p_worse']:.2f}; "
                  f"per season {', '.join(f'{s} {x:+.4f}' for s, x in v['per_season'].items())}; "
                  f"big-fav gap {v['fav_gap']:+.4f} vs {v['fav_gap_ref']:+.4f}; passes rule: {v['passes']}" for k, v in res.items()),
        f"chosen: {chosen}"])
    REPORT_TXT.write_text(text + "\n")
    SELECTION_JSON.write_text(json.dumps({"chosen": chosen, "results": res}, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
