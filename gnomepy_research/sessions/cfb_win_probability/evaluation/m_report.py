"""Model M diagnostics: K/M cut table, big-favorite bias, postseason confidence, nu_team, rating spread.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.m_report 2023 2024

Appends to results/backtest/report_<seasons>_M.txt, replacing any extras from an earlier run.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT
from gnomepy_research.models.cfb.efficiency.inference import HYPER_DIR
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import hyper_name
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.g_report import _ci, append_extras, cut_frame, cut_sides
from gnomepy_research.sessions.cfb_win_probability.evaluation.l_report import (
    CONFIDENT, N_BOOT, _posterior_path, confidence_stats, cut_table, l_cuts, season_log_loss)

MODELS = ["K", "M"]
PAIRS = [("M", "K")]
BIG = 0.8
RHAT_LIMIT = 1.05
M_CUTS = ["big favorites", "favorites", "toss-ups", "postseason", "regular neutral-site", "regular home-site",
          "weeks 1-4", "weeks 10+", "P4 vs G5"]


def m_cuts(g: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pool = {**l_cuts(g), "favorites": cut_sides(g)["favorites"]}
    return {k: pool[k] for k in M_CUTS}


def favorite_bias(g: pd.DataFrame, models: list[str], big: float = BIG) -> pd.DataFrame:
    """Favorite side by the open, games with open >= big on that side: mean(result - p) per source."""
    o = g["p_open"].to_numpy()
    mask = (o >= big) | (o <= 1 - big)
    s, flip = g[mask], (o <= 1 - big)[mask]
    y = np.where(flip, 1 - s["home_win"], s["home_win"]).astype(float)
    rows = []
    for m in models + ["open"]:
        p = np.where(flip, 1 - s[f"p_{m}"], s[f"p_{m}"])
        rows.append({"source": m, "n": len(s), "win_rate": float(y.mean()), "mean_p": float(p.mean()),
                     "bias": float((y - p).mean()), "bias 95% CI": _ci(y - p)})
    return pd.DataFrame(rows)


def mcmc_diagnostics(post: pd.DataFrame) -> pd.DataFrame:
    c = post.drop_duplicates(["season", "season_type", "week"])
    return c.groupby("season").agg(
        cutoffs=("max_rhat", "size"), max_rhat=("max_rhat", "max"),
        **{"rhat_gt_1.05": ("max_rhat", lambda r: int((r > RHAT_LIMIT).sum()))},
        divergences=("divergences", "sum"), cutoffs_with_div=("divergences", lambda d: int((d > 0).sum()))).reset_index()


def final_spread(post: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Per season: spread of offense_mean + defense_mean (pass unit) across teams at the last regular week."""
    d = post[(post["season_type"] == "regular") & post["season"].isin(seasons)]
    if "unit" in d:
        d = d[d["unit"] == "pass"]
    d = d[d["week"] == d.groupby("season")["week"].transform("max")]
    r = (d["offense_mean"] + d["defense_mean"]).groupby(d["season"])
    return pd.DataFrame({"sd": r.std(ddof=1), "min": r.min(), "max": r.max(),
                         "top5_mean": r.apply(lambda v: v.nlargest(5).mean())})


def nu_row(season: int, raw: dict) -> dict:
    h, d = raw["hyperparameters"], raw["diagnostics"]["hyper"]
    return {"season": season, "nu_team": h["nu_team"], "s_off": h["s_off"], "s_def": h["s_def"],
            "max_rhat": d["max_rhat"], "worst_rhat_param": d["worst_rhat_param"],
            "min_ess_bulk": round(d["min_ess_bulk"]), "divergences": d["divergences"]}


def _hyper(season: int, heavy_tails: bool) -> dict:
    name = hyper_name("normal", season, UNITS_SPLIT, True, drives=True, independents_separate=True, heavy_tails=heavy_tails)
    return json.loads((HYPER_DIR / f"{name}.json").read_text())


def nu_section(seasons: list[int]) -> str:
    rows = [nu_row(s, _hyper(s, True)) for s in seasons]
    k = pd.DataFrame([{"season": s, "K s_off": _hyper(s, False)["hyperparameters"]["s_off"],
                       "K s_def": _hyper(s, False)["hyperparameters"]["s_def"]} for s in seasons])
    t = pd.DataFrame(rows).merge(k, on="season")
    return ("M FITTED nu_team PER HYPERPARAMETER SEASON (fitted on 2019..S-1; StudentT start-state df; "
            "K's start-state scales for reference)\n" + t.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", type=int, nargs="+")
    args = ap.parse_args()
    tag = "_".join(map(str, args.seasons))
    last = max(args.seasons)
    g = cut_frame(tag, MODELS)
    post = g[g["season_type"] == "postseason"]
    conf = pd.DataFrame({m: confidence_stats(post["home_win"], post[f"p_{m}"]) for m in MODELS + ["open"]}).T
    posts = {m: pd.read_parquet(_posterior_path(m, 2020, last)) for m in MODELS}
    spread = pd.concat({m: final_spread(posts[m], args.seasons) for m in MODELS}, axis=1)
    sections = [
        f"K vs M CUT TABLE (side-oriented log loss; paired M-K with {N_BOOT}-resample game bootstrap 95% CI)\n"
        + cut_table(g, MODELS, PAIRS, cuts=m_cuts(g)).to_string(index=False),
        f"BIG-FAVORITE BIAS (favorite side by the open, open >= {BIG}; bias = mean(result - p), 95% normal CI)\n"
        + favorite_bias(g, MODELS).round(4).to_string(index=False),
        f"POSTSEASON CONFIDENCE (mean |p-0.5|; wrong = wrong side; confident_wrong = wrong and |p-0.5| > {CONFIDENT})\n"
        + conf.to_string(),
        nu_section(list(range(2020, last + 1))),
        "PER-SEASON LOG LOSS\n" + season_log_loss(g, MODELS).round(4).to_string(),
        "END-OF-SEASON RATING SPREAD (pass unit offense_mean + defense_mean across teams at the final regular week)\n"
        + spread.round(4).to_string(),
        f"M CUTOFF MCMC DIAGNOSTICS (per season; max_rhat over cutoffs, cutoffs with max_rhat > {RHAT_LIMIT}, divergences)\n"
        + mcmc_diagnostics(posts["M"]).round(4).to_string(index=False),
    ]
    text = "\n\n".join(sections)
    append_extras(RESULTS / f"report_{tag}_M.txt", text)
    print(text)


if __name__ == "__main__":
    main()
