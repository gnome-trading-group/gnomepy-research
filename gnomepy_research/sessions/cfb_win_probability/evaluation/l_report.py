"""Model L diagnostics: G/K/L cut table with bootstrap CIs, postseason confidence, rating volatility, phase taus.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.l_report 2023 2024

Appends to results/backtest/report_<seasons>_L.txt, replacing any extras from an earlier run.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency.data import TAU_PHASES, UNITS_SPLIT
from gnomepy_research.models.cfb.efficiency.inference import HYPER_DIR
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import _suffix, hyper_name
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS, resolve_config
from gnomepy_research.sessions.cfb_win_probability.evaluation.g_report import P4, append_extras, cut_frame
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss

MODELS = ["G", "K", "L"]
PAIRS = [("L", "K"), ("L", "G")]
N_BOOT = 2000
CONFIDENT = 0.25
BUCKETS = {"wk2-5": (2, 5), "wk6-10": (6, 10), "wk11+": (11, 99)}
POSTERIOR_DIR = RESULTS.parent / "posterior_draws"


def l_cuts(g: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cut name -> (row mask, flip to the away side?)."""
    reg = (g["season_type"] == "regular").to_numpy()
    neutral = g["neutral_site"].astype(bool).to_numpy()
    week = g["week"].to_numpy()
    o = g["p_open"].to_numpy()
    p4 = lambda conf, team: (conf.isin(P4) | (team == "Notre Dame")).to_numpy()
    hp, ap = p4(g["home_conference"], g["home_team"]), p4(g["away_conference"], g["away_team"])
    none = np.zeros(len(g), bool)
    return {
        "regular home-site": (reg & ~neutral, none),
        "regular neutral-site": (reg & neutral, none),
        "postseason": (~reg, none),
        "weeks 1-4": (reg & (week >= 1) & (week <= 4), none),
        "weeks 5-9": (reg & (week >= 5) & (week <= 9), none),
        "weeks 10+": (reg & (week >= 10), none),
        "big favorites": ((o >= 0.8) | (o <= 0.2), o <= 0.2),
        "toss-ups": ((o >= 0.4) & (o <= 0.6), none),
        "P4 vs G5": (hp ^ ap, ap & ~hp),
        "non-conference": (~g["conference_game"].astype(bool).to_numpy(), none),
    }


def bootstrap_diff(y, pa, pb, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    d = log_loss(np.asarray(y, float), np.asarray(pa, float)) - log_loss(np.asarray(y, float), np.asarray(pb, float))
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def cut_table(g: pd.DataFrame, models: list[str] = MODELS, pairs: list[tuple[str, str]] = PAIRS,
              n_boot: int = N_BOOT, cuts: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    rows = []
    for name, (mask, flip) in (l_cuts(g) if cuts is None else cuts).items():
        s, fl = g[mask], flip[mask]
        y = np.where(fl, 1 - s["home_win"], s["home_win"]).astype(float)
        side = {m: np.where(fl, 1 - s[f"p_{m}"], s[f"p_{m}"]) for m in models + ["open"]}
        row = {"cut": name, "n": len(s)}
        row.update({m: round(float(log_loss(y, side[m]).mean()), 4) for m in models + ["open"]})
        for a, b in pairs:
            m, lo, hi = bootstrap_diff(y, side[a], side[b], n_boot)
            row[f"{a}-{b}"] = f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_stats(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    wrong = (p > 0.5) != (y == 1)
    return {"n": len(y), "mean_conf": float(np.abs(p - 0.5).mean()),
            "wrong": int(wrong.sum()), "confident_wrong": int((wrong & (np.abs(p - 0.5) > CONFIDENT)).sum())}


def rating_volatility(post: pd.DataFrame, seasons: list[int]) -> dict:
    d = post[(post["season_type"] == "regular") & post["season"].isin(seasons)]
    if "unit" in d:
        d = d[d["unit"] == "pass"]
    d = d.assign(r=d["offense_mean"] + d["defense_mean"]).sort_values(["season", "team_id", "week"])
    d = d.assign(dr=d.groupby(["season", "team_id"])["r"].diff().abs())
    out = {b: float(d.loc[d["week"].between(lo, hi), "dr"].mean()) for b, (lo, hi) in BUCKETS.items()}
    last = d[d["week"] == d.groupby("season")["week"].transform("max")]
    out["final_sd"] = float(last.groupby("season")["r"].std(ddof=1).mean())
    return out


def season_log_loss(g: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    y = g["home_win"].astype(float)
    cols = {m: log_loss(y, g[f"p_{m}"]) for m in models} | {"open": log_loss(y, g["p_open"])}
    return pd.DataFrame(cols).assign(season=g["season"].to_numpy()).groupby("season").mean()


def tau_phase_rows(season: int, raw: dict) -> list[dict]:
    h = raw["hyperparameters"]
    return [{"season": season, "param": k, **dict(zip(TAU_PHASES, h[k]))} for k in ("tau_off", "tau_def")]


def _posterior_path(model: str, first: int, last: int):
    c = resolve_config(model)
    sfx = _suffix(c["units"], c.get("conference", False), c.get("projection", False), tuple(c.get("drop_features", ())),
                  c.get("drives", False), c.get("independents_separate", False), c.get("tau_phases", False),
                  c.get("heavy_tails", False))
    return POSTERIOR_DIR / f"cfb_team_week_posterior{sfx}_{first}_{last}.parquet"


def tau_section(seasons: list[int]) -> str:
    rows, diag = [], []
    for s in seasons:
        raw = json.loads((HYPER_DIR / f"{hyper_name('normal', s, UNITS_SPLIT, True, drives=True, independents_separate=True, tau_phases=True)}.json").read_text())
        rows += tau_phase_rows(s, raw)
        d = raw["diagnostics"]["hyper"]
        diag.append(f"{s}: max_rhat={d['max_rhat']:.3f} ({d['worst_rhat_param']}) min_ess_bulk={d['min_ess_bulk']:.0f} "
                    f"divergences={d['divergences']}")
    return ("L FITTED TAU PER PHASE (fitted on 2019..S-1)\n" + pd.DataFrame(rows).round(4).to_string(index=False)
            + "\nhyper diagnostics:\n" + "\n".join(diag))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", type=int, nargs="+")
    args = ap.parse_args()
    tag = "_".join(map(str, args.seasons))
    g = cut_frame(tag, MODELS)
    post = g[g["season_type"] == "postseason"]
    conf = pd.DataFrame({m: confidence_stats(post["home_win"], post[f"p_{m}"]) for m in MODELS + ["open"]}).T
    vol = pd.DataFrame({m: rating_volatility(pd.read_parquet(_posterior_path(m, 2020, max(args.seasons))), args.seasons)
                        for m in MODELS}).T
    sections = [
        f"G vs K vs L CUT TABLE (side-oriented log loss; paired L-K, L-G with {N_BOOT}-resample game bootstrap 95% CI)\n"
        + cut_table(g).to_string(index=False),
        f"POSTSEASON CONFIDENCE (mean |p-0.5|; wrong = wrong side; confident_wrong = wrong and |p-0.5| > {CONFIDENT})\n"
        + conf.to_string(),
        "RATING VOLATILITY (pass unit when present, regular season; mean |delta(offense_mean+defense_mean)| week to week "
        "bucketed by the later week; final_sd = sd across teams at the last regular week, averaged over seasons)\n"
        + vol.round(4).to_string(),
        tau_section(list(range(2020, max(args.seasons) + 1))),
        "PER-SEASON LOG LOSS\n" + season_log_loss(g, MODELS).round(4).to_string(),
    ]
    text = "\n\n".join(sections)
    append_extras(RESULTS / f"report_{tag}_L.txt", text)
    print(text)


if __name__ == "__main__":
    main()
