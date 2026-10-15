"""Model N diagnostics: cut table vs K, K's big-disagreement buckets, fitted extension hyperparameters,
per-season log loss and week-to-week rating volatility.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.n_report 2023 2024 --models K N1 N2 N

Appends to results/backtest/report_<seasons>_<last model>.txt, replacing any extras from an earlier run.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency.inference import HYPER_DIR
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import hyper_name
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS, resolve_config
from gnomepy_research.sessions.cfb_win_probability.evaluation.g_report import append_extras, cut_frame
from gnomepy_research.sessions.cfb_win_probability.evaluation.l_report import (
    N_BOOT, _posterior_path, cut_table, rating_volatility, season_log_loss)
from gnomepy_research.sessions.cfb_win_probability.evaluation.m_report import m_cuts
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss

BASE = "K"
DISAGREEMENT_EDGES = (0.0, 0.25, 0.5, 1.0, np.inf)
HYPER_KEYS = ("tau_off", "tau_def", "s_off", "s_def", "lam_ppd", "s_game_ppd", "s_drive_ppd", "kappa_ppd", "s_fcs",
              "s_fcs_ppd", "s_fin_off", "s_fin_def")


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def disagreement_table(g: pd.DataFrame, models: list[str], base: str = BASE) -> pd.DataFrame:
    """Games bucketed by |logit(open) - logit(base)|, fixed on the base model so every model is scored on the
    same games: mean and total log loss minus the open's."""
    y = g["home_win"].to_numpy(float)
    bucket = pd.cut(np.abs(_logit(g["p_open"]) - _logit(g[f"p_{base}"])), DISAGREEMENT_EDGES,
                    labels=["<0.25", "0.25-0.5", "0.5-1", ">1"], include_lowest=True)
    ll_open = log_loss(y, g["p_open"].to_numpy())
    rows = []
    for b in bucket.categories:
        m = np.asarray(bucket == b)
        row = {"|logit open - logit K|": b, "n": int(m.sum())}
        for mod in models:
            d = log_loss(y[m], g.loc[m, f"p_{mod}"].to_numpy()) - ll_open[m]
            row[f"{mod} mean"], row[f"{mod} total"] = round(float(d.mean()), 4), round(float(d.sum()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _hyper_kwargs(model: str) -> dict:
    c = resolve_config(model)
    return dict(units=c["units"], conference=c.get("conference", False), drives=c.get("drives", False),
                independents_separate=c.get("independents_separate", False), extensions=tuple(c.get("extensions", ())))


def hyper_section(models: list[str], seasons: list[int]) -> str:
    rows = []
    for m in models:
        for s in seasons:
            raw = json.loads((HYPER_DIR / f"{hyper_name('normal', s, **_hyper_kwargs(m))}.json").read_text())
            h, d = raw["hyperparameters"], raw["diagnostics"]["hyper"]
            rows.append({"model": m, "hyper seasons": f"2019-{s - 1}",
                         **{k: (np.round(h[k], 4).tolist() if k in h else "") for k in HYPER_KEYS},
                         "max_rhat": round(d["max_rhat"], 3), "divergences": d["divergences"]})
    return ("FITTED HYPERPARAMETERS (rolling origin; kappa per unit pass, rush)\n"
            + pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", type=int, nargs="+")
    ap.add_argument("--models", nargs="+", default=[BASE, "N1", "N2", "N"])
    args = ap.parse_args()
    tag = "_".join(map(str, args.seasons))
    models = args.models
    last = max(args.seasons)
    g = cut_frame(tag, models)
    pairs = [(m, BASE) for m in models if m != BASE]
    vol = pd.DataFrame({m: rating_volatility(pd.read_parquet(_posterior_path(m, 2020, last)), args.seasons)
                        for m in models}).T
    sections = [
        f"{' / '.join(models)} CUT TABLE (side-oriented log loss; paired X-{BASE} with {N_BOOT}-resample game "
        "bootstrap 95% CI)\n" + cut_table(g, models, pairs, cuts=m_cuts(g)).to_string(index=False),
        f"BY {BASE}'S DISAGREEMENT WITH THE OPEN (log loss minus open's; buckets fixed on {BASE})\n"
        + disagreement_table(g, models).to_string(index=False),
        hyper_section(models, list(range(2020, last + 1))),
        "PER-SEASON LOG LOSS\n" + season_log_loss(g, models).round(4).to_string(),
        "RATING VOLATILITY (pass unit offense_mean + defense_mean: mean |week-to-week change| by phase; "
        "final_sd = spread across teams at the last regular week)\n" + vol.round(4).to_string(),
    ]
    text = "\n\n".join(sections)
    append_extras(RESULTS / f"report_{tag}_{models[-1]}.txt", text)
    print(text)


if __name__ == "__main__":
    main()
