"""Chronological backtest of the pregame win-probability model against baselines.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2023 2024
    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2025 --holdout

For each test season S the outcome model and every baseline are fitted on 2020..S-1 only (rolling
origin). FBS-vs-FBS completed games. 2025 is the untouched holdout: it refuses to run without
--holdout so it isn't scored by accident while tuning.

Two noise sources are reported: the game-level paired bootstrap, and refit noise (the Bayesian
model refitted with different rating-draw subsets and seeds).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.data import UNITS_ALL, UNITS_SPLIT
from gnomepy_research.sessions.cfb_win_probability.evaluation import baselines
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import paired_bootstrap, reliability, summarize
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import N_FIT_DRAWS, fit, game_draws

FIRST_OUTCOME_SEASON = 2020
HOLDOUT = 2025
RESULTS = Path(__file__).resolve().parent.parent / "results" / "backtest"
N_REFITS = 3
MODEL_CONFIGS = {
    "A": dict(units=UNITS_ALL, form=False, qb=False),
    "B": dict(units=UNITS_SPLIT, form=False, qb=False),
    "C": dict(units=UNITS_SPLIT, form=True, qb=False),
    "D": dict(units=UNITS_SPLIT, form=True, qb=True),
}


def phase(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["season_type"] == "postseason", "post",
                              pd.cut(df["week"], [0, 2, 5, 9, 20], labels=["wk1-2", "wk3-5", "wk6-9", "wk10+"]).astype(str)),
                     index=df.index)


def run_season(season: int, games: pd.DataFrame, pregame: pd.DataFrame, lines: pd.DataFrame,
               units: tuple[str, ...] = UNITS_ALL, form: bool = False) -> tuple[pd.DataFrame, dict]:
    train_seasons = list(range(FIRST_OUTCOME_SEASON, season))
    train_gd = game_draws(games, train_seasons, n_draws=N_FIT_DRAWS, units=units, form=form)
    test_gd = game_draws(games, [season], units=units, form=form)
    test = test_gd.games.loc[test_gd.games["home_win"].notna()].copy()
    x_test = test_gd.x[test_gd.games["home_win"].notna().to_numpy()]

    refits = []
    for r in range(N_REFITS):
        gd = train_gd if r == 0 else game_draws(games, train_seasons, n_draws=N_FIT_DRAWS, seed=r, units=units, form=form)
        model = fit(gd, seed=r)
        if r == 0:
            coefs = model.gamma
        refits.append(model.predict(x_test))
    pred = refits[0]
    test["p_bayes"] = pred["p_home"].to_numpy()
    test["p_home"] = test["p_bayes"]
    test["p_bayes_lo"] = pred["p_lo"].to_numpy()
    test["p_bayes_hi"] = pred["p_hi"].to_numpy()

    train = train_gd.games.loc[train_gd.games["home_win"].notna()]
    test["p_home_field"] = baselines.home_field(train, test)

    elo_params = baselines.tune_elo(games, set(train["game_id"]))
    test["p_elo"] = test["game_id"].map(baselines.elo_predictions(games, elo_params))

    pg = pregame.set_index("game_id")
    tr_pg = pg.loc[train["game_id"]].assign(home_win=train["home_win"].to_numpy())
    te_pg = pg.loc[test["game_id"]]
    test["p_logistic"] = baselines.logistic(tr_pg, te_pg)

    market, sd = baselines.market_close(lines, train)
    test["p_market_close"] = test["game_id"].map(market)
    test["phase"] = phase(test)

    y = test["home_win"].to_numpy()
    refit_ll = [summarize(y, r["p_home"].to_numpy())["log_loss"] for r in refits]
    meta = {"season": season, "train_seasons": train_seasons, "elo": elo_params.__dict__, "spread_sd": sd,
            "refit_log_loss": refit_ll, "refit_noise_sd": float(np.std(refit_ll)),
            "terms": train_gd.terms, "gamma_mean": coefs.mean(0).tolist(),
            "gamma_lo": np.quantile(coefs, 0.05, axis=0).tolist(), "gamma_hi": np.quantile(coefs, 0.95, axis=0).tolist()}
    return test, meta


MODELS = ["p_bayes", "p_logistic", "p_elo", "p_home_field", "p_market_close"]


def report(df: pd.DataFrame, metas: list[dict]) -> str:
    lines = []
    y = df["home_win"].to_numpy()
    rows = {m: summarize(y, df[m].to_numpy()) for m in MODELS[:-1]}
    mk = df.dropna(subset=["p_market_close"])
    rows["p_market_close (closing, post-cutoff)"] = summarize(mk["home_win"].to_numpy(), mk["p_market_close"].to_numpy())
    lines.append("overall:\n" + pd.DataFrame(rows).T.round(4).to_string())

    by_season = {}
    for s, d in df.groupby("season"):
        by_season[s] = {m: summarize(d["home_win"].to_numpy(), d[m].fillna(0.5).to_numpy())["log_loss"] for m in MODELS[:-1]}
    lines.append("log loss by season:\n" + pd.DataFrame(by_season).round(4).to_string())

    ph = {}
    for p, d in df.groupby("phase"):
        ph[p] = {m: summarize(d["home_win"].to_numpy(), d[m].to_numpy())["log_loss"] for m in MODELS[:-1]}
    lines.append("log loss by phase:\n" + pd.DataFrame(ph).round(4).to_string())

    comp = {}
    for m in MODELS[1:-1]:
        comp[f"bayes - {m[2:]}"] = paired_bootstrap(y, df["p_bayes"].to_numpy(), df[m].to_numpy())
    comp["bayes - market_close"] = paired_bootstrap(mk["home_win"].to_numpy(), mk["p_bayes"].to_numpy(),
                                                    mk["p_market_close"].to_numpy())
    lines.append("paired log-loss differences (negative = Bayes better; game bootstrap):\n"
                 + pd.DataFrame(comp).T.round(4).to_string())
    lines.append("refit noise (Bayes log loss across refits): "
                 + "; ".join(f"{m['season']}: {np.round(m['refit_log_loss'], 4).tolist()} (sd {m['refit_noise_sd']:.4f})"
                             for m in metas))
    fav = np.maximum(df["p_market_close"].fillna(df["p_bayes"]), 1 - df["p_market_close"].fillna(df["p_bayes"]))
    df = df.assign(fav_band=pd.cut(fav, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True).astype(str))
    fb = {}
    for b, d in df.groupby("fav_band"):
        fb[b] = {m: summarize(d["home_win"].to_numpy(), d[m].fillna(0.5).to_numpy())["log_loss"] for m in ("p_bayes", "p_logistic", "p_elo")}
        fb[b]["n"] = len(d)
    lines.append("log loss by market-favourite band:\n" + pd.DataFrame(fb).round(4).to_string())
    lines.append("Bayes reliability:\n" + reliability(y, df["p_bayes"].to_numpy()).round(3).to_string())
    width = (df["p_bayes_hi"] - df["p_bayes_lo"])
    lines.append(f"90% interval on P(home): median width {width.median():.3f}; widest phase "
                 f"{df.assign(w=width).groupby('phase')['w'].median().round(3).to_dict()}")
    coef = [pd.DataFrame({"mean": m["gamma_mean"], "lo5": m["gamma_lo"], "hi95": m["gamma_hi"]}, index=m["terms"])
            .round(3).to_string() for m in metas]
    lines.append("outcome coefficients (posterior mean, 90% interval; first refit):\n"
                 + "\n".join(f"{m['season']}:\n{c}" for m, c in zip(metas, coef)))
    return "\n\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", type=int, nargs="+")
    ap.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="A")
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    if HOLDOUT in args.seasons and not args.holdout:
        raise SystemExit(f"{HOLDOUT} is the holdout season; pass --holdout to score it")
    cfg = MODEL_CONFIGS[args.model]
    if cfg["qb"]:
        raise NotImplementedError(f"model {args.model} is not wired yet")
    ds = DatasetStore()
    games, pregame, lines = ds.load("cfb_games"), ds.load("cfb_pregame_features"), ds.load("cfb_lines")
    frames, metas = [], []
    for s in args.seasons:
        df, meta = run_season(s, games, pregame, lines, units=cfg["units"], form=cfg["form"])
        frames.append(df)
        metas.append(meta)
        print(f"{s}: {len(df)} games", json.dumps(meta, default=str), flush=True)
    out = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "_".join(map(str, args.seasons)) + f"_{args.model}"
    out.to_parquet(RESULTS / f"predictions_{tag}.parquet", index=False)
    text = report(out, metas)
    (RESULTS / f"report_{tag}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
