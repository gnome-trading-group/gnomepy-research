"""Closing-line value: does the market move from open to close toward the model?

Regresses (p_close - p_open) on (p_model - p_open) with an intercept; a positive slope means the line moves
toward the model between open and kickoff.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.clv 2023_2024 G J K
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import BACKTEST, market_probs

N_BOOT = 200
LEAN_THRESHOLD = 0.05


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def clv_regression(p_model, p_open, p_close, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    x = np.asarray(p_model, float) - np.asarray(p_open, float)
    y = np.asarray(p_close, float) - np.asarray(p_open, float)
    slope, intercept = _fit(x, y)
    rng = np.random.default_rng(seed)
    boots = [_fit(x[idx], y[idx])[0] for idx in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return {"n": len(x), "slope": slope, "lo95": float(np.quantile(boots, 0.025)),
            "hi95": float(np.quantile(boots, 0.975)), "intercept": intercept, "corr": float(np.corrcoef(x, y)[0, 1])}


def clv_frame(preds: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    return preds[["game_id", "p_home"]].merge(market, on="game_id").dropna(subset=["p_home", "p_open", "p_close"])


def clv_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for m, f in frames.items():
        lean = (f["p_home"] - f["p_open"]).abs() > LEAN_THRESHOLD
        for subset, s in (("all", f), (f"|p-open|>{LEAN_THRESHOLD}", f[lean])):
            r = clv_regression(s["p_home"], s["p_open"], s["p_close"])
            rows.append({"model": m, "subset": subset, "n": r["n"], "slope": round(r["slope"], 4),
                         "95% CI": f"[{r['lo95']:+.4f}, {r['hi95']:+.4f}]", "intercept": round(r["intercept"], 4),
                         "corr": round(r["corr"], 4)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    market = market_probs(DatasetStore().load("cfb_lines"))
    frames = {m: clv_frame(pd.read_parquet(BACKTEST / f"predictions_{args.tag}_{m}.parquet"), market) for m in args.models}
    text = (f"CLOSING-LINE VALUE {args.tag}: OLS of (p_close - p_open) on (p_model - p_open) with intercept; "
            f"{N_BOOT}-resample game bootstrap 95% CI on the slope; corr = Pearson of the two differences.\n"
            "Positive slope = the market moved toward the model between open and close.\n"
            + clv_table(frames).to_string(index=False))
    (BACKTEST / f"clv_{args.tag}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
