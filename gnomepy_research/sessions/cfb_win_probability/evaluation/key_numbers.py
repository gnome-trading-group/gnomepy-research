"""Key-number-aware conversion from a point spread to a home win probability, and a re-score of the open.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.key_numbers 2023_2024

Final margins cluster on 3, 7, 10 and 14, so the win probability is not a smooth function of the spread:
moving from -2.5 to -3.5 is worth more than -4.5 to -5.5. Instead of a normal curve, use the empirical
distribution of r = margin + spread (how far the result landed from the line) from earlier seasons:
P(home wins | spread s) = P(r > s), with a light kernel so half-point steps stay monotone.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import BACKTEST, SPREAD_SD
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import paired_bootstrap
from gnomepy_research.sessions.cfb_win_probability.live.wp_data import pregame_prior

KERNEL_SD = 0.35


def residuals(games: pd.DataFrame, spread: pd.Series) -> pd.DataFrame:
    g = games.set_index("game_id")
    done = g["completed"].fillna(False).astype(bool)
    margin = (g["home_points"] - g["away_points"])[done]
    df = pd.DataFrame({"season": g.loc[margin.index, "season"], "margin": margin, "spread": spread.reindex(margin.index)})
    df = df.dropna()
    df["r"] = df["margin"] + df["spread"]
    return df


def win_prob(spread: np.ndarray, r: np.ndarray, kernel_sd: float = KERNEL_SD) -> np.ndarray:
    """P(r > s) under the empirical residual distribution, smoothed by a narrow normal kernel."""
    s = np.asarray(spread, float)[:, None]
    return norm.sf((s - np.asarray(r, float)[None, :]) / kernel_sd).mean(axis=1)


def log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="2023_2024")
    ap.add_argument("--model", default="K")
    args = ap.parse_args()
    ds = DatasetStore()
    games, lines = ds.load("cfb_games"), ds.load("cfb_lines")
    open_spread = lines.groupby("game_id")["spread_open"].median()
    res = residuals(games, open_spread)
    pred = pd.read_parquet(BACKTEST / f"predictions_{args.tag}_{args.model}.parquet").drop_duplicates("game_id").set_index("game_id")
    test = pred.join(open_spread.rename("spread_open"), how="inner").dropna(subset=["spread_open"])
    test["p_close"] = pregame_prior(lines).reindex(test.index)
    test = test.dropna(subset=["p_close"])
    seasons = sorted(test["season"].unique())
    test["p_open_normal"] = norm.cdf(-test["spread_open"] / SPREAD_SD)
    test["p_open_keys"] = np.nan
    for s in seasons:
        train = res[res["season"] < s]["r"].to_numpy()
        m = test["season"] == s
        test.loc[m, "p_open_keys"] = win_prob(test.loc[m, "spread_open"].to_numpy(), train)
    y = test["home_win"].to_numpy()
    P = {k: np.clip(test[c].to_numpy(float), 1e-4, 1 - 1e-4) for k, c in
         [("open, normal curve (old)", "p_open_normal"), ("open, key numbers (new)", "p_open_keys"),
          ("close", "p_close"), (f"model {args.model}", "p_home")]}
    L = {k: log_loss(y, p) for k, p in P.items()}
    lines_out = [f"{len(test)} games; key-number conversion fit on seasons before each test season "
                 f"({len(res[res['season'] < seasons[0]])}+ games)", "", "log loss:"]
    lines_out += [f"  {k:28s} {v.mean():.4f}" for k, v in L.items()]
    lines_out += ["", "paired differences (first - second; negative = first better):"]
    for a, b in [("open, key numbers (new)", "open, normal curve (old)"),
                 (f"model {args.model}", "open, normal curve (old)"), (f"model {args.model}", "open, key numbers (new)")]:
        r = paired_bootstrap(y, P[a], P[b])
        lines_out.append(f"  {a} - {b}: {r['diff']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]")
    grid = np.arange(-14, 0.5, 0.5)
    train = res[res["season"] < seasons[-1]]["r"].to_numpy()
    grid = grid[::-1]
    pk, pn = win_prob(grid, train), norm.cdf(-grid / SPREAD_SD)
    lines_out += ["", "home win prob by home spread (negative = home favored), key numbers vs normal curve:"]
    lines_out += [f"  {s:+5.1f}  keys {a:.3f}  normal {b:.3f}  step {a - c:+.3f}" for s, a, b, c in
                  zip(grid, pk, pn, np.r_[np.nan, pk[:-1]]) if s <= -0.5]
    text = "\n".join(lines_out)
    (BACKTEST / f"key_numbers_{args.tag}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
