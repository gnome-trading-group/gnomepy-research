"""Score backtested models against each other and against the opening and closing lines.

Usage: python -m gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models 2023_2024 A B C [D] [E] [F] [G] [C+E]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss, paired_bootstrap, summarize
from gnomepy_research.sessions.cfb_win_probability.live.wp_data import pregame_prior

BACKTEST = Path(__file__).resolve().parent.parent / "results" / "backtest"
SPREAD_SD = 15.5
BET_THRESHOLD = 0.05
N_BLEND_BOOT = 200
PAIRS = [("B", "A"), ("C", "B"), ("C", "A"), ("D", "C"), ("E", "C"), ("C+E", "C"), ("C+E", "E"), ("F", "C"), ("F", "E"), ("G", "D"), ("G", "C"), ("H", "G"), ("H", "D"), ("I", "G"), ("I0", "G"), ("I", "I0"), ("J0", "G"), ("J", "G"), ("J", "J0"), ("K", "G"), ("K", "J"), ("L", "K"), ("L", "G")]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def market_probs(lines: pd.DataFrame) -> pd.DataFrame:
    open_spread = lines.groupby("game_id")["spread_open"].median()
    out = pd.DataFrame({"p_open": norm.cdf(-open_spread / SPREAD_SD)}, index=open_spread.index)
    out["p_close"] = pregame_prior(lines)
    return out.rename_axis("game_id").reset_index()


def _blend_fit(y: np.ndarray, p_model: np.ndarray, p_ref: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([logit(p_model), logit(p_ref)])
    coef = LogisticRegression(C=1e6, fit_intercept=False).fit(X, y).coef_[0]
    return float(coef[0]), float(coef[1])


def blend_weight(y: np.ndarray, p_model: np.ndarray, p_ref: np.ndarray, n_boot: int = N_BLEND_BOOT,
                 seed: int = 0) -> dict:
    a, b = _blend_fit(y, p_model, p_ref)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        boots.append(_blend_fit(y[idx], p_model[idx], p_ref[idx])[0])
    return {"a": a, "b": b, "lo95": float(np.quantile(boots, 0.025)), "hi95": float(np.quantile(boots, 0.975))}


def _fmt_diff(r: dict) -> str:
    return f"{r['diff']:+.4f}  [{r['lo95']:+.4f}, {r['hi95']:+.4f}]  P(first worse)={r['p_a_worse']:.2f}"


def _source_table(y: np.ndarray, seasons: np.ndarray, sources: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, p in sources.items():
        for label, mask in [("all", np.ones(len(y), bool))] + [(str(s), seasons == s) for s in sorted(set(seasons))]:
            s = summarize(y[mask], p[mask])
            rows.append({"source": name, "season": label, "n": s["n"], "log_loss": s["log_loss"],
                         "brier": s["brier"], "calib_slope": s["calib_slope"]})
    return pd.DataFrame(rows)


def blend_frames(preds: dict[str, pd.DataFrame], labels: list[str]) -> dict[str, pd.DataFrame]:
    """Add each `+` label as the mean p_home of its component models on the games they share."""
    out = dict(preds)
    for label in labels:
        if "+" not in label:
            continue
        parts = [preds[c].drop_duplicates("game_id").set_index("game_id") for c in label.split("+")]
        shared = parts[0].index
        for p in parts[1:]:
            shared = shared.intersection(p.index)
        frame = parts[0].loc[shared].copy()
        frame["p_home"] = np.mean([p.loc[shared, "p_home"].to_numpy() for p in parts], axis=0)
        out[label] = frame.reset_index()
    return out


def compare(preds: dict[str, pd.DataFrame], market: pd.DataFrame) -> str:
    models = list(preds)
    games = set.intersection(*(set(df["game_id"]) for df in preds.values()))
    n_models_common = len(games)
    mk = market.dropna(subset=["p_open", "p_close"]).set_index("game_id")
    keep = sorted(games & set(mk.index))
    dropped = n_models_common - len(keep)

    first = preds[models[0]].drop_duplicates("game_id").set_index("game_id").loc[keep]
    y = first["home_win"].to_numpy().astype(int)
    seasons = first["season"].to_numpy()
    p = {m: preds[m].drop_duplicates("game_id").set_index("game_id").loc[keep, "p_home"].to_numpy() for m in models}
    p_open, p_close = mk.loc[keep, "p_open"].to_numpy(), mk.loc[keep, "p_close"].to_numpy()

    sanity = ""
    if "p_market_close" in first.columns:
        diff = (first["p_market_close"].to_numpy() - p_close)
        n_diff = int((np.abs(diff) > 0.02).sum())
        sanity = f"sanity: {n_diff} of {len(keep)} games have |backtest p_market_close - p_close| > 0.02"

    ll = {m: log_loss(y, p[m]).mean() for m in models}
    ll_open, ll_close = log_loss(y, p_open).mean(), log_loss(y, p_close).mean()
    best = min(models, key=ll.get)
    vs_open = {m: paired_bootstrap(y, p[m], p_open) for m in models}
    vs_close = {m: paired_bootstrap(y, p[m], p_close) for m in models}
    blend_open = {m: blend_weight(y, p[m], p_open) for m in models}
    blend_close = {m: blend_weight(y, p[m], p_close) for m in models}
    informative = [m for m in models if blend_open[m]["lo95"] > 0]

    summary = [
        "SUMMARY",
        f"- Best model by log loss: {best} ({ll[best]:.4f}); open {ll_open:.4f}; close {ll_close:.4f}.",
        f"- Gap to open (model - open, negative = model better): {_fmt_diff(vs_open[best])}",
        f"- Gap to close (model - close): {_fmt_diff(vs_close[best])}",
        "- Blend weight a on the model vs the open (a > 0 = information beyond the open): "
        + ", ".join(f"{m} a={blend_open[m]['a']:+.2f} [{blend_open[m]['lo95']:+.2f}, {blend_open[m]['hi95']:+.2f}]"
                    for m in models),
        ("- Carries information beyond the open (95% CI of a above 0): " + ", ".join(informative))
        if informative else "- No model's blend weight CI excludes 0 vs the open: no demonstrated information beyond the open.",
    ]

    lines = [*summary, "", f"models: {' '.join(models)}; games scored: {len(keep)}; "
             f"dropped {dropped} games present in every model but missing an opening or closing line"]
    if sanity:
        lines.append(sanity)

    sources = {**{f"model {m}": p[m] for m in models}, "open": p_open, "close": p_close}
    lines += ["", "PER-SOURCE SCORES (calib_slope: 1 = calibrated)",
              _source_table(y, seasons, sources).to_string(index=False, float_format=lambda v: f"{v:.4f}")]

    lines += ["", "PAIRED LOG-LOSS DIFFERENCES (first - second; negative = first better; game-bootstrap 95% CI)"]
    for a, b in PAIRS:
        if a in p and b in p:
            lines.append(f"diff {a} - {b}  {_fmt_diff(paired_bootstrap(y, p[a], p[b]))}")
    for m in models:
        lines.append(f"diff {m} - open  {_fmt_diff(vs_open[m])}")
    for m in models:
        lines.append(f"diff {m} - close  {_fmt_diff(vs_close[m])}")

    lines += ["", "BLEND: logit(y) ~ a*logit(model) + b*logit(ref); a > 0 = model carries information the reference lacks"]
    for m in models:
        bo, bc = blend_open[m], blend_close[m]
        lines.append(f"{m} vs open:  a={bo['a']:+.3f} [{bo['lo95']:+.3f}, {bo['hi95']:+.3f}]  b={bo['b']:+.3f}")
        lines.append(f"{m} vs close: a={bc['a']:+.3f} [{bc['lo95']:+.3f}, {bc['hi95']:+.3f}]  b={bc['b']:+.3f}  (reference)")

    lines += ["", f"BET-LIKE SUBSET: |p_model - p_open| > {BET_THRESHOLD} (log loss; open scored on the same games)"]
    for m in models:
        mask = np.abs(p[m] - p_open) > BET_THRESHOLD
        if mask.sum():
            lines.append(f"{m}: n={int(mask.sum())}  model={log_loss(y[mask], p[m][mask]).mean():.4f}  "
                         f"open={log_loss(y[mask], p_open[mask]).mean():.4f}")
        else:
            lines.append(f"{m}: n=0")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    singles = sorted({c for m in args.models for c in m.split("+")})
    loaded = {m: pd.read_parquet(BACKTEST / f"predictions_{args.tag}_{m}.parquet") for m in singles}
    preds = {m: blend_frames(loaded, args.models)[m] for m in args.models}
    report = compare(preds, market_probs(DatasetStore().load("cfb_lines")))
    out = BACKTEST / f"comparison_{args.tag}.txt"
    out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
