from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from gnomepy.data import DatasetStore

ROOT = Path("gnomepy_research/sessions/cfb_win_probability/results/backtest")
SD = 15.5
BAND_EDGES = [0, 3, 7, 14, 21, 70]
BAND_LABELS = ["0-3", "3-7", "7-14", "14-21", "21+"]


def slope(x, y) -> float:
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def band_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean margin by size of the open favorite, from the favorite's side. df needs open_margin,
    model_margin, actual and optionally margin_mean."""
    fav = np.sign(df["open_margin"]).replace(0, 1)
    band = pd.cut(df["open_margin"].abs(), BAND_EDGES, labels=BAND_LABELS, include_lowest=True)
    cols = {"open": df["open_margin"] * fav, "model_p": df["model_margin"] * fav, "actual": df["actual"] * fav}
    if "margin_mean" in df:
        cols["model_mean"] = df["margin_mean"] * fav
    g = pd.DataFrame(cols).assign(band=band).groupby("band", observed=True)
    out = g.mean()
    out.insert(0, "n", g.size())
    return out


def diagnose(df: pd.DataFrame) -> dict:
    return {"model_vs_open": slope(df["open_margin"], df["model_margin"]),
            "actual_vs_model": slope(df["model_margin"], df["actual"]),
            "actual_vs_open": slope(df["open_margin"], df["actual"]),
            "sd_model": float(df["model_margin"].std()), "sd_open": float(df["open_margin"].std()),
            "bands": band_table(df)}


def load_frame(tag: str, model: str, games: pd.DataFrame, open_spread: pd.Series) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / f"predictions_{tag}_{model}.parquet").drop_duplicates("game_id").set_index("game_id")
    df = pd.DataFrame({"p": p["p_home"].clip(1e-4, 1 - 1e-4)})
    df["model_margin"] = SD * norm.ppf(df["p"])
    if "margin_mean" in p:
        df["margin_mean"] = p["margin_mean"]
    df["open_margin"] = -open_spread.reindex(df.index)
    g = games.loc[df.index]
    df["actual"] = g["home_points"] - g["away_points"]
    return df.dropna()


def render(model: str, d: dict) -> str:
    head = (f"== {model}: model-vs-open slope {d['model_vs_open']:.3f} | actual~model slope {d['actual_vs_model']:.3f} | "
            f"actual~open slope {d['actual_vs_open']:.3f} | sd model {d['sd_model']:.1f} vs open {d['sd_open']:.1f}")
    return head + "\n" + d["bands"].round(1).to_string()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    ds = DatasetStore()
    games = ds.load("cfb_games").set_index("game_id")
    open_spread = ds.load("cfb_lines").groupby("game_id")["spread_open"].median()
    text = "\n\n".join(render(m, diagnose(load_frame(args.tag, m, games, open_spread))) for m in args.models)
    (ROOT / f"compression_{args.tag}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
