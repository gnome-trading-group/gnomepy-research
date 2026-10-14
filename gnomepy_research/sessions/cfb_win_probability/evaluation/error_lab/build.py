"""Build the interactive error-distribution page for any set of backtest predictions.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.error_lab.build 2023_2024 \
        K:"Full priors + drives" G:"Conference-aware" MINE:"My model"=path/to/my_predictions.parquet

Each model spec is LABEL[:Name][=path]. Without a path, the predictions come from
results/backtest/predictions_<tag>_<LABEL>.parquet. A predictions file needs one row per game with
game_id, home_win and p_home (home team's win probability). Opening and closing lines are added
from cfb_lines. The page embeds per-game market data, so keep the output out of version control.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import BACKTEST, market_probs

TEMPLATE = Path(__file__).resolve().parent / "template.html"
OUT_DIR = BACKTEST.parent / "error_lab"
P4 = ["ACC", "Big 12", "Big Ten", "Pac-12", "SEC"]
SPEC = re.compile(r"^(?P<label>[A-Za-z0-9_+-]+)(?::(?P<name>[^=]+))?(?:=(?P<path>.+))?$")


def parse_spec(spec: str) -> tuple[str, str, Path | None]:
    m = SPEC.match(spec)
    if not m:
        raise ValueError(f"bad model spec {spec!r}; use LABEL[:Name][=path]")
    label = m.group("label")
    return label, (m.group("name") or label).strip(), Path(m.group("path")) if m.group("path") else None


def load_predictions(label: str, path: Path | None, tag: str) -> pd.Series:
    df = pd.read_parquet(path or BACKTEST / f"predictions_{tag}_{label}.parquet")
    missing = {"game_id", "home_win", "p_home"} - set(df.columns)
    if missing:
        raise ValueError(f"{label}: predictions file lacks {sorted(missing)}")
    return df.drop_duplicates("game_id").set_index("game_id")[["home_win", "p_home"]]


def preseason_ranks(coaches: pd.DataFrame) -> pd.Series:
    return coaches.dropna(subset=["preseason_rank"]).groupby(["season", "team_id"])["preseason_rank"].min()


def build_rows(preds: dict[str, pd.DataFrame], games: pd.DataFrame, market: pd.DataFrame,
               ranks: pd.Series) -> list[dict]:
    ids = sorted(set.intersection(*[set(p.index) for p in preds.values()]) & set(market.dropna().index))
    g = games.set_index("game_id")
    y = next(iter(preds.values()))["home_win"]
    rows = []
    for gid in ids:
        r = g.loc[gid]
        season = int(r["season"])
        hr, ar = ranks.get((season, int(r["home_id"]))), ranks.get((season, int(r["away_id"])))
        rows.append({
            "id": int(gid), "s": season, "w": int(r["week"]), "post": int(r["season_type"] == "postseason"),
            "neu": int(bool(r["neutral_site"])), "conf": int(bool(r["conference_game"])),
            "h": r["home_team"], "a": r["away_team"], "hc": r["home_conference"] or "", "ac": r["away_conference"] or "",
            "hr": None if hr is None or not np.isfinite(hr) else int(hr),
            "ar": None if ar is None or not np.isfinite(ar) else int(ar),
            "y": int(y.loc[gid]),
            "p": {**{m: round(float(p.loc[gid, "p_home"]), 4) for m, p in preds.items()},
                  "Open": round(float(market.loc[gid, "p_open"]), 4), "Close": round(float(market.loc[gid, "p_close"]), 4)},
        })
    return rows


def render(rows: list[dict], models: list[tuple[str, str]], tag: str) -> str:
    data = json.dumps({"rows": rows, "p4": P4, "tag": tag, "models": [{"label": l, "name": n} for l, n in models]},
                      separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("__DATA__", data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="backtest tag, e.g. 2023_2024")
    ap.add_argument("models", nargs="+", help="LABEL[:Name][=path] per model; up to 18 keep distinct styles")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    specs = [parse_spec(s) for s in args.models]
    preds = {label: load_predictions(label, path, args.tag) for label, _, path in specs}
    ds = DatasetStore()
    games, lines, coaches = ds.load("cfb_games"), ds.load("cfb_lines"), ds.load("cfb_coaches")
    market = market_probs(lines).set_index("game_id")[["p_open", "p_close"]]
    rows = build_rows(preds, games, market, preseason_ranks(coaches))
    out = args.out or OUT_DIR / f"error_lab_{args.tag}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, [(l, n) for l, n, _ in specs], args.tag))
    print(f"{len(rows)} games, {len(specs)} models -> {out}")


if __name__ == "__main__":
    main()
