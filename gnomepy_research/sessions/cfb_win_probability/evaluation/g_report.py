"""Model G extras: conference hyperparameters, conference effects, and D vs G on the error-lab cuts.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.g_report 2023 2024

Appends to results/backtest/report_<seasons>_G.txt.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT, build_model_data, season_periods
from gnomepy_research.models.cfb.efficiency.gaussian_solver import _layout, solve
from gnomepy_research.models.cfb.efficiency.inference import HYPER_DIR, Inputs, load_hyperparameters
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import hyper_name
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss, summarize

P4 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
ACADEMIES = {"Army", "Navy", "Air Force"}
Z95 = 1.96


def hyper_section(seasons: list[int]) -> str:
    lines = ["CONFERENCE HYPERPARAMETERS (normal split conf, fitted on 2019..S-1; aggregate NUTS diagnostics)"]
    for s in seasons:
        raw = json.loads((HYPER_DIR / f"{hyper_name('normal', s, UNITS_SPLIT, True)}.json").read_text())
        h, d = raw["hyperparameters"], raw["diagnostics"]["hyper"]
        lines.append(f"{s}: s_conf_off={h['s_conf_off']:.4f} s_conf_def={h['s_conf_def']:.4f}  "
                     f"max_rhat={d['max_rhat']:.3f} ({d['worst_rhat_param']}) min_ess_bulk={d['min_ess_bulk']:.0f} "
                     f"({d['worst_ess_param']}) min_ess_tail={d['min_ess_tail']:.0f} divergences={d['divergences']}")
        betas = {**{n: b for n, b in zip(h["off_names"], h["beta_off"]) if "conf_" in n},
                 **{f"def/{n}": b for n, b in zip(h["def_names"], h["beta_def"]) if "conf_" in n}}
        lines.append("    conf betas (per sd of feature): " + ", ".join(f"{k}={v:+.4f}" for k, v in betas.items()))
    return "\n".join(lines)


def conference_effects(inputs: Inputs, season: int) -> pd.DataFrame:
    h = load_hyperparameters(hyper_name("normal", season, UNITS_SPLIT, True))
    per = season_periods(inputs.games, [season])
    cutoff = per.loc[per["season_type"] == "regular", "cutoff"].max()
    data = build_model_data(inputs.preseason, inputs.visible(cutoff), inputs.games, [season],
                            units=UNITS_SPLIT, conference=True)
    mean, _, lay = solve(data, h)
    n = len(data.conf_keys)
    k = data.conf_keys.assign(off=h.s_conf_off * mean[lay["z_conf_off"]:lay["z_conf_off"] + n],
                              de=h.s_conf_def * mean[lay["z_conf_def"]:lay["z_conf_def"] + n])
    out = k.pivot_table(index="conference", columns="unit", values=["off", "de"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["net"] = out.sum(axis=1)
    return out.sort_values("net", ascending=False)


def effects_section(inputs: Inputs, seasons: list[int]) -> str:
    lines = ["CONFERENCE EFFECTS at the final regular-season cutoff (posterior mean s_conf*z_conf, PPA/play; "
             "net = sum of off and def over pass and rush; higher = stronger)"]
    for s in seasons:
        e = conference_effects(inputs, s).round(4)
        lines.append(f"{s} top 3:\n{e.head(3).to_string()}\n{s} bottom 3:\n{e.tail(3).to_string()}")
    return "\n".join(lines)


def _ci(v: np.ndarray) -> str:
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return f"{m:+.4f} [{m - Z95 * se:+.4f}, {m + Z95 * se:+.4f}]"


def cut_sides(g: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cut name -> (row mask, flip to the away side?)."""
    p4 = lambda conf, team: conf.isin(P4) | (team == "Notre Dame")
    hp, ap = p4(g["home_conference"], g["home_team"]), p4(g["away_conference"], g["away_team"])
    reg = g["season_type"] == "regular"
    ha, aa = g["home_team"].isin(ACADEMIES), g["away_team"].isin(ACADEMIES)
    o = g["p_open"]
    none = np.zeros(len(g), bool)
    return {
        "favorites": (np.ones(len(g), bool), (o < 0.5).to_numpy()),
        "big favorites": (((o >= 0.8) | (o <= 0.2)).to_numpy(), (o <= 0.2).to_numpy()),
        "P4 vs G5": ((hp ^ ap).to_numpy(), ap.to_numpy()),
        "non-conference": ((~g["conference_game"].astype(bool)).to_numpy(), none),
        "weeks 1-4": ((reg & g["week"].between(1, 4)).to_numpy(), none),
        "weeks 5-9": ((reg & g["week"].between(5, 9)).to_numpy(), none),
        "service academies": ((ha | aa).to_numpy(), (~ha & aa).to_numpy()),
        "toss-ups": (o.between(0.4, 0.6).to_numpy(), none),
    }


def cuts_section(tag: str) -> str:
    d = pd.read_parquet(RESULTS / f"predictions_{tag}_D.parquet")
    g = pd.read_parquet(RESULTS / f"predictions_{tag}_G.parquet")
    g = g.merge(d[["game_id", "p_home"]].rename(columns={"p_home": "p_d"}), on="game_id")
    g = g.merge(market_probs(DatasetStore().load("cfb_lines")).dropna(subset=["p_open"]), on="game_id")
    rows = []
    for name, (mask, flip) in cut_sides(g).items():
        s = g[mask]
        fl = flip[mask]
        y = np.where(fl, 1 - s["home_win"], s["home_win"]).astype(float)
        o = np.where(fl, 1 - s["p_open"], s["p_open"])
        ll_o = log_loss(y, o)
        for m, col in [("D", "p_d"), ("G", "p_home")]:
            p = np.where(fl, 1 - s[col], s[col])
            rows.append({"cut": name, "model": m, "n": len(s), "bias y-p": _ci(y - p), "lean p-open": _ci(p - o),
                         "ll - open": _ci(log_loss(y, p) - ll_o), "calib_slope": round(summarize(y, p)["calib_slope"], 3)})
    return ("D vs G ON THE MOTIVATING CUTS (side-oriented; mean with 95% normal CI; ll - open is paired per game)\n"
            + pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", type=int, nargs="+")
    args = ap.parse_args()
    tag = "_".join(map(str, args.seasons))
    inputs = Inputs.load(with_form=False)
    text = "\n\n".join([hyper_section(list(range(2020, max(args.seasons) + 1))),
                        effects_section(inputs, args.seasons), cuts_section(tag)])
    path = RESULTS / f"report_{tag}_G.txt"
    path.write_text(path.read_text() + "\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
