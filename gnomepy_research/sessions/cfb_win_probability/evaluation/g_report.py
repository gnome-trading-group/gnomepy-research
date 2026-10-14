"""Model G/H extras: hyperparameters, conference effects, and the motivating error-lab cuts.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.g_report 2023 2024
    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.g_report 2023 2024 --model H

Appends to results/backtest/report_<seasons>_<model>.txt, replacing any extras from an earlier run.
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
from gnomepy_research.models.cfb.efficiency.projection import project_seasons, projection_table
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss, summarize

P4 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
ACADEMIES = {"Army", "Navy", "Air Force"}
Z95 = 1.96
EXTRAS_MARKER = "\n\n==== EXTRAS (g_report) ====\n"
CUT_MODELS = {"G": ["D", "G"], "H": ["D", "G", "H"]}


def hyper_section(seasons: list[int], projection: bool = False) -> str:
    lines = [f"CONFERENCE HYPERPARAMETERS (normal split conf{' proj' if projection else ''}, fitted on 2019..S-1; "
             "aggregate NUTS diagnostics)"]
    for s in seasons:
        raw = json.loads((HYPER_DIR / f"{hyper_name('normal', s, UNITS_SPLIT, True, projection)}.json").read_text())
        h, d = raw["hyperparameters"], raw["diagnostics"]["hyper"]
        lines.append(f"{s}: s_conf_off={h['s_conf_off']:.4f} s_conf_def={h['s_conf_def']:.4f}  "
                     f"max_rhat={d['max_rhat']:.3f} ({d['worst_rhat_param']}) min_ess_bulk={d['min_ess_bulk']:.0f} "
                     f"({d['worst_ess_param']}) min_ess_tail={d['min_ess_tail']:.0f} divergences={d['divergences']}")
        betas = {**{n: b for n, b in zip(h["off_names"], h["beta_off"]) if "conf_" in n},
                 **{f"def/{n}": b for n, b in zip(h["def_names"], h["beta_def"]) if "conf_" in n}}
        lines.append("    conf betas (per sd of feature): " + ", ".join(f"{k}={v:+.4f}" for k, v in betas.items()))
        if projection:
            proj = {**{f"off/{n}": b for n, b in zip(h["off_names"], h["beta_off"]) if n.endswith(("proj", "proj_missing"))},
                    **{f"def/{n}": b for n, b in zip(h["def_names"], h["beta_def"]) if n.endswith(("proj", "proj_missing"))}}
            lines.append("    proj betas (posterior mean; proj per sd): " + ", ".join(f"{k}={v:+.4f}" for k, v in proj.items())
                         + f"; gamma (z_unc, last = low_continuity): {np.round(h['gamma'], 4).tolist()}")
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


def cut_frame(tag: str, models: list[str]) -> pd.DataFrame:
    """Last model's predictions with every model's p_home as p_<model> and the opening-line probability."""
    g = pd.read_parquet(RESULTS / f"predictions_{tag}_{models[-1]}.parquet")
    for m in models:
        p = pd.read_parquet(RESULTS / f"predictions_{tag}_{m}.parquet")[["game_id", "p_home"]]
        g = g.merge(p.rename(columns={"p_home": f"p_{m}"}), on="game_id")
    return g.merge(market_probs(DatasetStore().load("cfb_lines")).dropna(subset=["p_open"]), on="game_id")


def cut_rows(g: pd.DataFrame, cuts: dict[str, tuple[np.ndarray, np.ndarray]], models: list[str]) -> list[dict]:
    rows = []
    for name, (mask, flip) in cuts.items():
        s = g[mask]
        fl = flip[mask]
        y = np.where(fl, 1 - s["home_win"], s["home_win"]).astype(float)
        o = np.where(fl, 1 - s["p_open"], s["p_open"])
        ll_o = log_loss(y, o)
        ll = {}
        for m in models:
            p = np.where(fl, 1 - s[f"p_{m}"], s[f"p_{m}"])
            ll[m] = log_loss(y, p)
            rows.append({"cut": name, "model": m, "n": len(s), "bias y-p": _ci(y - p), "lean p-open": _ci(p - o),
                         "ll - open": _ci(ll[m] - ll_o), "calib_slope": round(summarize(y, p)["calib_slope"], 3)})
        if len(models) > 2:
            a, b = models[-1], models[-2]
            rows.append({"cut": name, "model": f"{a}-{b}", "n": len(s), "bias y-p": "", "lean p-open": "",
                         "ll - open": _ci(ll[a] - ll[b]), "calib_slope": np.nan})
    return rows


def cuts_section(tag: str, models: list[str]) -> str:
    g = cut_frame(tag, models)
    pair = f"; {models[-1]}-{models[-2]} row is the paired log-loss difference" if len(models) > 2 else ""
    return (f"{' vs '.join(models)} ON THE MOTIVATING CUTS (side-oriented; mean with 95% normal CI; "
            f"ll - open is paired per game{pair})\n" + pd.DataFrame(cut_rows(g, cut_sides(g), models)).to_string(index=False))


def home_returning_share(inputs: Inputs, seasons: list[int]) -> pd.DataFrame:
    """Home side's returning share: mean of the projection's returning_share over pass/rush x off/def."""
    table = projection_table(inputs.preseason, SourceData.load(), inputs.games, seasons)
    proj = project_seasons(table, seasons)
    return proj.groupby(["season", "team_id"], as_index=False)["returning_share"].mean()


def returning_section(inputs: Inputs, tag: str, seasons: list[int], models: list[str]) -> str:
    g = cut_frame(tag, models)
    share = home_returning_share(inputs, seasons).rename(columns={"team_id": "home_id", "returning_share": "home_ret"})
    g = g.merge(share.astype({"home_id": "int64"}), on=["season", "home_id"], how="left")
    early = ((g["season_type"] == "regular") & g["week"].between(1, 4)).to_numpy()
    terc = pd.qcut(g.loc[early, "home_ret"], 3, labels=["low", "mid", "high"])
    edges = g.loc[early, "home_ret"].quantile([0, 1 / 3, 2 / 3, 1]).round(3).tolist()
    none = np.zeros(len(g), bool)
    cuts = {f"wk1-4 home ret {t}": (early & (g["home_ret"].index.isin(terc[terc == t].index)), none)
            for t in ["low", "mid", "high"]}
    return (f"WEEKS 1-4 BY HOME RETURNING-SHARE TERCILE (home team's projection returning_share averaged over "
            f"pass/rush off/def; tercile edges {edges}; {int(g.loc[early, 'home_ret'].isna().sum())} early games lack a share)\n"
            + pd.DataFrame(cut_rows(g, cuts, models)).to_string(index=False))


def append_extras(path, text: str) -> None:
    base = path.read_text().split(EXTRAS_MARKER)[0].rstrip("\n")
    path.write_text(base + EXTRAS_MARKER + text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", type=int, nargs="+")
    ap.add_argument("--model", choices=sorted(CUT_MODELS), default="G")
    args = ap.parse_args()
    tag = "_".join(map(str, args.seasons))
    inputs = Inputs.load(with_form=False)
    models = CUT_MODELS[args.model]
    sections = [hyper_section(list(range(2020, max(args.seasons) + 1)), projection=args.model == "H")]
    if args.model == "G":
        sections.append(effects_section(inputs, args.seasons))
    sections.append(cuts_section(tag, models))
    if args.model == "H":
        sections.append(returning_section(inputs, tag, args.seasons, models))
    text = "\n\n".join(sections)
    append_extras(RESULTS / f"report_{tag}_{args.model}.txt", text)
    print(text)


if __name__ == "__main__":
    main()
