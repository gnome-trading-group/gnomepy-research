"""Team-week posterior snapshots for every weekly cutoff, with rolling-origin hyperparameters.

    poetry run python -m gnomepy_research.models.cfb.efficiency.posterior_snapshots 2020 2026

gnome-pipelines publishes the outputs (gnome_pipelines.cfb.posterior).

Season S uses hyperparameters fitted on 2019..S-1 only, so nothing about S (or later) shapes its
priors. 2019 is the first season with prior-season ratings, so it only ever trains.

Outputs:
- dataset cfb_team_week_posterior: one row per (cutoff, team) with posterior means/sds, O-D
  correlation, data counts and versions (the spec's team-week contract)
- artifact posterior_draws/cfb_team_week_draws_<season>: npz with thinned joint draws of
  (alpha, O, D) per cutoff, for the outcome model and the future live engine
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency.data import (
    EXTENSIONS, UNITS_ALL, UNITS_SPLIT, canonical_extensions, season_periods)
from gnomepy_research.models.cfb.efficiency.gaussian_solver import fit_cutoff_exact
from gnomepy_research.models.cfb.efficiency.projection import project_seasons, projection_table
from gnomepy_research.models.cfb.features.drives import GARBAGE_WEIGHT
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.models.cfb.features.team_form import FORM_FEATURES, team_form, with_luck
from gnomepy_research.models.cfb.features.rush_share import league_rush_share, rush_share
from gnomepy_research.models.cfb.features.style import option_offense
from gnomepy_research.models.cfb.efficiency.inference import (
    EFFICIENCY,
    HYPER_DIR,
    PRESEASON_FEATURES,
    Inputs,
    fit_cutoff,
    fit_hyperparameters,
    load_hyperparameters,
    save_hyperparameters,
)

SPLIT_SUFFIX = "_split"
CONF_SUFFIX = "_conf"
IND_SUFFIX = "_ind"
PROJ_SUFFIX = "_proj"
DRIVES_SUFFIX = "_drv"
TAU_PHASES_SUFFIX = "_tph"
HEAVY_TAILS_SUFFIX = "_ht"
EXTENSION_SUFFIX = {"couple": "_cpl", "fcs": "_fcs", "finish": "_fin"}
FIRST_TRAIN_SEASON = 2019
N_DRAWS_KEPT = 400
HEAVY_TAILS_MCMC = {"draws": 300, "tune": 300, "chains": 2}
# Hyperparameter fits with observation-model extensions mix tau more slowly; a longer tune keeps rhat < 1.02.
EXTENSION_HYPER_MCMC = {"draws": 1500, "tune": 3000}
DRAWS_DIR = Path(__file__).resolve().parents[3] / "sessions" / "cfb_win_probability" / "results" / "posterior_draws"


def drop_suffix(drop_features: tuple[str, ...] = ()) -> str:
    if not drop_features:
        return ""
    return "_drop-" + hashlib.sha1(",".join(sorted(drop_features)).encode()).hexdigest()[:8]


def garbage_suffix(garbage_weight: float = GARBAGE_WEIGHT) -> str:
    return "" if garbage_weight == GARBAGE_WEIGHT else f"_gw{int(round(garbage_weight * 100)):03d}"


def _suffix(units: tuple[str, ...], conference: bool = False, projection: bool = False,
            drop_features: tuple[str, ...] = (), drives: bool = False, independents_separate: bool = False,
            tau_phases: bool = False, heavy_tails: bool = False, garbage_weight: float = GARBAGE_WEIGHT,
            extensions: tuple[str, ...] = ()) -> str:
    """Order: split, conf, ind, proj, drv, gw, tph, ht, cpl, fcs, fin, drop."""
    if garbage_weight != GARBAGE_WEIGHT and not drives:
        raise ValueError("a garbage-time weight only applies with drives=True")
    ext = "".join(EXTENSION_SUFFIX[e] for e in canonical_extensions(extensions, drives))
    return (("" if units == UNITS_ALL else SPLIT_SUFFIX) + (CONF_SUFFIX if conference else "")
            + (IND_SUFFIX if independents_separate else "") + (PROJ_SUFFIX if projection else "") + (DRIVES_SUFFIX if drives else "")
            + garbage_suffix(garbage_weight) + (TAU_PHASES_SUFFIX if tau_phases else "") + (HEAVY_TAILS_SUFFIX if heavy_tails else "")
            + ext + drop_suffix(drop_features))


def model_version(units: tuple[str, ...], conference: bool = False, projection: bool = False,
                  drives: bool = False, independents_separate: bool = False, tau_phases: bool = False,
                  heavy_tails: bool = False, garbage_weight: float = GARBAGE_WEIGHT,
                  extensions: tuple[str, ...] = ()) -> str:
    suffix = _suffix(units, conference, projection, drives=drives, independents_separate=independents_separate,
                     tau_phases=tau_phases, heavy_tails=heavy_tails, garbage_weight=garbage_weight,
                     extensions=extensions)
    return f"dyn-eff{suffix.replace('_', '-')}-1"


MODEL_VERSION = model_version(UNITS_ALL)


def hyper_name(likelihood: str, season: int, units: tuple[str, ...] = UNITS_ALL, conference: bool = False,
               projection: bool = False, drop_features: tuple[str, ...] = (), drives: bool = False,
               independents_separate: bool = False, tau_phases: bool = False, heavy_tails: bool = False,
               garbage_weight: float = GARBAGE_WEIGHT, extensions: tuple[str, ...] = ()) -> str:
    suffix = _suffix(units, conference, projection, drop_features, drives, independents_separate, tau_phases,
                     heavy_tails, garbage_weight, extensions)
    return f"{likelihood}{suffix}_{FIRST_TRAIN_SEASON}_{season - 1}"


def draws_name(season: int, units: tuple[str, ...] = UNITS_ALL, conference: bool = False,
               projection: bool = False, drop_features: tuple[str, ...] = (), drives: bool = False,
               independents_separate: bool = False, tau_phases: bool = False, heavy_tails: bool = False,
               garbage_weight: float = GARBAGE_WEIGHT, extensions: tuple[str, ...] = ()) -> str:
    sfx = _suffix(units, conference, projection, drop_features, drives, independents_separate, tau_phases, heavy_tails,
                  garbage_weight, extensions)
    return f"cfb_team_week_draws{sfx}_{season}"


def ensure_hyperparameters(inputs: Inputs, likelihood: str, season: int, units: tuple[str, ...] = UNITS_ALL,
                           conference: bool = False, projection: bool = False,
                           drop_features: tuple[str, ...] = (), drives: bool = False,
                           independents_separate: bool = False, tau_phases: bool = False,
                           heavy_tails: bool = False, garbage_weight: float = GARBAGE_WEIGHT,
                           extensions: tuple[str, ...] = ()) -> str:
    name = hyper_name(likelihood, season, units, conference, projection, drop_features, drives,
                      independents_separate, tau_phases, heavy_tails, garbage_weight, extensions)
    if not (HYPER_DIR / f"{name}.json").exists():
        h, diag = fit_hyperparameters(inputs, list(range(FIRST_TRAIN_SEASON, season)), likelihood, units=units,
                                      conference=conference, projection=projection, drop_features=drop_features,
                                      drives=drives, independents_separate=independents_separate,
                                      tau_phases=tau_phases, heavy_tails=heavy_tails,
                                      garbage_weight=garbage_weight, extensions=extensions,
                                      **(EXTENSION_HYPER_MCMC if extensions else {}))
        save_hyperparameters(h, diag, name)
        print(f"fitted {name}: {diag['hyper']}", flush=True)
    return name


def form_at(form_rows: pd.DataFrame, season: int, cutoff: pd.Timestamp, team_ids: np.ndarray) -> np.ndarray:
    visible = form_rows[(form_rows["available_at"] < cutoff) & (form_rows["season"] == season)]
    f = team_form(with_luck(visible), season) if len(visible) else pd.DataFrame(columns=FORM_FEATURES)
    return f.reindex(team_ids)[FORM_FEATURES].astype(float).fillna(0.0).to_numpy(np.float32)


def snapshot_season(inputs: Inputs, season: int, likelihood: str, seed: int = 0,
                    out_dir: Path = DRAWS_DIR, units: tuple[str, ...] = UNITS_ALL,
                    conference: bool = False, projection: bool = False,
                    drop_features: tuple[str, ...] = (), drives: bool = False,
                    independents_separate: bool = False, tau_phases: bool = False,
                    heavy_tails: bool = False, mcmc: dict | None = None,
                    garbage_weight: float = GARBAGE_WEIGHT,
                    extensions: tuple[str, ...] = ()) -> tuple[pd.DataFrame, Path]:
    """mcmc: draws/tune/chains of each cutoff's MCMC fit under heavy_tails (default HEAVY_TAILS_MCMC)."""
    name = ensure_hyperparameters(inputs, likelihood, season, units, conference, projection, drop_features, drives,
                                  independents_separate, tau_phases, heavy_tails, garbage_weight, extensions)
    h = load_hyperparameters(name)
    rng = np.random.default_rng(seed)
    summaries, draws = [], {}
    e = inputs.efficiency
    for p in season_periods(inputs.games, [season]).itertuples():
        # The Gaussian model has a closed-form posterior (validated against NUTS); Student-t
        # (likelihood or heavy-tailed start states) needs MCMC.
        if h.heavy_tails:
            post, _ = fit_cutoff(inputs, season, p.cutoff, h, **(mcmc or HEAVY_TAILS_MCMC), seed=seed)
        else:
            fit = fit_cutoff_exact if likelihood == "normal" else fit_cutoff
            post, _ = fit(inputs, season, p.cutoff, h, seed=seed)
        keep = rng.choice(post.off.shape[0], size=min(N_DRAWS_KEPT, post.off.shape[0]), replace=False)
        key = f"{p.season_type}_{p.week}"
        draws[f"{key}__off"] = post.off[keep].astype(np.float32)
        draws[f"{key}__def"] = post.de[keep].astype(np.float32)
        draws[f"{key}__alpha"] = post.alpha[keep].astype(np.float32)
        draws[f"{key}__team_ids"] = post.team_ids
        if post.fin_off is not None:
            draws[f"{key}__fin_off"] = post.fin_off[keep].astype(np.float32)
            draws[f"{key}__fin_def"] = post.fin_def[keep].astype(np.float32)
        seen = e[(e["available_at"] < p.cutoff) & (e["season"] == season) & e["usable"]]
        shares = rush_share(seen, season)
        lg = league_rush_share(seen, season)
        draws[f"{key}__units"] = post.units.astype("U8")
        draws[f"{key}__rush_share"] = shares.reindex(post.team_ids).fillna(lg).to_numpy(np.float32)
        if conference:
            style_rows = e[(e["available_at"] < p.cutoff) & e["season"].isin([season - 1, season]) & e["usable"]]
            draws[f"{key}__option"] = option_offense(style_rows, season).reindex(post.team_ids).fillna(0).to_numpy(np.float32)
        if inputs.form_rows is not None:
            draws[f"{key}__form"] = form_at(inputs.form_rows, season, p.cutoff, post.team_ids)
            draws[f"{key}__form_names"] = np.array(FORM_FEATURES)
        counts = seen.groupby("team_id").agg(games_observed=("game_id", "nunique"), plays_observed=("plays", "sum"))
        s = post.summary().merge(counts, left_on="team_id", right_index=True, how="left")
        s[["games_observed", "plays_observed"]] = s[["games_observed", "plays_observed"]].fillna(0)
        summaries.append(s.assign(season_type=p.season_type, week=p.week, period=p.period,
                                  max_rhat=post.diagnostics["max_rhat"],
                                  divergences=post.diagnostics["divergences"]))
        print(f"{season} {key}: rhat {post.diagnostics['max_rhat']:.3f} div {post.diagnostics['divergences']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{draws_name(season, units, conference, projection, drop_features, drives, independents_separate, tau_phases, heavy_tails, garbage_weight, extensions)}.npz"
    np.savez_compressed(path, **draws)
    out = pd.concat(summaries, ignore_index=True).assign(
        model_version=model_version(units, conference, projection, drives, independents_separate, tau_phases, heavy_tails,
                                    garbage_weight, extensions),
        hyperparameters=name, likelihood=likelihood,
        training_data_version=f"{PRESEASON_FEATURES},{EFFICIENCY}@{inputs.as_of.isoformat()}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return out, path


def snapshot_seasons(inputs: Inputs, first: int, last: int, likelihood: str,
                     out_dir: Path = DRAWS_DIR, units: tuple[str, ...] = UNITS_ALL,
                     conference: bool = False, projection: bool = False,
                     drop_features: tuple[str, ...] = (), drives: bool = False,
                     independents_separate: bool = False,
                     tau_phases: bool = False, heavy_tails: bool = False,
                     garbage_weight: float = GARBAGE_WEIGHT,
                     mcmc: dict | None = None,
                     extensions: tuple[str, ...] = ()) -> tuple[pd.DataFrame, list[tuple[int, Path]]]:
    """The team-week table for seasons first..last, and each season's draws file under out_dir."""
    if first <= FIRST_TRAIN_SEASON:
        raise ValueError(f"first snapshot season must be after {FIRST_TRAIN_SEASON}")
    if projection and inputs.projection is None:
        raise ValueError("projection=True needs Inputs.projection (projection.project_seasons)")
    frames, paths = [], []
    for season in range(first, last + 1):
        df, path = snapshot_season(inputs, season, likelihood, out_dir=out_dir, units=units, conference=conference,
                                     projection=projection, drop_features=drop_features, drives=drives,
                                     independents_separate=independents_separate, tau_phases=tau_phases,
                                     heavy_tails=heavy_tails, garbage_weight=garbage_weight, mcmc=mcmc,
                                     extensions=extensions)
        frames.append(df)
        paths.append((season, path))
    table = pd.concat(frames, ignore_index=True)
    sfx = _suffix(units, conference, projection, drop_features, drives, independents_separate, tau_phases, heavy_tails,
                  garbage_weight, extensions)
    table.to_parquet(out_dir / f"cfb_team_week_posterior{sfx}_{first}_{last}.parquet", index=False)
    return table, paths


def read_drop_features(path: Path | str) -> tuple[str, ...]:
    return tuple(json.loads(Path(path).read_text())["drop_features"])


def snapshot_options(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--likelihood", default="studentt")
    ap.add_argument("--units", choices=["all", "split"], default="all")
    ap.add_argument("--form", action="store_true")
    ap.add_argument("--conference", action="store_true")
    ap.add_argument("--projection", action="store_true")
    ap.add_argument("--drives", action="store_true")
    ap.add_argument("--independents-separate", action="store_true")
    ap.add_argument("--tau-phases", action="store_true")
    ap.add_argument("--heavy-tails", action="store_true")
    ap.add_argument("--garbage-weight", type=float, default=GARBAGE_WEIGHT)
    ap.add_argument("--mcmc-draws", type=int, default=HEAVY_TAILS_MCMC["draws"])
    ap.add_argument("--mcmc-tune", type=int, default=HEAVY_TAILS_MCMC["tune"])
    ap.add_argument("--mcmc-chains", type=int, default=HEAVY_TAILS_MCMC["chains"])
    ap.add_argument("--drop-features-from", type=Path, default=None)
    ap.add_argument("--extensions", default="", help=f"comma-separated subset of {','.join(EXTENSIONS)}")
    args = ap.parse_args(argv)
    return {"first": args.first, "last": args.last, "likelihood": args.likelihood,
            "units": UNITS_ALL if args.units == "all" else UNITS_SPLIT, "form": args.form,
            "conference": args.conference, "projection": args.projection, "drives": args.drives,
            "independents_separate": args.independents_separate, "tau_phases": args.tau_phases,
            "heavy_tails": args.heavy_tails, "garbage_weight": args.garbage_weight,
            "mcmc": {"draws": args.mcmc_draws, "tune": args.mcmc_tune, "chains": args.mcmc_chains},
            "drop_features": read_drop_features(args.drop_features_from) if args.drop_features_from else (),
            "extensions": canonical_extensions([e for e in args.extensions.split(",") if e], args.drives)}


def main() -> None:
    o = snapshot_options()
    inputs = Inputs.load(with_form=o["form"], with_drives=o["drives"], garbage_weight=o["garbage_weight"])
    if o["projection"]:
        seasons = list(range(FIRST_TRAIN_SEASON, o["last"] + 1))
        table = projection_table(inputs.preseason, SourceData.load(), inputs.games, seasons)
        inputs.projection = project_seasons(table, seasons)
    table, _ = snapshot_seasons(inputs, o["first"], o["last"], o["likelihood"], units=o["units"],
                                conference=o["conference"], projection=o["projection"],
                                drop_features=o["drop_features"], drives=o["drives"],
                                independents_separate=o["independents_separate"], tau_phases=o["tau_phases"],
                                heavy_tails=o["heavy_tails"], garbage_weight=o["garbage_weight"], mcmc=o["mcmc"],
                                extensions=o["extensions"])
    print(f"{len(table):,} team-week rows")


if __name__ == "__main__":
    main()
