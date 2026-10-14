"""Pre-registered backward elimination on the 2022 validation season (train 2020-2021) for model J.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.ablation --validate 2022

Outcome-layer terms first (G's split + conference draws, 3 seeds averaged), then preseason prior
feature groups with the reduced terms (normal likelihood, rolling hyperparameters 2019..S-1,
snapshots 2020-2022 per variant). Drop the item whose removal changes 2022 log loss by less than
THRESHOLD (best such item each round); stop when every removal costs more. 2023+ is never scored.
"""
from __future__ import annotations

import argparse
import json
from typing import Callable

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.data import CONF_SOURCES, UNITS_SPLIT
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR, draws_name, snapshot_season
from gnomepy_research.models.cfb.features.qb_status import apply_overrides, historical_starters, load_qb_status, qb_deltas
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import (
    FIRST_OUTCOME_SEASON, MODEL_CONFIGS, OUTCOME_FLAGS, QB_STATUS_CSV, RESULTS, VALIDATION_SEASON)
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import summarize
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import N_FIT_DRAWS, fit, game_draws

THRESHOLD = 0.0005
N_SEEDS = 3
LIKELIHOOD = "normal"
GROUPS: dict[str, tuple[str, ...]] = {
    "qb": ("qb_prior_ppa", "qb_returning_starter"),
    "coach": ("coach_srs_resid", "coach_new"),
    "recruiting_talent": ("talent_offense", "talent_defense", "talent_qb", "recruiting_points"),
    "portal": ("portal_in_stars",),
    "returning": ("returning_pct_ppa", "returning_pct_passing_ppa"),
    "conference_means": tuple(CONF_SOURCES),
}
J_FEATURES = RESULTS / "J_features.json"
ABLATION_TXT = RESULTS / f"ablation_{VALIDATION_SEASON}.txt"


def expand_groups(groups: list[str]) -> tuple[str, ...]:
    return tuple(sorted({f for g in groups for f in GROUPS[g]}))


def pick_removal(current: float, scores: dict[str, float], threshold: float = THRESHOLD) -> str | None:
    if not scores:
        return None
    best = min(scores, key=lambda k: scores[k])
    return best if scores[best] - current < threshold else None


def backward_eliminate(items: list[str], score: Callable[[tuple[str, ...]], float], threshold: float = THRESHOLD,
                       min_items: int = 1, log: Callable[[str], None] = lambda s: None) -> tuple[list[str], list[dict]]:
    kept = list(items)
    current = score(tuple(kept))
    log(f"start: {len(kept)} items, log loss {current:.5f}")
    steps, rnd = [], 0
    while len(kept) > min_items:
        rnd += 1
        scores = {}
        for c in kept:
            scores[c] = score(tuple(k for k in kept if k != c))
            log(f"round {rnd}: -{c}: {scores[c]:.5f} (delta {scores[c] - current:+.5f})")
        drop = pick_removal(current, scores, threshold)
        steps += [{"round": rnd, "candidate": c, "log_loss": v, "delta": v - current, "dropped": c == drop}
                  for c, v in scores.items()]
        if drop is None:
            break
        kept.remove(drop)
        current = scores[drop]
        log(f"round {rnd}: dropped {drop}; log loss now {current:.5f}")
    return kept, steps


class Scorer:
    """Average 2022 log loss over N_SEEDS outcome refits for a term subset and a prior-feature drop set."""

    def __init__(self, games: pd.DataFrame, qb: pd.DataFrame, flags: dict, form: bool):
        self.games, self.qb, self.flags, self.form = games, qb, flags, form
        self.train_seasons = list(range(FIRST_OUTCOME_SEASON, VALIDATION_SEASON))
        self._draws: dict[tuple, tuple] = {}
        self._cache: dict[tuple, float] = {}

    def _design(self, drop: tuple[str, ...]):
        if drop not in self._draws:
            kw = dict(units=UNITS_SPLIT, form=self.form, qb=self.qb, drop_features=drop, **self.flags)
            train = [game_draws(self.games, self.train_seasons, n_draws=N_FIT_DRAWS, seed=r, **kw) for r in range(N_SEEDS)]
            test = game_draws(self.games, [VALIDATION_SEASON], **kw)
            rows = test.games["home_win"].notna().to_numpy()
            self._draws[drop] = (train, test.x[rows], test.games.loc[rows, "home_win"].to_numpy(), test.terms)
        return self._draws[drop]

    def all_terms(self) -> list[str]:
        return list(self._design(())[3])

    def __call__(self, terms: tuple[str, ...], drop: tuple[str, ...] = ()) -> float:
        key = (tuple(terms), tuple(sorted(drop)))
        if key not in self._cache:
            train, x_test, y, all_terms = self._design(tuple(sorted(drop)))
            idx = [all_terms.index(t) for t in terms]
            lls = []
            for r, gd in enumerate(train):
                sub = type(gd)(games=gd.games, x=gd.x[..., idx], terms=list(terms))
                lls.append(summarize(y, fit(sub, seed=r).predict(x_test[..., idx])["p_home"].to_numpy())["log_loss"])
            self._cache[key] = float(np.mean(lls))
        return self._cache[key]


def ensure_variant(inputs: Inputs, drop: tuple[str, ...], conference: bool) -> None:
    for season in range(FIRST_OUTCOME_SEASON, VALIDATION_SEASON + 1):
        if not (DRAWS_DIR / f"{draws_name(season, UNITS_SPLIT, conference, drop_features=drop)}.npz").exists():
            snapshot_season(inputs, season, LIKELIHOOD, units=UNITS_SPLIT, conference=conference, drop_features=drop)


def format_steps(title: str, steps: list[dict]) -> str:
    if not steps:
        return f"{title}: no steps"
    t = pd.DataFrame(steps)
    t["dropped"] = np.where(t["dropped"], "DROP", "")
    return f"{title}:\n" + t.round({"log_loss": 5, "delta": 5}).to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, required=True)
    args = ap.parse_args()
    if args.validate != VALIDATION_SEASON:
        raise SystemExit(f"ablation only runs on the {VALIDATION_SEASON} validation season")
    cfg = MODEL_CONFIGS["G"]
    flags = {k: cfg[k] for k in OUTCOME_FLAGS if k in cfg}
    ds = DatasetStore()
    games = ds.load("cfb_games")
    src = SourceData.load(ds)
    starters = apply_overrides(historical_starters(src), load_qb_status(QB_STATUS_CSV, src))
    qb = qb_deltas(src, games[games["season"].between(FIRST_OUTCOME_SEASON, VALIDATION_SEASON)], starters)
    scorer = Scorer(games, qb, flags, cfg["form"])
    lines = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    terms = scorer.all_terms()
    full_ll = scorer(tuple(terms))
    log(f"G full: {len(terms)} terms {terms}; {VALIDATION_SEASON} log loss {full_ll:.5f} (mean of {N_SEEDS} seeds)")
    kept_terms, term_steps = backward_eliminate(terms, lambda k: scorer(k), log=log)
    reduced_ll = scorer(tuple(kept_terms))

    inputs = Inputs.load(ds, with_form=cfg["form"])

    def group_score(kept_groups: tuple[str, ...]) -> float:
        drop = expand_groups([g for g in GROUPS if g not in kept_groups])
        if drop:
            log(f"variant drop {list(drop)}: ensuring snapshots")
            ensure_variant(inputs, drop, flags["conference"])
        return scorer(tuple(kept_terms), drop)

    kept_groups, group_steps = backward_eliminate(list(GROUPS), group_score, min_items=0, log=log)
    drop_features = list(expand_groups([g for g in GROUPS if g not in kept_groups]))
    final_ll = scorer(tuple(kept_terms), tuple(drop_features))

    summary = (f"\nG full {VALIDATION_SEASON} log loss {full_ll:.5f}; reduced terms {reduced_ll:.5f}; "
               f"reduced terms + prior drops {final_ll:.5f}\nkept terms {kept_terms}\n"
               f"dropped groups {[g for g in GROUPS if g not in kept_groups]}\ndrop_features {drop_features}")
    text = "\n\n".join([
        f"model J ablation: train {scorer.train_seasons}, validate {VALIDATION_SEASON}; threshold {THRESHOLD}; "
        f"{N_SEEDS} seeds averaged",
        format_steps("outcome-term elimination", term_steps),
        format_steps("prior-group elimination", group_steps),
        summary, "log:\n" + "\n".join(lines)])
    RESULTS.mkdir(parents=True, exist_ok=True)
    ABLATION_TXT.write_text(text + "\n")
    J_FEATURES.write_text(json.dumps({"keep_terms": kept_terms, "drop_features": drop_features}, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
