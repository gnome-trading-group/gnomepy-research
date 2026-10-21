"""
Fit PlayerSkillConfig by the filter's one-step-ahead map log loss over a training window.

Every map from the start of history warms the filter; only maps in [start, end) are scored, and history from
end onwards is dropped so the fit cannot see it. Nelder-Mead on transformed settings (logs for the scales, a logit
for rho): each evaluation is a full pass over history, a few seconds, and the loss is not smooth enough in the
chemistry and best-player terms for a gradient method to gain much.

    python -m gnomepy_research.sessions.cs2_win_probability.fit_player_skill --start 2025-09-01 --end 2026-01-01

Reads cs2_match_history, cs2_team_rankings and cs2_match_lineups parquet files from --data-dir and prints the
fitted settings to paste into PlayerSkillConfig's defaults.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig, compute_player_skill_features
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

logger = logging.getLogger(__name__)


def _from_vector(v: np.ndarray, base: PlayerSkillConfig) -> PlayerSkillConfig:
    return dataclasses.replace(
        base, tau=math.exp(v[0]), sigma0=math.exp(v[1]), a=v[2], b=v[3], lam=v[4], kappa_max=v[5],
        c0=math.exp(v[6]), delta=v[7], rho=1 / (1 + math.exp(-v[8])),
    )


def _to_vector(c: PlayerSkillConfig) -> np.ndarray:
    return np.array([math.log(c.tau), math.log(c.sigma0), c.a, c.b, c.lam, c.kappa_max, math.log(c.c0), c.delta,
                     math.log(c.rho / (1 - c.rho))])


def map_log_loss(cfg: PlayerSkillConfig, history: pd.DataFrame, rankings: pd.DataFrame, lineups: pd.DataFrame,
                 start: pd.Timestamp, end: pd.Timestamp) -> float:
    feats = compute_player_skill_features(history, rankings, lineups, cfg)
    scored = history[(history.match_date >= start) & (history.match_date < end)
                     & (history.team_a_score + history.team_b_score > 0)]
    j = scored[["match_id", "map_name", "team_a_won"]].merge(feats[["match_id", "map_name", "ps_p_map"]],
                                                             on=["match_id", "map_name"]).dropna()
    p = np.clip(j.ps_p_map.to_numpy(), 1e-4, 1 - 1e-4)
    y = j.team_a_won.to_numpy().astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit(history: pd.DataFrame, rankings: pd.DataFrame, lineups: pd.DataFrame, start: pd.Timestamp,
        end: pd.Timestamp, x0: PlayerSkillConfig, max_evals: int) -> tuple[float, PlayerSkillConfig]:
    history = history.assign(match_date=pd.to_datetime(history.match_date).dt.normalize())
    history = history[history.match_date < end]
    best: list = [np.inf, x0]

    def objective(v: np.ndarray) -> float:
        cfg = _from_vector(v, x0)
        if not (0 <= cfg.lam <= 1 and cfg.kappa_max >= 0 and cfg.delta >= 0 and cfg.c0 < 500):
            return 1.0
        try:
            loss = map_log_loss(cfg, history, rankings, lineups, start, end)
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            return 1.0
        if loss < best[0]:
            best[:] = [loss, cfg]
            logger.info("map log loss %.5f  %s", loss, cfg)
        return loss

    minimize(objective, _to_vector(x0), method="Nelder-Mead",
             options={"maxfev": max_evals, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True})
    return best[0], best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first scored day")
    ap.add_argument("--end", required=True, help="first day dropped from history (exclusive)")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--max-evals", type=int, default=450)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("gnomepy_research.models.cs2.hltv.pit").setLevel(logging.WARNING)

    history = pd.read_parquet(args.data_dir / "cs2_match_history.parquet")
    rankings = pd.read_parquet(args.data_dir / "cs2_team_rankings.parquet")
    lineups = pd.read_parquet(args.data_dir / "cs2_match_lineups.parquet")
    loss, cfg = fit(history, rankings, lineups, pd.Timestamp(args.start), pd.Timestamp(args.end),
                    PlayerSkillConfig(), args.max_evals)
    print(f"map log loss {loss:.5f} on [{args.start}, {args.end})")
    for f in ("tau", "sigma0", "a", "b", "lam", "kappa_max", "c0", "delta", "rho"):
        print(f"    {f}: float = {float(getattr(cfg, f))!r}")


if __name__ == "__main__":
    main()
