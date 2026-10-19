"""Walk-forward validation for backtesting.

Divides a total date range into n_folds evaluation windows and runs the best
strategy on each, reporting per-fold and aggregate OOS metrics.

No refitting happens between folds — the strategy's parameters are fixed from the
session's best iteration — so these are evaluation windows, not train/test splits.
``step_mode`` selects how the windows are laid out; see ``_generate_fold_windows``.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WalkForwardConfig:
    total_start: datetime
    total_end: datetime
    n_folds: int = 5
    step_mode: str = "rolling"


@dataclass
class FoldResult:
    fold_index: int
    test_start: datetime
    test_end: datetime
    summary: dict[str, Any]
    output_dir: str


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def oos_sharpes(self) -> list[float]:
        return [f.summary.get("sharpe", 0.0) for f in self.folds]

    @property
    def oos_pnls(self) -> list[float]:
        return [f.summary.get("final_pnl", 0.0) for f in self.folds]

    @property
    def mean_oos_sharpe(self) -> float:
        sharpes = self.oos_sharpes
        return sum(sharpes) / len(sharpes) if sharpes else 0.0

    @property
    def mean_oos_pnl(self) -> float:
        pnls = self.oos_pnls
        return sum(pnls) / len(pnls) if pnls else 0.0

    @property
    def pct_positive_folds(self) -> float:
        pnls = self.oos_pnls
        if not pnls:
            return 0.0
        return sum(1 for p in pnls if p > 0) / len(pnls)

    def to_dict(self) -> dict:
        return {
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "test_start": f.test_start.isoformat(),
                    "test_end": f.test_end.isoformat(),
                    "summary": f.summary,
                    "output_dir": f.output_dir,
                }
                for f in self.folds
            ],
            "mean_oos_sharpe": self.mean_oos_sharpe,
            "mean_oos_pnl": self.mean_oos_pnl,
            "pct_positive_folds": self.pct_positive_folds,
            "oos_sharpes": self.oos_sharpes,
        }


STEP_MODES = ("expanding", "rolling")


def _generate_fold_windows(cfg: WalkForwardConfig) -> list[tuple[datetime, datetime]]:
    """Lay out each fold's evaluation window.

    ``rolling`` gives n_folds equal, disjoint windows marching forward, so each fold
    scores a distinct stretch of the range. ``expanding`` anchors every window at
    ``total_start`` and grows it, so fold N scores everything seen up to that point and
    the last fold covers the whole range.
    """
    if cfg.step_mode not in STEP_MODES:
        raise ValueError(f"step_mode must be one of {STEP_MODES}, got {cfg.step_mode!r}")
    if cfg.n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {cfg.n_folds}")
    if cfg.total_end <= cfg.total_start:
        raise ValueError("total_end must be after total_start")

    step = (cfg.total_end - cfg.total_start) / cfg.n_folds
    if cfg.step_mode == "expanding":
        return [
            (cfg.total_start, cfg.total_start + (i + 1) * step)
            for i in range(cfg.n_folds)
        ]
    return [
        (cfg.total_start + i * step, cfg.total_start + (i + 1) * step)
        for i in range(cfg.n_folds)
    ]


def run_walk_forward_local(
    base_config_path: str | Path,
    walk_forward_config: WalkForwardConfig,
    output_base_dir: str | Path,
    project_root: str | Path | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation locally via subprocess."""
    base_config_path = Path(base_config_path)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    fold_windows = _generate_fold_windows(walk_forward_config)
    result = WalkForwardResult()

    for i, (fold_start, fold_end) in enumerate(fold_windows):
        fold_config = dict(base_config)
        fold_config["start_date"] = fold_start.isoformat()
        fold_config["end_date"] = fold_end.isoformat()

        fold_output_dir = output_base_dir / f"fold_{i + 1:03d}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(fold_config, tmp)
            tmp_path = tmp.name

        try:
            subprocess.run(
                [
                    "poetry", "run", "gnomepy", "backtest", "run",
                    "--config", tmp_path,
                    "--output", str(fold_output_dir),
                ],
                check=True,
                cwd=str(project_root) if project_root else None,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        summary_path = fold_output_dir / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        result.folds.append(FoldResult(
            fold_index=i + 1,
            test_start=fold_start,
            test_end=fold_end,
            summary=summary,
            output_dir=str(fold_output_dir),
        ))

    return result
