"""Load backtest presets from YAML files in the ``presets/`` directory."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from gnomepy.backtest_config import backtest_from_dict, load_yaml

PRESETS_DIR = Path(__file__).parent.parent / "presets"


def load_preset(
    name: str,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
) -> "Backtest":
    """Load a preset by name and return a ready-to-run ``Backtest``.

    ``start_date`` and ``end_date`` are required — pass them here or
    include them in the YAML (runtime args take precedence).

    Example::

        from gnomepy_research.presets import load_preset
        from datetime import datetime

        backtest = load_preset(
            "momentum_btc_30m",
            start_date=datetime(2026, 1, 23, 10, 30),
            end_date=datetime(2026, 1, 23, 11, 0),
        )
        results = backtest.run()
    """
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_presets()) or "(none)"
        raise FileNotFoundError(f"No preset '{name}' found at {path}. Available: {available}")
    config = load_yaml(path)
    if start_date is not None:
        config["start_date"] = start_date
    if end_date is not None:
        config["end_date"] = end_date
    if "start_date" not in config or "end_date" not in config:
        raise ValueError("start_date and end_date must be provided either in the YAML or as arguments")
    backtest = backtest_from_dict(config)
    backtest._preset_config = config
    return backtest


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))
