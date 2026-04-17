"""Load backtest presets from YAML files in the ``presets/`` directory."""
from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import yaml

from gnomepy.java.backtest.runner import Backtest

PRESETS_DIR = Path(__file__).parent.parent / "presets"


def load_preset(
    name: str,
    strategy=None,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
) -> Backtest:
    """Load a preset by name and return a ready-to-run ``Backtest``.

    ``start_date`` and ``end_date`` are required — pass them here or
    include them in the YAML (runtime args take precedence).

    The optional ``strategy`` arg overrides the strategy specified in the YAML
    (useful for swapping in a custom Python strategy while keeping simulation
    config from the preset).

    Example::

        from gnomepy_research.presets import load_preset
        from datetime import datetime

        backtest = load_preset(
            "mm_btc_30m",
            start_date=datetime(2026, 1, 23, 10, 30),
            end_date=datetime(2026, 1, 23, 13, 0),
        )
        results = backtest.run()
    """
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_presets()) or "(none)"
        raise FileNotFoundError(f"No preset '{name}' found at {path}. Available: {available}")

    config = yaml.safe_load(path.read_text())

    if start_date is not None:
        config["start_date"] = _format_date(start_date)
    if end_date is not None:
        config["end_date"] = _format_date(end_date)

    if "start_date" not in config or "end_date" not in config:
        raise ValueError(
            "start_date and end_date must be provided either in the YAML or as arguments"
        )

    # Write the resolved config to a temp file so Java can parse it.
    # The temp file lives for the duration of this function call only;
    # Backtest() reads it synchronously during _build_driver(), which is
    # called lazily from run(). We keep the file alive until Backtest is
    # returned by using a NamedTemporaryFile with delete=False and cleaning
    # it up after run() is called — or just keep the file until GC via _tmp.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"gnomepy_preset_{name}_"
    )
    yaml.dump(config, tmp, default_flow_style=False, allow_unicode=True)
    tmp.flush()
    tmp.close()

    bt = Backtest(tmp.name, strategy=strategy)
    bt._preset_name = name
    bt._preset_config = config
    bt._preset_tmp = tmp.name  # caller can inspect; file is deleted after run()
    return bt


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def _format_date(d: date | datetime) -> str:
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%dT%H:%M:%S")
    return d.strftime("%Y-%m-%d")
