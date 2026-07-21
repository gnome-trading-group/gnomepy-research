"""Capture reproducibility metadata for a research iteration."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from importlib.metadata import version as _pkg_version


def _git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def capture_environment(metadata_json_path: Path | None = None) -> dict:
    """Collect reproducibility metadata for an iteration record.

    Reads from metadata.json (written by gnomepy after the backtest) and
    supplements with local git state and Python runtime info.
    """
    env: dict = {}

    if metadata_json_path and Path(metadata_json_path).exists():
        meta = json.loads(Path(metadata_json_path).read_text())
        for key in ("gnomepy_version", "gnomepy_research_version", "gnomepy_commit",
                    "backtest_jar_hash", "java_version", "python_version", "os_info"):
            if meta.get(key):
                env[key] = meta[key]

    if not env.get("gnomepy_version"):
        try:
            env["gnomepy_version"] = _pkg_version("gnomepy")
        except Exception:
            pass

    if not env.get("gnomepy_research_version"):
        try:
            env["gnomepy_research_version"] = _pkg_version("gnomepy_research")
        except Exception:
            pass

    if not env.get("python_version"):
        env["python_version"] = platform.python_version()

    if not env.get("os_info"):
        env["os_info"] = platform.platform()

    research_repo = Path(__file__).parents[1]
    env["research_commit"] = _git_commit(research_repo)

    return {k: v for k, v in env.items() if v is not None}
