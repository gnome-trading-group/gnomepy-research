"""
Which dependency extras a strategy needs, and the locked versions to install them at.

Extras are named for what they provide (xgboost, pymc, websocket, research). A package
declares the ones its code needs with EXTRAS in its __init__.py, e.g. EXTRAS = ("xgboost",),
and the nearest enclosing package that declares EXTRAS decides for every module in it; a
strategy whose packages declare none runs on the base install.
gnome-orchestrator's start-session.sh runs this file before installing anything, so it
uses only the standard library and is run by path, not imported as part of the package:

    python3 gnomepy_research/dependencies.py extras gnomepy_research.sessions.cs2_series.strategy:CS2Series
    python3 gnomepy_research/dependencies.py constraints    # rewrite constraints.txt after `poetry lock`

constraints.txt pins every package to its poetry.lock version, so a session installs
exactly what the tests ran against. gnomepy is left out: the launcher pins it per session.
"""
from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "gnomepy_research"
CONSTRAINTS = "constraints.txt"
UNCONSTRAINED = {"gnomepy"}
HEADER = "# Generated from poetry.lock by `python3 gnomepy_research/dependencies.py constraints` - do not edit.\n"


def declared_extras(init_file: Path) -> tuple[str, ...] | None:
    """The EXTRAS a package's __init__.py assigns, or None if it assigns none."""
    for node in ast.parse(init_file.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXTRAS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    return None


def module_extras(module: str, root: Path = ROOT) -> tuple[str, ...]:
    """The extras for a dotted module name: the nearest enclosing package that declares EXTRAS decides."""
    parts = module.split(".")
    if parts[0] != PACKAGE:
        raise ValueError(f"{module} is not in {PACKAGE}")
    for depth in range(len(parts), 0, -1):
        init_file = root.joinpath(*parts[:depth], "__init__.py")
        if init_file.exists() and (extras := declared_extras(init_file)) is not None:
            return extras
    return ()


def strategy_extras(strategy_class: str, root: Path = ROOT) -> tuple[str, ...]:
    """The extras for `package.module:Class`."""
    return module_extras(strategy_class.split(":")[0], root)


def constraints(lock_text: str) -> str:
    packages = sorted((p["name"], p["version"]) for p in tomllib.loads(lock_text)["package"])
    return HEADER + "".join(f"{name}=={version}\n" for name, version in packages if name not in UNCONSTRAINED)


def main(argv: list[str]) -> None:
    match argv:
        case ["extras", strategy_class]:
            print(",".join(strategy_extras(strategy_class)))
        case ["constraints"]:
            (ROOT / CONSTRAINTS).write_text(constraints((ROOT / "poetry.lock").read_text()))
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
