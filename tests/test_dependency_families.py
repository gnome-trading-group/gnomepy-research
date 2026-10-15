"""
Each part of the package imports only what its declared extras install.

A module gets the base plus the EXTRAS of the nearest package that declares them, the same
rule a live session's install follows (gnomepy_research/dependencies.py); gnome-pipelines
images install only the extras their pipelines need. An undeclared import would only fail
there, so it fails here. Imports are followed through the package: a session declaring
xgboost that reaches pymc through a model package fails too. What is allowed is everything
the extras' requirements pull in, read from the installed package metadata.

Not caught: a gnomepy module that itself needs a gnomepy extra (gnomepy.reporting needs plotly).
"""
from __future__ import annotations

import ast
import sys
import tomllib
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from gnomepy_research.dependencies import PACKAGE, ROOT, constraints, declared_extras, module_extras, strategy_extras

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
EXTRAS = tuple(PYPROJECT["optional-dependencies"])
# Research tooling: never part of a live session or a pipeline image without the research extra.
RESEARCH_TOOLING = ("analysis", "reporting", "explore", "cli", "validation", "notes_sync")


def _requirement_closure(requirements: list[str]) -> frozenset[str]:
    seen: set[tuple[str, frozenset[str]]] = set()
    stack = [Requirement(r) for r in requirements]
    while stack:
        req = stack.pop()
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        key = (canonicalize_name(req.name), frozenset(req.extras))
        if key in seen:
            continue
        seen.add(key)
        try:
            requires = distribution(req.name).requires or []
        except PackageNotFoundError:
            continue
        for spec in map(Requirement, requires):
            if spec.marker is None or any(spec.marker.evaluate({"extra": e}) for e in ("", *req.extras)):
                stack.append(Requirement(f"{spec.name}{'[' + ','.join(spec.extras) + ']' if spec.extras else ''}"))
    return frozenset(name for name, _ in seen)


@cache
def allowed(extras: tuple[str, ...]) -> frozenset[str]:
    reqs = list(PYPROJECT["dependencies"])
    for extra in extras:
        reqs += PYPROJECT["optional-dependencies"][extra]
    return _requirement_closure(reqs)


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES = {_module_name(p): p for p in (ROOT / PACKAGE).rglob("*.py")}


def _resolve(name: str) -> str | None:
    while name and name not in MODULES:
        name = name.rpartition(".")[0]
    return name or None


@cache
def direct_imports(module: str) -> tuple[frozenset[str], frozenset[str]]:
    """(third-party top-level names, package modules) a module imports."""
    path = MODULES[module]
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    external, internal = set(), set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1] + ([base] if base else []))
            names = [f"{base}.{a.name}" for a in node.names] + [base]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top == PACKAGE:
                if (target := _resolve(name)) is not None:
                    internal.add(target)
            elif top and top not in sys.stdlib_module_names and top != "__future__":
                external.add(top)
    return frozenset(external), frozenset(internal - {module})


def reached(module: str) -> dict[str, str]:
    """Every third-party top-level name the module reaches, mapped to the package module that imports it."""
    found: dict[str, str] = {}
    seen, stack = set(), [module]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        parts = current.split(".")
        stack += [".".join(parts[:i]) for i in range(1, len(parts)) if ".".join(parts[:i]) in MODULES]
        external, internal = direct_imports(current)
        for name in external:
            found.setdefault(name, current)
        stack += internal
    return found


def family(module: str) -> tuple[str, ...]:
    if module.split(".")[1:2] and module.split(".")[1] in RESEARCH_TOOLING:
        return EXTRAS
    return module_extras(module)


def test_every_module_stays_inside_its_family():
    dists = packages_distributions()
    violations = []
    for module in sorted(MODULES):
        extras = family(module)
        ok = allowed(extras)
        for name, via in sorted(reached(module).items()):
            if not any(canonicalize_name(d) in ok for d in dists.get(name, [name])):
                violations.append(f"{module} [{'+'.join(('base', *extras))}] reaches {name} (imported by {via})")
    assert not violations, "imports outside their dependency family:\n" + "\n".join(violations)


def test_declared_extras_exist():
    for init_file in (ROOT / PACKAGE).rglob("__init__.py"):
        assert set(declared_extras(init_file) or ()) <= set(EXTRAS), init_file


def test_extras_come_from_the_nearest_package():
    assert strategy_extras("gnomepy_research.sessions.cs2_series.strategy:CS2Series") == ("xgboost",)
    assert module_extras("gnomepy_research.models.cfb.efficiency.inference") == ("pymc",)
    assert module_extras("gnomepy_research.models.cfb.features.source") == ()
    assert strategy_extras("gnomepy_research.strategies.random_trader:RandomTrader") == ()
    with pytest.raises(ValueError):
        strategy_extras("somewhere_else.module:Strategy")


def test_constraints_match_the_lock():
    assert (ROOT / "constraints.txt").read_text() == constraints((ROOT / "poetry.lock").read_text()), \
        "constraints.txt is stale: run `python3 gnomepy_research/dependencies.py constraints`"
