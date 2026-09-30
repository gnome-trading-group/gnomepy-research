"""Emit the signal catalog tables for tutorials/03_strategy_building.md.

The tables drifted badly while hand-maintained, so they are generated from the
package's own __all__ and docstrings instead. Run with --check in CI/tests.
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

import gnomepy_research.signals as S
from gnomepy_research.signals import book, fair_value, flow, market_state, operations, volatility

TUTORIAL = Path(__file__).resolve().parents[1] / "tutorials" / "03_strategy_building.md"
START = "<!-- BEGIN GENERATED SIGNAL CATALOG -->"
END = "<!-- END GENERATED SIGNAL CATALOG -->"

SECTIONS = [
    ("Fair Value", fair_value, "`Signal[int]` — scaled integer prices, same units as bid/ask."),
    ("Volatility", volatility, "`Signal[float]` — market noise and spread environment."),
    ("Flow", flow, "`Signal[float]` — trade activity and order-flow pressure."),
    ("Book", book, "`Signal[float]` — order book shape and imbalance."),
    ("Market State", market_state, "`Signal[float]` — regime and meta-market signals."),
]


def _params(obj) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    parts = []
    for name, p in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        parts.append(name if p.default is inspect.Parameter.empty else f"{name}={p.default!r}")
    return ", ".join(parts)


def _summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    return re.sub(r"\s+", " ", first) or "—"


# Pure helpers that are not themselves signals; documented in prose instead.
HELPERS = {"is_trade_event", "apply"}


def _is_concrete(name: str, obj) -> bool:
    """Signals are exported both as classes and as factory functions (e.g. VolOfVol)."""
    if obj is None or name.endswith("Signal") or name in HELPERS:
        return False
    if inspect.isclass(obj):
        return not inspect.isabstract(obj)
    return inspect.isfunction(obj)


def render() -> str:
    out = [START, ""]
    for title, module, blurb in SECTIONS:
        names = [n for n in getattr(module, "__all__", []) if _is_concrete(n, getattr(S, n, None))]
        out.append(f"### {title} — {len(names)} signals")
        out.append("")
        out.append(blurb)
        out.append("")
        out.append("| Signal | Parameters | What it measures |")
        out.append("|--------|-----------|------------------|")
        for n in names:
            obj = getattr(S, n)
            params = _params(obj)
            out.append(f"| `{n}` | {'`' + params + '`' if params else '—'} | {_summary(obj)} |")
        out.append("")

    # Adapters and composite plumbing are internal and are not re-exported at top level.
    op_names = [
        n for n in operations.__all__
        if hasattr(S, n)
        and not n.endswith("Operation") and n not in ("Operation", "apply")
        and not n.startswith("Weighted") and n != "PerAsset"
    ]
    out.append(f"### Operations — {len(op_names)} operators")
    out.append("")
    out.append(
        "Every operator wraps a signal and **preserves its type**, so an operator applied to a "
        "fair-value signal is still int-valued. Note the spelling `Zscore`, not `ZScore`."
    )
    out.append("")
    out.append("| Operator | Parameters | What it does |")
    out.append("|----------|-----------|--------------|")
    for n in op_names:
        obj = getattr(S, n)
        params = _params(obj)
        params = params.replace("signal, ", "").replace("signal", "")
        out.append(f"| `{n}` | {'`' + params + '`' if params else '—'} | {_summary(obj)} |")
    out.append("")
    out.append(
        "Each operator also has a lower-level `<Name>Operation` form (`ZscoreOperation`, "
        "`LagOperation`, ...) that you attach with `apply(op, signal)` when you want to build the "
        "operation once and reuse it."
    )
    out.append("")
    out.append(
        "Plus `PerAsset` to filter a signal to one listing, the `Weighted*` combiners "
        "(`WeightedFairValue`, `WeightedVolatility`, `WeightedFlow`, `WeightedBook`, "
        "`WeightedMarketState`) for weighted blends of same-type signals, and `is_trade_event(data)` "
        "to test whether a tick was a trade."
    )
    out.append("")
    out.append(END)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Exit non-zero if the tutorial is stale")
    args = ap.parse_args()

    text = TUTORIAL.read_text()
    if START not in text or END not in text:
        print(f"markers not found in {TUTORIAL}", file=sys.stderr)
        return 2

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    updated = head + render() + tail

    if args.check:
        if updated != text:
            print("signal catalog is stale — run: poetry run python scripts/gen_signal_catalog.py")
            return 1
        print("signal catalog is up to date")
        return 0

    TUTORIAL.write_text(updated)
    print(f"regenerated signal catalog in {TUTORIAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
