"""Guard the tutorials against drifting from the code they document.

The signal tables, Intent examples and constructor kwargs in
tutorials/03_strategy_building.md were all wrong before these checks existed.
"""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import gnomepy_research.signals as S
from gnomepy.java.oms import Intent

REPO = Path(__file__).resolve().parents[1]
TUTORIAL = REPO / "tutorials" / "03_strategy_building.md"
GENERATOR = REPO / "scripts" / "gen_signal_catalog.py"

# Docs that drive the analysis library through a `report` object.
REPORT_DOCS = [
    REPO / ".claude" / "commands" / "research.md",
    REPO / "tutorials" / "05_analysis_toolkit.md",
]


@pytest.fixture(scope="module")
def tutorial() -> str:
    return TUTORIAL.read_text()


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.S)


def test_signal_catalog_is_regenerated():
    """tutorials/03's tables must match the package's __all__ and docstrings."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, (
        f"{result.stdout.strip()}\n{result.stderr.strip()}"
    )


DOCUMENTED = [
    n for n in S.__all__
    if not n.endswith("Signal") and n != "Signal" and not n.endswith("Operation")
]


@pytest.mark.parametrize("name", DOCUMENTED)
def test_every_exported_signal_is_documented(name: str, tutorial: str):
    """An exported name a strategy author could reach for must appear in the tutorial."""
    # Match the name inside backticks, allowing a call signature after it: `apply(op, signal)`.
    assert re.search(rf"`{re.escape(name)}[`(]", tutorial), (
        f"{name} is exported from gnomepy_research.signals but undocumented in {TUTORIAL.name}"
    )


@pytest.mark.parametrize("name", [n for n in S.__all__ if n.endswith("Operation") and n != "Operation"])
def test_every_operation_has_a_documented_helper(name: str, tutorial: str):
    """The low-level <X>Operation forms are covered by documenting their <X> helper."""
    helper = name[: -len("Operation")]
    assert helper in DOCUMENTED, f"{name} has no {helper} helper exported"
    assert re.search(rf"`{re.escape(helper)}[`(]", tutorial), (
        f"{name} is reachable via {helper}(), which is undocumented in {TUTORIAL.name}"
    )


def test_intent_kwargs_are_real(tutorial: str):
    """Every Intent(...) example must construct against the real signature."""
    valid = set(inspect.signature(Intent.__init__).parameters) - {"self"}
    used: set[str] = set()
    for block in _python_blocks(tutorial):
        for call in re.findall(r"Intent\((.*?)\n\s*\)", block, re.S):
            used |= set(re.findall(r"^\s*(\w+)\s*=", call, re.M))
    assert used, "no Intent examples found — did the tutorial change shape?"
    assert used <= valid, f"unknown Intent kwargs: {sorted(used - valid)}"


def test_signal_constructor_kwargs_are_real(tutorial: str):
    """Documented signal constructor arguments must exist on the signal."""
    bad: list[str] = []
    for block in _python_blocks(tutorial):
        for name, args in re.findall(r"\b([A-Z][A-Za-z]+)\(([^()]*)\)", block):
            obj = getattr(S, name, None)
            if obj is None:
                continue
            try:
                params = set(inspect.signature(obj).parameters)
            except (TypeError, ValueError):
                continue
            for kw in re.findall(r"(\w+)\s*=", args):
                if kw not in params:
                    bad.append(f"{name}(...{kw}=...)")
    assert not bad, f"stale signal kwargs in tutorial: {sorted(set(bad))}"


def test_side_convention_is_stated_correctly(tutorial: str):
    """Side.BID buys and Side.ASK sells; the tutorial once said the opposite."""
    assert "`Side.BID` buys and `Side.ASK` sells" in tutorial
    assert "ASK = buy" not in tutorial


def _embedded_python(text: str) -> list[str]:
    """Python from ```python fences AND from heredocs inside ```bash fences.

    research.md runs most of its Python as `poetry run python3 - <<'EOF' ... EOF`
    inside a bash fence, so a python-fence-only scan misses the Step 5 diagnostics
    entirely — which is how the broken accessor survived the first version of this test.
    """
    blocks = list(_python_blocks(text))
    for bash in re.findall(r"```bash\n(.*?)```", text, re.S):
        blocks += [body for _, body in re.findall(r"<<'?(\w+)'?\n(.*?)\n\s*\1\b", bash, re.S)]
    return blocks


def _report_attributes(text: str) -> set[str]:
    """Attributes accessed on a `report` object in any embedded Python."""
    used: set[str] = set()
    for block in _embedded_python(text):
        used |= set(re.findall(r"\breport\.(\w+)", block))
    return used


@pytest.mark.parametrize("doc", REPORT_DOCS, ids=lambda p: p.name)
def test_documented_report_attributes_exist(doc: Path):
    """Every `report.X` in the docs must exist on BacktestReport.

    Import-only validation missed this: `report.fills_df()` is a real method — on
    BacktestResults, not on the BacktestReport that load_results() returns. The
    docs shipped broken because every *symbol* resolved.
    """
    from gnomepy.reporting.report import BacktestReport

    used = _report_attributes(doc.read_text())
    assert used, f"no report.* usage found in {doc.name} — did the docs change shape?"
    missing = sorted(a for a in used if not hasattr(BacktestReport, a))
    assert not missing, (
        f"{doc.name} uses BacktestReport attributes that do not exist: {missing}"
    )


def test_docs_do_not_teach_private_attribute_access():
    """The analysis helpers have public accessors now; docs must not reach past them.

    Matches attribute access specifically — `load_results` / `compare_results` are
    public function names that happen to contain `_results`.
    """
    private = re.compile(r"\breport\._\w+|\.\_(market_df|intent_df|exec_df|results)\b")
    for doc in REPORT_DOCS:
        hits = private.findall(doc.read_text())
        assert not hits, f"{doc.name} reaches into private attributes: {sorted(set(hits))}"
