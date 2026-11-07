"""Benchmark plus in-map form (economy-adjusted round residual and lagged kill differential) in the map model; see
core.py and NOTES.md."""
from core import CFG, fit, predict  # noqa: F401

CFG.update(map_extra=["res_mean", "kill_mean"], map_kw=dict(min_child_weight=200))
