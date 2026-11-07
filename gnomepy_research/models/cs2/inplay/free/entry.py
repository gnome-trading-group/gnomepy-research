"""Benchmark plus in-map form in the map model: economy-adjusted round residuals and lagged kill differentials,
overall, per side and recency-weighted; see core.py and NOTES.md."""
from core import CFG, fit, predict  # noqa: F401

CFG.update(map_extra=["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean", "res_dec", "kill_dec"],
           map_kw=dict(min_child_weight=200), expect="rich", decay=0.4)
