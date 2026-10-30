from dataclasses import replace

from gnomepy_research.sessions.cs2_comp.e07.model import ModelConfig

_BASE = ModelConfig()
FINAL = replace(_BASE, hltv=replace(_BASE.hltv, tau=0.012, recency=0.8, sigma0=0.45, b=0.3), newcomer_scale=1.0)

# Fitted on HLTV bo3 series of Jul 2025 - Mar 2026 (the walk-forward in dev.py); every tune target is after that window.
STACK_FACEIT = {"z_hltv": 0.863, "faceit_skill": 0.785, "faceit_coverage": -0.409}
STACK_HLTV = {"z_hltv": 0.887}
