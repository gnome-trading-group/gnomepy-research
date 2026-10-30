from dataclasses import replace

from gnomepy_research.sessions.cs2_comp.e07.model import ModelConfig

_BASE = ModelConfig()
FINAL = replace(_BASE, hltv=replace(_BASE.hltv, tau=0.012, recency=0.8))

STACK_FACEIT = {"z_hltv": 0.808, "faceit_skill": 0.903, "rank_points": 0.226, "faceit_coverage": -0.377}
STACK_HLTV = {"z_hltv": 0.859, "rank_points": 0.212}
