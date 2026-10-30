from dataclasses import replace

from gnomepy_research.sessions.cs2_comp.e07.model import ModelConfig

_BASE = ModelConfig()
FINAL = replace(_BASE, hltv=replace(_BASE.hltv, tau=0.012, recency=0.8, sigma0=0.45, b=0.3), newcomer_prob=0.1)

STACK_FACEIT = {"z_hltv": 0.991, "faceit_skill": 0.565, "rank_points": 0.054}
STACK_HLTV = {"z_hltv": 1.019, "rank_points": 0.067}
