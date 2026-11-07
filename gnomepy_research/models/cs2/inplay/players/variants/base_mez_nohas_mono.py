"""Configuration base_mez_nohas_mono of the players entry."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(pm="base", stack="mez", map_nohas=True, map_mono=True, round_feats=False, round_prior=False)
