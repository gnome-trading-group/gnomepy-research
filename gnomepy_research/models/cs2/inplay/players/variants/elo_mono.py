"""Configuration elo_mono of the players entry."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(map_prior="elo", stack="none", map_nohas=False, map_mono=True, round_feats=False, round_prior=False)
