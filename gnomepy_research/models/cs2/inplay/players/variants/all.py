"""Configuration all of the players entry."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(round_feats=True, round_prior=True, map_prior="roster")
