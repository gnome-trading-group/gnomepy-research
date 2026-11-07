"""Configuration prior_only of the players entry."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(round_feats=False, round_prior=False, map_prior="roster")
