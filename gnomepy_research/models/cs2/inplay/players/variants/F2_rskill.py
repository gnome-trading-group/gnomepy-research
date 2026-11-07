"""Configuration F2 plus alive-player strength in the round model."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(pm="full2", stack="memap", a_team=10, map_nohas=True, map_mono=True, round_feats=True, round_prior=False,
           map_kw=dict(max_depth=6, min_child_weight=300, learning_rate=0.05))
