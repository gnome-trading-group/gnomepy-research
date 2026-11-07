"""Configuration F3: F2 plus adjusted plus-minus player ratings in the roster prior."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(pm="full3", stack="memap", a_team=10, map_nohas=True, map_mono=True, round_feats=False, round_prior=False,
           map_kw=dict(max_depth=6, min_child_weight=300, learning_rate=0.05))
