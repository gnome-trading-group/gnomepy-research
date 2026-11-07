"""Configuration F: full roster prior stacked with market and e12, monotone d6/mcw300/lr.05 map model."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(pm="full", stack="mez", map_nohas=True, map_mono=True, round_feats=False, round_prior=False,
           map_kw=dict(max_depth=6, min_child_weight=300, learning_rate=0.05))
