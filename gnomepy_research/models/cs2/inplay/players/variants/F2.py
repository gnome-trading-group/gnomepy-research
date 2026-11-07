"""Configuration F2: F plus roster stability and team map rate in the roster prior, market stacked with e12 and the
map-specific roster terms, team shrinkage 10 rounds."""
from gnomepy_research.models.cs2.inplay.players.entry import CFG, fit, predict  # noqa: F401

CFG.update(pm="full2", stack="memap", a_team=10, map_nohas=True, map_mono=True, round_feats=False, round_prior=False,
           map_kw=dict(max_depth=6, min_child_weight=300, learning_rate=0.05))
