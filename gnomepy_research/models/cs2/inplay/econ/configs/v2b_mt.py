"""Sweep configuration v2b_mt."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2b', 'map_tier': True})
