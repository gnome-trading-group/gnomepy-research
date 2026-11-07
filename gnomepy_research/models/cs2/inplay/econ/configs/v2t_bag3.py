"""Sweep configuration v2t_bag3."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2t', 'bag': 3})
