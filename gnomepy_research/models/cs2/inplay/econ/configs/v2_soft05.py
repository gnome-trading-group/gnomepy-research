"""Sweep configuration v2_soft05."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2', 'soft': 0.5})
