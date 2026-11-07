"""Sweep configuration v2tc."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2tc'})
