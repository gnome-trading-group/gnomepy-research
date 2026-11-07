"""Sweep configuration v1."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v1'})
