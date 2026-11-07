"""Sweep configuration v2t_soft03_mono."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2t', 'soft': 0.3, 'monotone': True})
