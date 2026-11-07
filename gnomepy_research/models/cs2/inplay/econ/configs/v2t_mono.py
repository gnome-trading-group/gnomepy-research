"""Sweep configuration v2t_mono."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2t', 'monotone': True})
