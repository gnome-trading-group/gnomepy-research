"""Sweep configuration v2rx."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'round_extra': True})
