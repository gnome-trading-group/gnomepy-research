"""Competitor e17: one player-skill filter over HLTV and FACEIT maps, calibrated weekly.

The existing per-player Gaussian state-space filter (models/cs2/hltv/player_skill.py) is run over the HLTV maps and the
cleaned FACEIT maps as one chronological stream. FACEIT is a second measurement of the same players: its maps carry
their own observation weight, and players first seen on FACEIT start from their own prior. Unlinked FACEIT players get
ids derived from their FACEIT id, so FACEIT-only opponents connect the rating graph. An HLTV-only filter runs alongside
it; a logistic calibration refit each week on the trailing year of bo3 series turns the two lineup gaps into a series
price, which is also how the model falls back to HLTV alone where FACEIT has nothing to add.
"""
