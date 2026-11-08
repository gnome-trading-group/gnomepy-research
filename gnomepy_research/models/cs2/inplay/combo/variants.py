"""Named combinations scored in development."""
from dataclasses import replace

from ..structure.variants import VARIANTS as S
from .model import Combo

MAP_P = dict(max_depth=6, min_child_weight=300, learning_rate=0.05, monotone_constraints={"prior_logit": 1, "need_a": -1, "need_b": 1})
SMAP = replace(S["slow-all"], map_kw=MAP_P, map_drop=["has_prior"])

VARIANTS = {
    "s": Combo(roster_prior=False),
    "sp": Combo(),
    "sp-map": Combo(structure=SMAP),
    "spe": Combo(econ=True),
    "spe-map": Combo(structure=SMAP, econ=True),
}
