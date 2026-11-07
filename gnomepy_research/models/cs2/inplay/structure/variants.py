"""Named configurations scored in development (see NOTES.md)."""
from .core import RC, Config

ECO = ["ct_money", "t_money", "ct_loss", "t_loss", "ot", "ct_streak"]

VARIANTS = {
    "dpblend-v1": Config(lookahead=False),
    "look": Config(),
    "look-w1": Config(w_dp=1.0),
    "look-eco": Config(rcols=RC + ECO),
    "look-ctq": Config(rcols=RC + ["ct_q"]),
    "look-eco-ctq": Config(rcols=RC + ECO + ["ct_q"]),
    "look-r1000": Config(round_kw=dict(max_depth=6, learning_rate=0.03, n_estimators=1000)),
}

T = ["tier_code"]
VARIANTS.update({
    "look-tier": Config(extra=T),
    "look-tierbo": Config(extra=T + ["bo", "map_no"]),
    "look-tier-map": Config(extra=T, map_extra=True),
    "look-tier-avg5": Config(extra=T, q_kw=dict(n_seeds=5)),
    "look-tier-rtier": Config(extra=T, rcols=RC + T),
    "look-tier-w1": Config(extra=T, w_dp=1.0),
    "look-tier-rs3": Config(extra=T, round_seeds=3),
})
