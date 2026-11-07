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
VARIANTS.update({
    "look-tier-half": Config(extra=T, half=True),
})
MONO = {"prior_logit": 1, "need_a": -1, "need_b": 1}
MONOP = {"prior_logit": 1}
VARIANTS.update({
    "look-tier-mono": Config(extra=T, q_kw=dict(monotone_constraints=MONO)),
    "look-tier-monop": Config(extra=T, q_kw=dict(monotone_constraints=MONOP)),
    "look-tier-mono-half": Config(extra=T, half=True, q_kw=dict(monotone_constraints=MONO)),
})
VARIANTS.update({
    "monop-avg5": Config(extra=T, q_kw=dict(monotone_constraints=MONOP, n_seeds=5)),
    "monop-q1": Config(extra=T, q_kw=dict(monotone_constraints=MONOP), q1_kw=dict(monotone_constraints=MONOP)),
    "monop-all": Config(extra=T, q_kw=dict(monotone_constraints=MONOP), q1_kw=dict(monotone_constraints=MONOP),
                        map_kw=dict(min_child_weight=200, monotone_constraints=MONOP),
                        round_kw=dict(max_depth=6, learning_rate=0.05, monotone_constraints={"ct_prior": 1})),
    "monop-w1": Config(extra=T, w_dp=1.0, q_kw=dict(monotone_constraints=MONOP)),
    "monop-slow": Config(extra=T, q_kw=dict(monotone_constraints=MONOP, learning_rate=0.02, n_estimators=1000)),
    "monop-d4": Config(extra=T, q_kw=dict(monotone_constraints=MONOP, max_depth=4, n_estimators=600)),
})


def all_mono(q_kw, q1_kw=None, **kw):
    return Config(extra=T, q_kw=dict(monotone_constraints=MONOP, **q_kw), q1_kw=dict(monotone_constraints=MONOP, **(q1_kw or {})),
                  map_kw=dict(min_child_weight=200, monotone_constraints=MONOP),
                  round_kw=dict(max_depth=6, learning_rate=0.05, monotone_constraints={"ct_prior": 1}), **kw)


SLOW = dict(learning_rate=0.02, n_estimators=1000)
IC = [["prior_logit", "has_prior", "tier_code", "need_a", "need_b", "ot"],
      ["ct_a", "map_code", "round_in_half", "pistol", "streak", "need_a", "need_b", "ot", "tier_code"]]
VARIANTS.update({
    "slow-all": all_mono(SLOW),
    "slower-all": all_mono(dict(learning_rate=0.01, n_estimators=2000)),
    "slow-all-mcw300": all_mono(dict(SLOW, min_child_weight=300)),
    "slow-all-ic": all_mono(dict(SLOW, interaction_constraints=IC)),
    "slow-all-q1slow": all_mono(SLOW, SLOW),
})
