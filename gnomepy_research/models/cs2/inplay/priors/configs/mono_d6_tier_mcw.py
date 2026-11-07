from gnomepy_research.models.cs2.inplay.priors.core import Config, entry

fit, predict = entry(Config(market_mode="avg", nm_set="p2", mono=True, map_extra=("tier_c",), map_kw=(("max_depth", 6), ("min_child_weight", 200))))
