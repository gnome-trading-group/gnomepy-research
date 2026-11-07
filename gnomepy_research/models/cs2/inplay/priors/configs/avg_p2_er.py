from gnomepy_research.models.cs2.inplay.priors.core import Config, entry

fit, predict = entry(Config(market_mode="avg", nm_set="p2", edge_round=True, edge_from_prior=True))
