from gnomepy_research.models.cs2.inplay.priors.core import Config, entry

fit, predict = entry(Config(market_mode="avg", nm_set="x1", nm_kind="xgb"))
