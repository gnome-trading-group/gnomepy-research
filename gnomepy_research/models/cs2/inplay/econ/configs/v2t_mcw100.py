"""Sweep configuration v2t_mcw100."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2t', 'eco_trees': {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.8, 'min_child_weight': 100, 'reg_lambda': 20}})
