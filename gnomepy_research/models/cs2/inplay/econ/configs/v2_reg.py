"""Sweep configuration v2_reg."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_feats': 'v2', 'eco_trees': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.8, 'min_child_weight': 1000, 'reg_lambda': 50}})
