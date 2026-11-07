"""Sweep configuration v2d6."""
from gnomepy_research.models.cs2.inplay.econ.model import make

fit, predict = make(**{'eco_trees': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.8, 'min_child_weight': 300, 'reg_lambda': 20}})
