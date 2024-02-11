"""<>"""
import numpy as np

from pandas import DataFrame
from optuna import Trial
from sklearn import ensemble, model_selection
from keys.random_forest_regressor_keys import RandomForestRegressor

def objective(trial : Trial, x : DataFrame, y : np.ndarray) -> float:
    """<>"""
    n_estimators = trial.suggest_int(RandomForestRegressor.N_ESTIMATORS, 2, 20)
    max_depth = int(trial.suggest_float(RandomForestRegressor.MAX_DEPTH, 1, 32, log=True))
    clf = ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    return model_selection.cross_val_score(clf, x, y, n_jobs=-1, cv=3).mean()
