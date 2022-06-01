import os
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from utils.path_utils import *

SEED = 42

def save_model(model, model_class, name) -> None:
    name += ".joblib"
    path = get_model_class_path(model_class)
    path = os.path.join(path, name)
    dump(model, path)

def get_model(model_class, name) -> None:
    name += ".joblib"
    path = get_model_class_path(model_class)
    path = os.path.join(path, name)
    return load(path)

def grid_search_cv(model, parameters, *data, K=5, verbose=0):
    cv = StratifiedKFold(n_splits=K)
    best_model = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=8,
                              cv=cv, verbose=verbose)
    best_model.fit(*data)
    print("Best set of parameters for single tree:\n",
          best_model.best_params_)
    print("Corresponding score: ",
          best_model.best_score_)
    return best_model
