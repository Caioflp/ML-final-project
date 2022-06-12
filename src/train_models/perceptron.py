import sys
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append("../")
from utils.data_utils import get_data_for
from utils.model_utils import *

X, y = get_data_for("perceptron", "train")

pipe = Pipeline([("scaler", StandardScaler()),
                 ("perceptron", Perceptron(penalty="elasticnet",
                                           tol=1e-4,
                                           fit_intercept=False,
                                           random_state=SEED,
                                           max_iter=1000))])
# Parameters' names are adapted to work with pipelines
param_grid = {
    "perceptron__eta0": np.linspace(start=0.1, stop=1, num=10),
    "perceptron__tol": [np.power(10.0, i) for i in range(-5, -2)],
    "perceptron__l1_ratio": np.linspace(start=0, stop=1, num=10),
    "perceptron__class_weight": [None, "balanced"]
}
model = grid_search_cv(pipe, param_grid, X, y, scoring="roc_auc", verbose=4)
save_model(model, model_class="perceptron", name="perceptron_roc_auc_cv")

## Last validation for roc_auc
# Best set of parameters for current Pipeline(steps=[('scaler', StandardScaler()),
#                 ('perceptron',
#                  Perceptron(fit_intercept=False, penalty='elasticnet',
#                             random_state=42, tol=0.0001))]):
#  {'perceptron__class_weight': 'balanced', 'perceptron__eta0': 0.4, 'perceptron__l1_ratio': 0.4444444444444444, 'perceptron__tol': 1e-05}
# Corresponding score:  0.8453735907955652

