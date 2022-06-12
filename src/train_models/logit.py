import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append("../")
from utils.data_utils import get_data_for
from utils.model_utils import *

X, y = get_data_for("logit", "train")

pipe = Pipeline([("scaler", StandardScaler()),
                 ("logit", LogisticRegression(penalty="elasticnet",
                                              fit_intercept=False,
                                              random_state=SEED,
                                              solver="saga",
                                              max_iter=500))])
# Parameters' names are adapted to work with pipelines
param_grid = {
    "logit__tol": [np.power(10.0, i) for i in range(-4, -2)],
    "logit__C": [np.power(10.0, i) for i in range(-2, 1)],
    "logit__l1_ratio": np.linspace(start=0, stop=1, num=5),
    "logit__class_weight": [None, "balanced"],
}
model = grid_search_cv(pipe, param_grid, X, y, scoring="roc_auc", verbose=4)
save_model(model, model_class="logit", name="logit_roc_auc_cv")

# Last cross validation results
# Best set of parameters for current Pipeline(steps=[('scaler', StandardScaler()),
#                 ('logit',
#                  LogisticRegression(fit_intercept=False, max_iter=500,
#                                     penalty='elasticnet', random_state=42,
#                                     solver='saga'))]):
#  {'logit__C': 0.01, 'logit__class_weight': 'balanced', 'logit__l1_ratio': 0.25, 'logit__tol': 0.001}
# Corresponding score:  0.9036877022769026
