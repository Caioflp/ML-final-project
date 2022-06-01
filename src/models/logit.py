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
                                              tol=1e-4,
                                              random_state=SEED,
                                              solver="saga",
                                              max_iter=1000,
                                              warm_start=True))])
# Parameters' names are adapted to work with pipelines
param_grid = {
    "logit__C": [np.power(10.0, i) for i in range(-1, 2)],
    "logit__l1_ratio": np.linspace(start=0, stop=1, num=3),
}
model = grid_search_cv(pipe, param_grid, X, y)
save_model(model, model_class="logit", name="logit_cross_validated.joblib")
