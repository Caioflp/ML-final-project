import sys
import numpy as np
from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append("../")
from utils.data_utils import get_data_for
from utils.model_utils import *

X, y = get_data_for("svm", "train")

pipe = Pipeline([("scaler", StandardScaler()),
                 ("svm", SVC(cache_size=1000,
                             class_weight="balanced"))])
# Parameters' names are adapted to work with pipelines
# Use coef_0?
param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [np.power(10.0, i) for i in range(-1, 2)],
    "svm__gamma": ["scale", "auto"]
}
model = grid_search_cv(pipe, param_grid, X, y, scoring="balanced_accuracy",
                       verbose=4, n_jobs=6, K=2)
save_model(model, model_class="svm", name="svm_bal_acc_cv")

