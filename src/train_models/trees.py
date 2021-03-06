import sys
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import joblib

sys.path.append("../")

from utils.model_utils import *
from utils.data_utils import get_data_for

X, y = get_data_for("tree", "train")

## Single tree

single_tree = DecisionTreeClassifier(criterion="gini", random_state=SEED)

param_grid_tree = {
    "class_weight": ["balanced", None],
    "ccp_alpha": [np.power(10.0, i) for i in range(-3, 2)],
    "max_depth": list(range(3, 10)),
    "min_samples_leaf": np.linspace(start=0.01, stop=.2, num=5),
    "min_samples_split": np.linspace(start=0.001, stop=0.01, num=5),
}
if True:
    single_tree_model = grid_search_cv(single_tree, param_grid_tree, X, y,
                                       scoring="roc_auc", verbose=1)
    save_model(single_tree_model,
               model_class="trees",
               name="single_tree_roc_auc_cv")

best_tree = get_model(model_class="trees", name="single_tree_roc_auc_cv")
best_params = best_tree.best_params_

## Bagging

bag_of_trees = BaggingClassifier(DecisionTreeClassifier(**best_params),
                                n_jobs=2, random_state=SEED)

param_grid_bag = {
    "n_estimators": [100, 200, 300, 400],
    "max_samples": np.linspace(.2, 1.0, num=5),
    "max_features": np.linspace(.2, 1.0, num=5),
}

if True:
    bag_of_trees_model = grid_search_cv(bag_of_trees, param_grid_bag, X, y,
                                        scoring="roc_auc", verbose=2, n_jobs=4)
    save_model(bag_of_trees_model, model_class="trees",
               name="bag_of_trees_roc_auc_cv")

## Random Forest

random_forest = RandomForestClassifier(**best_params, n_jobs=2)

param_grid_forest = {
    "n_estimators": [100, 200, 300, 400],
    "max_samples": np.linspace(.1, 1.0, num=5),
    "max_features": np.linspace(.1, 1.0, num=5),
}

if True:
    random_forest_model = grid_search_cv(random_forest,
                                         param_grid_forest,
                                         X, y,
                                         scoring="roc_auc",
                                         n_jobs=4,
                                         verbose=2)
    save_model(random_forest_model,
               model_class="trees",
               name="random_forest_roc_auc_cv")
