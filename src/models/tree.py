import sys
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import joblib

sys.path.append("../")

from utils.model_utils import *
from utils.data_utils import get_data_for

X, y = get_data_for("tree", "train")

## Single tree

single_tree = DecisionTreeClassifier(criterion="gini",
                                     random_state=SEED,
                                     class_weight="balanced",
                                     ccp_alpha=0.01)
param_grid_tree = {
    "max_depth": list(range(0, 5)),
    "min_samples_leaf": np.linspace(start=0.01, stop=.2, num=10),
    "min_samples_split": np.linspace(start=0.001, stop=0.01, num=10),
}
# single_tree_model = grid_search_cv(single_tree, param_grid_tree, X, y, verbose=3)
# save_model(single_tree_model,
#            model_class="trees",
#            name="single_tree_cross_validated")


## Bagging

# Build bag of trees using best parameters found with single tree

# best_params = get_model("trees", "single_tree_cross_validated").best_params_
# best_params["criterion"] = "gini"
# best_params["random_state"] = SEED
# best_params["class_weight"] = "balanced"
# best_params["ccp_alpha"] = 0.01


bag_of_trees = BaggingClassifier(DecisionTreeClassifier(**best_params),
                                n_estimators=20,
                                n_jobs=2)
param_grid_bag = {
    "max_samples": np.linspace(.1, 1.0, num=10),
    "max_features": np.linspace(.1, 1.0, num=10),
    "max_depth": list(range(0, 5)),
    "min_samples_leaf": np.linspace(start=0.01, stop=.2, num=10),
    "min_samples_split": np.linspace(start=0.001, stop=0.01, num=10),
}

bag_of_trees_model = grid_search_cv(bag_of_trees, param_grid_bag, X, y,
                                    verbose=3, n_jobs=4)
save_model(bag_of_trees_model, model_class="trees",
           name="bag_of_trees_cross_validated")

## Random Forest

# Also built with the same set of parameters for tree

param_grid_forest = {
    "max_samples": np.linspace(.1, 1.0, 10),
    "max_depth": list(range(0, 5)),
    "min_samples_leaf": np.linspace(start=0.01, stop=.2, num=10),
    "min_samples_split": np.linspace(start=0.001, stop=0.01, num=10),
}

random_forest = RandomForestClassifier(**best_params)
random_forest_model = grid_search_cv(random_forest,
                                     param_grid_forest,
                                     X, y,
                                     n_jobs=4,
                                     verbose=3)
save_model(random_forest_model,
           model_class="trees",
           name="random_forest_cross_validated")
