import sys
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import joblib

sys.path.append("../")

from utils.model_utils import *
from utils.data_utils import get_data_for

X, y = get_data_for("tree", "train")

## Single tree

single_tree = DecisionTreeClassifier(random_state=SEED, class_weight="balanced")
param_grid = {
    "max_depth": list(range(5, 11)),
    "criterion": ["gini", "entropy", "log_loss"],
    "min_samples_leaf": np.linspace(start=0.01, stop=.2, num=10),
    "min_samples_split": np.linspace(start=0.001, stop=0.01, num=10),
    "ccp_alpha": [10**i for i in range(-3, 2)],

}
single_tree_model = grid_search_cv(single_tree, param_grid, X, y, verbose=2)
save_model(single_tree_model,
           model_class="trees",
           name="single_tree_cross_validated")

single_tree_estimator = get_model("trees", "single_tree_cross_validated") \
                        .best_estimator_
feature_names = X.columns
importances = pd.Series(single_tree_estimator.feature_importances_,
                        index=feature_names)
fig, ax = plt.subplots()
importances.plot.bar(ax=ax)
fig.tight_layout()
plt.show()


# Cross validation and grid search
# Feature selection
# Prunning
