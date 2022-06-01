from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import graphviz

from utils import *

model = DecisionTreeClassifier(max_depth=5,
                               min_samples_split=.01,
                               random_state=SEED,
                               class_weight="balanced")

X, y = get_data_for("tree", "train")
model.fit(X, y)

dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=X.columns,
                                class_names=["low_inc", "high_inc"],
                                filled=True, rounded=True,
                                special_characters=True,
                                proportion=True)
graph = graphviz.Source(dot_data)
graph.render("income")

