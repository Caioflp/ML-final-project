import os
import sys
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import (roc_auc_score,
                             ConfusionMatrixDisplay,
                             confusion_matrix,
                             RocCurveDisplay)
import matplotlib.pyplot as plt
import graphviz

sys.path.append("../")
from utils.data_utils import get_data_for
from utils.model_utils import *
from utils.path_utils import get_images_path

X, y = get_data_for("trees", "test")

model = get_model("trees", "random_forest_roc_auc_cv")

print("Best parameters:")
print(model.best_params_)

pred = model.predict(X)
pred_prob = model.predict_proba(X)

print(f"Score on training: {model.best_score_:.3f}")
print(f"AUC score: {roc_auc_score(y, pred_prob[:,1]):.3f}")

ConfusionMatrixDisplay.from_estimator(model, X, y)
plt.savefig(os.path.join(get_images_path(), "matrix_random_forest.pdf"))
plt.clf()

RocCurveDisplay.from_estimator(model, X, y)
plt.legend("",frameon=False)
plt.savefig(os.path.join(get_images_path(), "roc_curve_random_forest.pdf"))
plt.clf()

mat = confusion_matrix(y, pred)
print(f"recall: {mat[1, 1] / (mat[1, 1] + mat[1, 0])}")
print(f"precisão: {mat[1, 1] / (mat[1, 1] + mat[0, 1])}")
print(f"especificidade: {mat[0, 0] / (mat[0, 1] + mat[0, 0])}")

clf = model.best_estimator_
coef_list = list(zip(clf.feature_names_in_,
                     clf.feature_importances_))
coef_list.sort(key=lambda t: t[1])
front = coef_list[:5]
biggest = front
names = [t[0] for t in biggest]
values = [t[1] for t in biggest]
sns.barplot(x=names, y=values)
plt.xticks(ticks=range(len(names)), labels=names, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(get_images_path(), "random_forest_feature_importances.pdf"))
plt.clf()

