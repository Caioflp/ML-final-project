import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import *

SEED = 42

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

X_train, X_test, y_train, y_test = get_data()

pipe = Pipeline([("scaler", StandardScaler()),
                 ("logit", LogisticRegression(penalty="elasticnet",
                                              tol=1e-4,
                                              random_state=SEED,
                                              solver="saga",
                                              max_iter=1000,
                                              warm_start=True))])
# Param names are adapted to work with pipelines
param_grid = {
    "logit__C":[np.power(10.0, i) for i in range(-2, 3)],
    "logit__l1_ratio": np.linspace(start=0, stop=1, num=5),
}
cv = StratifiedKFold(n_splits=5)
clf = GridSearchCV(estimator=pipe,
                   param_grid=param_grid,
                   n_jobs=4,
                   cv=cv,
                   verbose=3)
clf.fit(X_train, y_train)
