import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import *

pipe = Pipeline([("scaler", StandardScaler()),
                 ("logit", LogisticRegression(penalty="elasticnet",
                                              tol=1e-4,
                                              random_state=SEED,
                                              solver="saga",
                                              max_iter=1000,
                                              warm_start=True))])
# Param names are adapted to work with pipelines
param_grid = {
    "logit__C": [np.power(10.0, i) for i in range(-1, 2)],
    "logit__l1_ratio": np.linspace(start=0, stop=1, num=3),
}
cv = StratifiedKFold(n_splits=2)
model = GridSearchCV(estimator=pipe,
                     param_grid=param_grid,
                     n_jobs=4,
                     cv=cv,
                     verbose=3)

X, y = get_data_for("logit", "train")
model.fit(X, y)
save_model(model, "logistic_regression", "logistic_regression.joblib")
