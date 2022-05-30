import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import *

X_train, X_test, y_train, y_test = get_data()
model = LogisticRegression(random_state=0, max_iter=5000)
model.fit(X_train, y_train)
print(f"Train score: {model.score(X_train, y_train)}")
print(f"Test score: {model.score(X_test, y_test)}")
