from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.path_utils import *

def get_train_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(get_train_data_path())

def get_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(get_test_data_path())

def get_data_for(model: str, purpose: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Get data prepared for use by a specific model

    Parameters
    ----------
    model:
        Name of kind of model. One of: "tree", "svm", "logit" or "perceptron".
    purpose:
        "train" or "test"
    """
    data = get_train_data() if purpose == "train" else get_test_data()

    if model == "logit" or model == "perceptron" or model == "svm":
        pre =     ["wrk_cls", "marriage", "occup", "rel", "race", "", "from",]
        pre_sep = ["wrk_cls", "marriage", "occup", "rel", "race", "", "from",]
        col =     ["workclass", "marital_status", "occupation", "relationship",
                      "race", "sex", "native_country"]
        data = pd.get_dummies(data, prefix=pre, prefix_sep=pre_sep, columns=col)
    elif model == "tree":
        # Select only numeric and binary categorical variables
        data = data[["age", "education_num", "sex", "capital_gain",
                     "capital_loss", "hours_per_week", "target"]]
        data["sex"].replace({"male": 1, "female": -1}, inplace=True)
    else:
        pass

    y = data["target"]
    X = data.drop("target", axis=1)
    return X, y
