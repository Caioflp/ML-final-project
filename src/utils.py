import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load

SEED = 42

def get_project_root() -> str:
    """Get absolute path for project root"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path() -> str:
    """Get absolute path for data folder"""
    return os.path.join(get_project_root(), "data")

def get_raw_data_path() -> str:
    """Get absolute path for raw data"""
    return os.path.join(get_data_path(), "us_census.csv")

def get_clean_data_path() -> str:
    """Get absolute path for raw data"""
    return os.path.join(get_data_path(), "cleaned.csv")

def get_train_data_path() -> str:
    """Get absolute path for training data"""
    return os.path.join(get_data_path(), "train.csv")

def get_test_data_path() -> str:
    """Get absolute path for test data"""
    return os.path.join(get_data_path(), "test.csv")

def get_models_folder_path() -> str:
    """Gets absolute path for model storing folder"""
    return os.path.join(get_project_root(), "models")

def get_model_class_path(model_class) -> str:
    """Gets absolute path for model storing folder for a specific model class"""
    path = os.path.join(get_models_folder_path(), model_class)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_train_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(get_train_data_path())

def get_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(get_test_data_path())

def get_data_for(model: str, purpose: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Get data prepared for use by a specific model

    Parameters
    ----------
    model:
        Name of kind of model. One of: "tree", "svm", "logit".
    purpose:
        "train" or "test"
    """
    data = get_train_data() if purpose == "train" else get_test_data()

    if model == "logit":
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

def save_model(model, model_class, name) -> None:
    path = get_model_class_path(model_class)
    path = os.path.join(path, name)
    dump(model, path)
