import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load

SEED = 42

def get_project_root() -> str:
    "Get absolute path for project root"
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path() -> str:
    "Get absolute path for data folder"
    return os.path.join(get_project_root(), "data")

def get_raw_data_path() -> str:
    "Get absolute path for raw data"
    return os.path.join(get_data_path(), "us_census.csv")

def get_train_data_path() -> str:
    "Get absolute path for training data"
    return os.path.join(get_data_path(), "train.csv")

def get_test_data_path() -> str:
    "Get absolute path for test data"
    return os.path.join(get_data_path(), "test.csv")

def get_models_folder_path() -> str:
    "Gets absolute path for model storing folder"
    return os.path.join(get_project_root(), "models")

def get_model_class_path(model_class) -> str:
    "Gets absolute path for model storing folder for a specific model class"
    path = os.path.join(get_models_folder_path(), model_class)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_train_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(get_train_data_path()).astype(np.float64)
    y = df["target"]
    X = df.drop("target", axis=1)
    return X, y

def get_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(get_test_data_path()).astype(np.float64)
    y = df["target"]
    X = df.drop("target", axis=1)
    return X, y

def save_model(model, model_class, name) -> None:
    path = get_model_class_path(model_class)
    path = os.path.join(path, name)
    dump(model, path)
