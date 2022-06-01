import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_project_root() -> str:
    """Get absolute path for project root"""
    path = os.path.abspath(__file__)
    for _ in range(3): # How deep is this file
        path = os.path.dirname(path)
    return path

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
    return os.path.join(get_project_root(), "saved_models")

def get_model_class_path(model_class) -> str:
    """Gets absolute path for model storing folder for a specific model class"""
    path = os.path.join(get_models_folder_path(), model_class)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def get_images_path() -> str:
    """ Gets absolute path for images folder
    """
    return os.path.join(get_project_root(), "images")

