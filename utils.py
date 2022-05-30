import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data() -> pd.DataFrame:
    df = pd.read_csv("./data/cleaned_us_census.csv").astype(np.float64)
    y = df["target"]
    X = df.drop("target", axis=1)
    return train_test_split(X, y, test_size=0.25, random_state=42)

