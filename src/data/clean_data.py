import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("../")

from utils.path_utils import (get_raw_data_path,
                              get_train_data_path,
                              get_test_data_path)
from utils.model_utils import SEED

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

df = pd.read_csv(get_raw_data_path(), na_values=" ?")

# For now, we discard missing values
df.dropna(axis=0, inplace=True)

# Format string type columns
format_list = ["workclass", "education","marital_status", "occupation",
              "relationship", "race", "sex", "native_country", "target"]

def format_string(string: str) -> str:
    return string.strip().replace("-", "_").lower()

df[format_list] = df[format_list].applymap(format_string)

# Specify generical "Other" in race
df["race"].replace("Other", "other_race", inplace=True)

# Replace strings in target column by ones and zeroes
rep_dict_target = {
    "<=50k": 0,
    "<=50k.": 0,
    ">50k": 1,
    ">50k.": 1
}
df["target"].replace(to_replace=rep_dict_target, inplace=True)
print(1 - df["target"].mean())

# Better marital status names
rep_dict_marriage = {
    "married_civ_spouse": "civ_spouse",
    "married_spouse_absent": "spouse_absent",
    "married_AF_spouse": "af_spouse",
}
df["marital_status"].replace(to_replace=rep_dict_marriage, inplace=True)

# Drop useless columns (we already have `education_num`)
df.drop(labels=["fnlwgt", "education"], axis=1, inplace=True)

# df.rename(columns={col: format_string(col) for col in df.columns})

# # Get dummies
# df = pd.get_dummies(df, prefix=["wrk_cls",
#                                 "marriage",
#                                 "occup",
#                                 "rel",
#                                 "race",
#                                 "",
#                                 "from",],
#                         prefix_sep=["_",
#                                     "_",
#                                     "_",
#                                     "_",
#                                     "_",
#                                     "",
#                                     "_"],
#                         columns=["workclass",
#                                  "marital_status",
#                                  "occupation",
#                                  "relationship",
#                                  "race",
#                                  "sex",
#                                  "native_country"])

df_train, df_test = train_test_split(df, test_size=.25, stratify=df["target"],
                                     random_state=SEED)
df_train.to_csv(get_train_data_path(), index=False)
df_test.to_csv(get_test_data_path(), index=False)
