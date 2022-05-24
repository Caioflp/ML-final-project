import os
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "./data/us_census.csv")
CLEANED_DATA_PATH = os.path.join(FILE_PATH, "./data/cleaned_us_census.csv")
df = pd.read_csv(DATA_PATH, na_values=" ?")

# For now, we discard missing values
df.dropna(axis=0, inplace=True)

# Format string type columns
strip_list = ["workclass", "education","marital_status", "occupation",
              "relationship", "race", "sex", "native_country", "target"]
def strip_string_if_not_nan(string: str) -> str:
    if string is not np.nan:
        return string.strip().replace("-", "_").lower()
    else:
        return string
df[strip_list] = df[strip_list].applymap(strip_string_if_not_nan)

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

# Better marital status names
rep_dict_marriage = {
    "married_civ_spouse": "civ_spouse",
    "married_spouse_absent": "spouse_absent",
    "married_AF_spouse": "af_spouse",
}
df["marital_status"].replace(to_replace=rep_dict_marriage, inplace=True)

# Drop useless columns (we already have `education_num`)
df.drop(labels=["fnlwgt", "education"], axis=1, inplace=True)

# Get dummies
df = pd.get_dummies(df, prefix=["wrk_cls",
                                "marriage",
                                "occup",
                                "rel",
                                "race",
                                "",
                                "from",],
                        prefix_sep=["_",
                                    "_",
                                    "_",
                                    "_",
                                    "_",
                                    "",
                                    "_"],
                        columns=["workclass",
                                 "marital_status",
                                 "occupation",
                                 "relationship",
                                 "race",
                                 "sex",
                                 "native_country"])
df.to_csv(CLEANED_DATA_PATH, index=False)
df.info()
