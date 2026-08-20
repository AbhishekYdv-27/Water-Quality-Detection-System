import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET_COLUMN = "Potability"


def prepare_data(data: pd.DataFrame):
    data = data.copy()
    data = data.dropna(subset=[TARGET_COLUMN])
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    data[FEATURE_COLUMNS] = imputer.fit_transform(data[FEATURE_COLUMNS])

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    if y.nunique() > 1 and y.value_counts().min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )
    return X_train, X_test, y_train, y_test
