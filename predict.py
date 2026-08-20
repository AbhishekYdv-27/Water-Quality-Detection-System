import pandas as pd

from utils.helper import FEATURE_COLUMNS, load_model_bundle


def predict_water_quality(values: dict):
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = pd.DataFrame([values], columns=feature_columns)
    probability = model.predict_proba(df)[0, 1]
    prediction = int(model.predict(df)[0])
    return prediction, probability
