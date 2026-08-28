import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.helper import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN
from utils.preprocessing import prepare_data


def train_and_evaluate():
    data = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(data)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    models = {
        "Logistic Regression": pipeline,
        "Decision Tree": Pipeline([("classifier", DecisionTreeClassifier(random_state=42))]),
        "Random Forest": Pipeline([("classifier", RandomForestClassifier(random_state=42, n_estimators=200))]),
        "Gradient Boosting": Pipeline([("classifier", GradientBoostingClassifier(random_state=42))]),
    }

    param_grids = {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
        "Decision Tree": {"classifier__max_depth": [3, 5, 7], "classifier__min_samples_leaf": [1, 2, 4]},
        "Random Forest": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 5, 10]},
        "Gradient Boosting": {"classifier__n_estimators": [50, 100], "classifier__learning_rate": [0.05, 0.1]},
    }

    results = {}
    best_model = None
    best_name = None
    best_score = -1.0

    for name, base_model in models.items():
        grid = GridSearchCV(base_model, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {
            "model": grid.best_estimator_,
            "accuracy": acc,
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, preds),
        }
        if acc > best_score:
            best_score = acc
            best_name = name
            best_model = grid.best_estimator_

    if best_model is None:
        raise RuntimeError("No model trained")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "results": results,
        "best_model_name": best_name,
        "best_accuracy": best_score,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Best model: {best_name} with accuracy {best_score:.4f}")
    return bundle


if __name__ == "__main__":
    train_and_evaluate()
