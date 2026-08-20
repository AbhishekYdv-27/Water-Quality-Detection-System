import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px


def plot_missing_values(data: pd.DataFrame):
    missing = data.isna().sum().reset_index()
    missing.columns = ["feature", "missing"]
    missing = missing[missing["missing"] > 0]
    if missing.empty:
        return None
    fig = px.bar(missing, x="feature", y="missing", color="feature", title="Missing Values")
    return fig


def plot_correlation(data: pd.DataFrame):
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False)
    fig = px.bar(feat_imp, x="Feature", y="Importance", color="Feature", title="Feature Importance")
    return fig


def plot_distribution(data: pd.DataFrame, column: str):
    fig = px.histogram(data, x=column, color="Potability", title=f"Distribution of {column}", marginal="box")
    return fig
