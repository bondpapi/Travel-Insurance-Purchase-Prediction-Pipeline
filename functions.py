from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def xgb_feature_selection(X, y, n_estimators=1000, threshold="median", plot=True):
    """
    Fits XGBClassifier, selects important features, and optionally plots them.

    Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        n_estimators (int): Number of boosting rounds
        threshold (str or float): Feature importance threshold ("mean", "median", or float)
        plot (bool): Whether to show a barplot of feature importance

    Returns:
        selected_features (list): List of selected feature names
        feature_importance_df (DataFrame): Sorted importance scores
    """
    model = XGBClassifier(n_estimators=n_estimators,
                          use_label_encoder=False,
                          eval_metric='logloss',
                          random_state=42)

    model.fit(X, y)

    # Get importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    # Select features above threshold
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    selected_features = list(X.columns[selector.get_support()])

    # Display results
    display(pd.DataFrame({'Selected Features': selected_features}))

    if plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature',
                    data=feature_importance_df, palette='viridis')
        plt.title(f"XGBoost Feature Importance (n_estimators={n_estimators})")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    return selected_features, feature_importance_df
