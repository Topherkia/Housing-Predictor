import numpy as np
import pandas as pd
import shap


def explain_prediction(
    model,
    X_row,
    feature_names=None,
    top_n=10,
):

    """
    Explain one XGBoost prediction using SHAP.
    """

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X_row
        )
    )

    shap_values = np.asarray(
        shap_values
    ).reshape(-1)

    row = np.asarray(
        X_row
    ).reshape(-1)

    if feature_names is None:

        feature_names = [
            f"feature_{i}"
            for i in range(
                len(shap_values)
            )
        ]

    result = pd.DataFrame(
        {
            "feature":
                feature_names,

            "value":
                row,

            "shap_value":
                shap_values,

            "absolute_impact":
                np.abs(
                    shap_values
                ),
        }
    )

    return result.sort_values(
        "absolute_impact",
        ascending=False,
    ).head(top_n)