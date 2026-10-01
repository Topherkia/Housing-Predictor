import json
from pathlib import Path

import numpy as np
import optuna

from sklearn.model_selection import (
    KFold,
    cross_val_score,
)

from xgboost import XGBRegressor


def tune_xgboost(
    X,
    y,
    n_trials=30,
    random_state=42,
    save_path=None,
):

    """
    Tune XGBoost using cross-validation.

    IMPORTANT:
    Only training data should be passed to this function.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state,
    )

    def objective(trial):

        params = {

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    200,
                    1000,
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    10,
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3,
                    log=True,
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.6,
                    1.0,
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.6,
                    1.0,
                ),

            "min_child_weight":
                trial.suggest_int(
                    "min_child_weight",
                    1,
                    10,
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-8,
                    10.0,
                    log=True,
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-8,
                    10.0,
                    log=True,
                ),
        }

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        return -float(
            np.mean(scores)
        )

    study = optuna.create_study(
        direction="minimize",
        study_name="housing_xgboost",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    best_params = (
        study.best_params.copy()
    )

    if save_path:

        output = Path(
            save_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                best_params,
                indent=2,
            ),
            encoding="utf-8",
        )

    return (
        best_params,
        study.best_value,
        study,
    )