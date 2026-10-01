from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.hyperparameter_tuning import (
    tune_xgboost,
)


FEATURES = [
    "Rooms",
    "Distance",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
]

TARGET = "Price"


def train_tuned_xgboost(
    data_path,
    n_trials=30,
):

    df = pd.read_csv(
        data_path
    )

    df = df[
        FEATURES + [TARGET]
    ].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Test data is separated BEFORE tuning.
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ---------------------------------------------------------
    # Tune using training data ONLY
    # ---------------------------------------------------------

    best_params, cv_rmse, study = tune_xgboost(
        X_train,
        y_train,
        n_trials=n_trials,
        save_path="models/xgboost_best_params.json",
    )

    print(
        "\nBest parameters:"
    )

    print(best_params)

    print(
        f"\nCross-validation RMSE: "
        f"${cv_rmse:,.2f}"
    )

    # ---------------------------------------------------------
    # Train final tuned model
    # ---------------------------------------------------------

    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = root_mean_squared_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(
        "\n--- Tuned XGBoost Performance ---"
    )

    print(
        f"MAE:  ${mae:,.2f}"
    )

    print(
        f"RMSE: ${rmse:,.2f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    return model, {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "best_params": best_params,
    }


if __name__ == "__main__":

    data_path = (
        "data/processed/"
        "melb_data_processed.csv"
    )

    train_tuned_xgboost(
        data_path,
        n_trials=30,
    )