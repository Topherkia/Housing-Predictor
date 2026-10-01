from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
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


def train():

    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "melb_data_processed.csv"
    )

    df = pd.read_csv(
        data_path
    )

    df = df[
        FEATURES + [TARGET]
    ].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    print(
        "Training rows:",
        len(X_train),
    )

    print(
        "Testing rows:",
        len(X_test),
    )

    print(
        "\nStarting Optuna tuning..."
    )

    best_params, cv_rmse, _ = (
        tune_xgboost(
            X_train,
            y_train,
            n_trials=30,
            save_path=(
                "models/"
                "xgboost_best_params.json"
            ),
        )
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        best_params.items()
    ):

        print(
            f"{key}: {value}"
        )

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
        "\n================================"
    )

    print(
        "TUNED XGBOOST RESULTS"
    )

    print(
        "================================"
    )

    print(
        f"CV RMSE: ${cv_rmse:,.2f}"
    )

    print(
        f"Test MAE: ${mae:,.2f}"
    )

    print(
        f"Test RMSE: ${rmse:,.2f}"
    )

    print(
        f"Test R²: {r2:.4f}"
    )

    return model


if __name__ == "__main__":
    train()