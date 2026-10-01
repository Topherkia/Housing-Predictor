import optuna
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


FEATURES = [
    "Rooms",
    "Distance",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea"
]

TARGET = "Price"


def load_data(data_path):
    df = pd.read_csv(data_path)

    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def objective(trial, X_train, y_train, X_valid, y_valid):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 200, 1000
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 12
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10.0, log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 10.0, log=True
        )
    }

    model = XGBRegressor(
        **params,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    rmse = mean_squared_error(
        y_valid,
        predictions
    ) ** 0.5

    return rmse


def tune_xgboost(data_path, n_trials=50):

    X, y = load_data(data_path)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_valid,
            y_valid
        ),
        n_trials=n_trials
    )

    print("Best parameters:")
    print(study.best_params)

    print("Best RMSE:")
    print(study.best_value)

    return study.best_params