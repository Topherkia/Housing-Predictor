import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


class XGBoostModel:

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

    def __init__(
        self,
        data_path: str,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42,
        **kwargs,
    ):

        self.data_path = data_path

        self.features = self.FEATURES
        self.target = self.TARGET

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            **kwargs,
        )

    def train_and_evaluate(self):

        df = pd.read_csv(
            self.data_path
        )

        df_clean = (
            df[
                self.features
                + [self.target]
            ]
            .dropna()
        )

        X = df_clean[
            self.features
        ]

        y = df_clean[
            self.target
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        self.model.fit(
            X_train,
            y_train,
        )

        y_pred = self.model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        rmse = root_mean_squared_error(
            y_test,
            y_pred,
        )

        r2 = r2_score(
            y_test,
            y_pred,
        )

        print(
            "--- XGBoost Regressor Performance ---"
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

        return self.model, {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }