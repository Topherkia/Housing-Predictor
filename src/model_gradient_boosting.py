import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class GradientBoostingModel:
    def __init__(self, data_path: str, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 5, random_state: int = 42):
        self.data_path = data_path
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            random_state=random_state
        )
        self.features = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']
        self.target = 'Price'

    def train_and_evaluate(self):
        df = pd.read_csv(self.data_path)
        df_clean = df[self.features + [self.target]].dropna()

        X = df_clean[self.features]
        y = df_clean[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print("--- Gradient Boosting Regressor Performance ---")
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")

        return self.model, {"MAE": mae, "RMSE": rmse}