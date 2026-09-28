import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class LinearRegressionModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.features = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']
        self.target = 'Price'

    def train_and_evaluate(self):
        df = pd.read_csv(self.data_path)
        df_clean = df[self.features + [self.target]].dropna()

        X = df_clean[self.features]
        y = df_clean[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print("--- Linear Regression Performance ---")
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")

        return self.model, {"MAE": mae, "RMSE": rmse}