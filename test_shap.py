from pathlib import Path

import pandas as pd

from xgboost import XGBRegressor

from src.shap_explainer import (
    explain_prediction,
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


model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
)


model.fit(
    X,
    y,
)


sample = X.iloc[
    [0]
]


prediction = model.predict(
    sample
)[0]


print(
    f"Prediction: "
    f"${prediction:,.2f}"
)


explanation = explain_prediction(
    model,
    sample,
    feature_names=FEATURES,
)


print(
    "\nSHAP explanation:"
)

print(
    explanation.to_string(
        index=False
    )
)