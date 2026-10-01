import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import optuna
import joblib


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "melb_data_processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

PARAMS_PATH = MODELS_DIR / "xgboost_best_params.json"
MODEL_PATH = MODELS_DIR / "xgboost_tuned.joblib"


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

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

RANDOM_STATE = 42


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 50)
print("LOADING DATA")
print("=" * 50)

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=[TARGET]).copy()

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )


# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# ---------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------
# IMPORTANT:
# Calculate medians from training data only.
# Then apply those medians to both train and test.

train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)


# ---------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------

print()
print("Starting Optuna tuning...")

cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000,
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        ),
    }

    model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    return -float(np.mean(scores))


study = optuna.create_study(
    direction="minimize",
    study_name="housing_xgboost",
)

study.optimize(
    objective,
    n_trials=30,
)


# ---------------------------------------------------------
# Get best parameters
# ---------------------------------------------------------

best_params = study.best_params

print()
print("=" * 50)
print("BEST PARAMETERS")
print("=" * 50)

for key, value in best_params.items():
    print(f"{key}: {value}")


# ---------------------------------------------------------
# Save best parameters
# ---------------------------------------------------------

with open(PARAMS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        best_params,
        f,
        indent=4,
    )

print()
print(f"Saved parameters to:")
print(PARAMS_PATH)


# ---------------------------------------------------------
# Train FINAL XGBoost model
# ---------------------------------------------------------

print()
print("=" * 50)
print("TRAINING FINAL XGBOOST MODEL")
print("=" * 50)

final_model = XGBRegressor(
    **best_params,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Evaluate final model
# ---------------------------------------------------------

y_pred = final_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred,
)

rmse = mean_squared_error(
    y_test,
    y_pred,
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred,
)


print()
print("=" * 50)
print("TUNED XGBOOST RESULTS")
print("=" * 50)

print(f"CV RMSE:   ${study.best_value:,.2f}")
print(f"Test MAE:  ${mae:,.2f}")
print(f"Test RMSE: ${rmse:,.2f}")
print(f"Test R²:   {r2:.4f}")


# ---------------------------------------------------------
# SAVE TRAINED MODEL
# ---------------------------------------------------------

joblib.dump(
    final_model,
    MODEL_PATH,
)


print()
print("=" * 50)
print("MODEL SAVED")
print("=" * 50)

print(f"Model:")
print(MODEL_PATH)

print()
print("Files in models directory:")

for file in MODELS_DIR.iterdir():
    print(
        f" - {file.name} "
        f"({file.stat().st_size:,} bytes)"
    )