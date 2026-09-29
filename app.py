# app.py

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Import classes from src
from src.clustering_stage import ClusterStage
from src.model_decision_tree import DecisionTreeModel
from src.model_gradient_boosting import GradientBoostingModel
from src.model_linear_regression import LinearRegressionModel
from src.model_random_forest import RandomForestModel
from src.model_xgboost import XGBoostModel

# Project root directory reference
PROJECT_ROOT = Path(__file__).resolve().parent


class HousingPipelineApp:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path_str = str(data_path)

    def ask_for_regressor(self):
        """Prompts the user to choose a regressor class from the src module."""
        print("\n==============================================")
        print("Select a Regression Model to Train:")
        print("==============================================")
        print("[1] Linear Regression")
        print("[2] Decision Tree Regressor")
        print("[3] Random Forest Regressor")
        print("[4] Gradient Boosting Regressor")
        print("[5] XGBoost Regressor")
        print("==============================================")

        choice = input("Enter choice (1-5): ").strip()

        # Instantiate model classes from src using self.data_path_str
        if choice == '1':
            print("\nSelected: Linear Regression (src.model_linear_regression)")
            return LinearRegressionModel(data_path=self.data_path_str).model
        elif choice == '2':
            print("\nSelected: Decision Tree Regressor (src.model_decision_tree)")
            return DecisionTreeModel(data_path=self.data_path_str).model
        elif choice == '3':
            print("\nSelected: Random Forest Regressor (src.model_random_forest)")
            return RandomForestModel(data_path=self.data_path_str).model
        elif choice == '4':
            print("\nSelected: Gradient Boosting Regressor (src.model_gradient_boosting)")
            return GradientBoostingModel(data_path=self.data_path_str).model
        elif choice == '5':
            print("\nSelected: XGBoost Regressor (src.model_xgboost)")
            return XGBoostModel(data_path=self.data_path_str).model
        else:
            print("\nInvalid choice. Defaulting to Random Forest Regressor from src.")
            return RandomForestModel(data_path=self.data_path_str).model

    def build_and_train_pipeline(self):
        # 1. Ask for regressor selection from src
        chosen_regressor = self.ask_for_regressor()

        # 2. Raw data loading
        print(f"\nLoading data from '{self.data_path}'...")
        raw_df = pd.read_csv(self.data_path)
        print(f"Raw data shape: {raw_df.shape}")

        # 3. Data cleaning & drop non-predictive attributes
        drop_cols = ['Address', 'SellerG', 'Date', 'Postcode', 'CouncilArea']
        df_filtered = raw_df.drop(columns=[col for col in drop_cols if col in raw_df.columns])

        # 4. Clustering Stage using ClusterStage class from src
        print("\nRunning Clustering Stage...")
        cluster_stage = ClusterStage(data_path=self.data_path_str, n_clusters=3, random_state=42)
        df_clustered, kmeans_model, cluster_scaler = cluster_stage.preprocess_and_cluster(df_filtered)
        print(f"Clustered data shape: {df_clustered.shape}")

        # 5. Prepare Features & Target for Regression
        X = df_clustered.drop(columns=['Price'])
        y = df_clustered['Price']

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['int64', 'float64', 'Int64']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        # 6. Build & Train Full Pipeline using the src regressor estimator
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', chosen_regressor)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("\nTraining selected regression model...")
        full_pipeline.fit(X_train, y_train)

        # 7. Model Evaluation
        y_pred = full_pipeline.predict(X_test)

        print("\n--- Model Performance ---")
        print(f"MAE:  ${mean_absolute_error(y_test, y_pred):,.2f}")
        print(f"RMSE: ${root_mean_squared_error(y_test, y_pred):,.2f}")
        print(f"R2:   {r2_score(y_test, y_pred):.4f}")

        return full_pipeline, kmeans_model, cluster_scaler


if __name__ == "__main__":
    # Ask user for CSV relative path defaulting to data/raw/melb_data.csv
    default_path = "data/raw/melb_data.csv"
    raw_input_path = input(f"Enter the CSV relative file path (default: {default_path}): ").strip()
    if not raw_input_path:
        raw_input_path = default_path

    # Enforce relative project path context
    input_path = Path(raw_input_path)
    clean_rel_path = input_path.relative_to(input_path.anchor) if input_path.is_absolute() else input_path
    dataset_path = (PROJECT_ROOT / clean_rel_path).relative_to(PROJECT_ROOT)

    # Run application
    app = HousingPipelineApp(data_path=dataset_path)
    trained_pipeline, kmeans, scaler = app.build_and_train_pipeline()