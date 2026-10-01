from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.clustering_stage import ClusterStage
from src.model_decision_tree import DecisionTreeModel
from src.model_gradient_boosting import GradientBoostingModel
from src.model_linear_regression import LinearRegressionModel
from src.model_random_forest import RandomForestModel
from src.model_xgboost import XGBoostModel

PROJECT_ROOT = Path(__file__).resolve().parent


class HousingPipelineApp:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data_path_str = str(data_path)

    def ask_for_regressor(self):
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

        if choice == '1':
            return LinearRegressionModel(data_path=self.data_path_str).model
        elif choice == '2':
            return DecisionTreeModel(data_path=self.data_path_str).model
        elif choice == '3':
            return RandomForestModel(data_path=self.data_path_str).model
        elif choice == '4':
            return GradientBoostingModel(data_path=self.data_path_str).model
        elif choice == '5':
            return XGBoostModel(data_path=self.data_path_str).model
        else:
            return RandomForestModel(data_path=self.data_path_str).model

    def build_and_train_pipeline(self):
        chosen_regressor = self.ask_for_regressor()

        print(f"\nLoading data from '{self.data_path}'...")
        raw_df = pd.read_csv(self.data_path)

        # 1. Clean & Preprocess initial dataframe
        cluster_stage = ClusterStage(data_path=self.data_path_str, n_clusters=3, random_state=42)
        df_cleaned = cluster_stage.preprocess_data(raw_df)

        drop_cols = ['Address', 'SellerG', 'Date', 'Postcode', 'CouncilArea']
        df_filtered = df_cleaned.drop(columns=[col for col in drop_cols if col in df_cleaned.columns])

        # 2. Separate Target and Features BEFORE Clustering & Model Fitting
        X = df_filtered.drop(columns=['Price'])
        y = df_filtered['Price']

        # 3. Train / Test Split BEFORE KMeans
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Fit KMeans ONLY on Training Data & Transform Test Set
        print("\nRunning Leak-Free K-Means Clustering...")
        X_train_clustered = cluster_stage.fit_transform_train(X_train)
        X_test_clustered = cluster_stage.transform_test(X_test)

        # 5. Build Column Transformer including the new 'Cluster' feature
        categorical_cols = X_train_clustered.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X_train_clustered.select_dtypes(include=['int64', 'float64', 'Int64']).columns.tolist()

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

        # 6. Fit & Evaluate Model
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', chosen_regressor)
        ])

        print("\nTraining selected regression model...")
        full_pipeline.fit(X_train_clustered, y_train)

        y_pred = full_pipeline.predict(X_test_clustered)

        print("\n--- Model Performance ---")
        print(f"MAE:  ${mean_absolute_error(y_test, y_pred):,.2f}")
        print(f"RMSE: ${root_mean_squared_error(y_test, y_pred):,.2f}")
        print(f"R2:   {r2_score(y_test, y_pred):.4f}")

        return full_pipeline, cluster_stage.kmeans, cluster_stage.scaler


if __name__ == "__main__":
    default_path = "data/raw/melb_data.csv"
    raw_input_path = input(f"Enter the CSV relative file path (default: {default_path}): ").strip()
    if not raw_input_path:
        raw_input_path = default_path

    input_path = Path(raw_input_path)
    clean_rel_path = input_path.relative_to(input_path.anchor) if input_path.is_absolute() else input_path
    dataset_path = (PROJECT_ROOT / clean_rel_path).relative_to(PROJECT_ROOT)

    app = HousingPipelineApp(data_path=dataset_path)
    trained_pipeline, kmeans, scaler = app.build_and_train_pipeline()