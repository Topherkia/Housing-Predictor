from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import streamlit.components.v1 as components

from src.clustering_stage import ClusterStage
from src.map_generator import MapGenerator
from src.model_decision_tree import DecisionTreeModel
from src.model_gradient_boosting import GradientBoostingModel
from src.model_linear_regression import LinearRegressionModel
from src.model_random_forest import RandomForestModel
from src.model_xgboost import XGBoostModel

st.set_page_config(page_title="Melbourne Housing Price Predictor", layout="wide")

st.title("🏡 Melbourne Housing Price Predictor")
st.write("A pipeline combining **K-Means Clustering** and **Machine Learning Regressors**.")

PROJECT_ROOT = Path(__file__).resolve().parent

st.sidebar.header("Pipeline Configuration")
input_path_str = st.sidebar.text_input("Dataset Relative Path", value="data/raw/melb_data.csv")

clean_rel_path = Path(input_path_str).relative_to(Path(input_path_str).anchor) if Path(input_path_str).is_absolute() else Path(input_path_str)
dataset_path = (PROJECT_ROOT / clean_rel_path).relative_to(PROJECT_ROOT)

MODEL_MAP = {
    "Linear Regression": LinearRegressionModel,
    "Decision Tree Regressor": DecisionTreeModel,
    "Random Forest Regressor": RandomForestModel,
    "Gradient Boosting Regressor": GradientBoostingModel,
    "XGBoost Regressor": XGBoostModel,
}

selected_model_name = st.sidebar.selectbox(
    "Select Regression Model",
    options=list(MODEL_MAP.keys()),
    index=2
)

n_clusters = st.sidebar.slider("Number of K-Means Clusters", min_value=2, max_value=10, value=3)

if st.button("🚀 Run Pipeline"):
    try:
        str_dataset_path = str(dataset_path)

        # 1. Load & Clean Raw Data
        with st.spinner(f"Loading dataset from `{dataset_path}`..."):
            raw_df = pd.read_csv(dataset_path)

        cluster_stage = ClusterStage(data_path=str_dataset_path, n_clusters=n_clusters, random_state=42)
        df_cleaned = cluster_stage.preprocess_data(raw_df)

        drop_cols = ['Address', 'SellerG', 'Date', 'Postcode', 'CouncilArea']
        df_filtered = df_cleaned.drop(columns=[col for col in drop_cols if col in df_cleaned.columns])

        # 2. Train/Test Split BEFORE fitting KMeans or Scaler
        X = df_filtered.drop(columns=['Price'])
        y = df_filtered['Price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 3. Fit Scaler + KMeans on Training Data Only & Transform Both
        with st.spinner("Fitting K-Means on Training set..."):
            X_train_clustered = cluster_stage.fit_transform_train(X_train)
            X_test_clustered = cluster_stage.transform_test(X_test)

        st.subheader("1. Dataset & Cluster Split Overview")
        st.write(f"Training set: **{X_train_clustered.shape[0]} rows** | Test set: **{X_test_clustered.shape[0]} rows**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(X_train_clustered[['Rooms', 'Distance', 'Cluster']].head(5), use_container_width=True)
        with col2:
            cluster_counts = X_train_clustered['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Train Count']
            st.bar_chart(cluster_counts.set_index('Cluster'))

        # 4. Generate Interactive Map (using training set assignments for visualization)
        st.subheader("2. Property Geographical Map (Training Clusters)")
        with st.spinner("Rendering Interactive Map..."):
            map_df = X_train_clustered.copy()
            map_df['Price'] = y_train
            map_gen = MapGenerator(map_df, dataset_label=dataset_path.name)
            map_html = map_gen.generate_html(color_by="cluster")

            with st.container(border=True):
                components.html(map_html, height=500, scrolling=False)

        # 5. Build Preprocessor and Train Regressor
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

        model_class = MODEL_MAP[selected_model_name]
        chosen_regressor = model_class(data_path=str_dataset_path).model

        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', chosen_regressor)
        ])

        # 6. Fit Regressor on Training Set & Predict on Test Set
        with st.spinner(f"Training `{selected_model_name}`..."):
            full_pipeline.fit(X_train_clustered, y_train)
            y_pred = full_pipeline.predict(X_test_clustered)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader(f"3. Model Performance ({selected_model_name})")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"${mae:,.2f}")
        m2.metric("RMSE", f"${rmse:,.2f}")
        m3.metric("R² Score", f"{r2:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")