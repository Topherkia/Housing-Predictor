from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from src.ai_housing_assistant import HousingAIAssistant
from src.clustering_stage import ClusterStage
from src.map_generator import MapGenerator
from src.model_decision_tree import DecisionTreeModel
from src.model_gradient_boosting import GradientBoostingModel
from src.model_linear_regression import LinearRegressionModel
from src.model_random_forest import RandomForestModel
from src.model_xgboost import XGBoostModel


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Melbourne Housing AI",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 Melbourne Housing Price Predictor")

st.markdown(
    """
    Historical Melbourne housing price prediction using:

    **Machine Learning · XGBoost · K-Means · SHAP · RAG · Qwen · Folium**
    """
)

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# CONSTANTS
# ============================================================

AI_FEATURES = [
    "Rooms",
    "Distance",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
]

MODEL_MAP = {
    "Linear Regression": LinearRegressionModel,
    "Decision Tree Regressor": DecisionTreeModel,
    "Random Forest Regressor": RandomForestModel,
    "Gradient Boosting Regressor": GradientBoostingModel,
    "XGBoost Regressor": XGBoostModel,
}


# ============================================================
# CACHED AI COMPONENTS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_qwen():
    from src.qwen import QwenModel

    return QwenModel()


@st.cache_resource(show_spinner=False)
def load_rag(knowledge_dir: str):
    from src.rag import HousingRAG

    return HousingRAG(
        knowledge_dir=Path(knowledge_dir)
    )


@st.cache_resource(show_spinner=False)
def train_ai_xgboost(
    train_data: pd.DataFrame,
):
    """
    Train the dedicated seven-feature model used by the AI assistant.

    This model intentionally has the exact same seven-feature schema expected
    by Qwen feature extraction.
    """

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    X = train_data[AI_FEATURES].copy()
    y = train_data["Price"]

    model.fit(X, y)

    return model


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Pipeline Configuration")

input_path_str = st.sidebar.text_input(
    "Dataset Relative Path",
    value="data/raw/melb_data.csv",
)

dataset_path = PROJECT_ROOT / Path(
    input_path_str
)

selected_model_name = st.sidebar.selectbox(
    "Regression Model",
    options=list(MODEL_MAP.keys()),
    index=2,
)

n_clusters = st.sidebar.slider(
    "K-Means Clusters",
    min_value=2,
    max_value=10,
    value=3,
)

st.sidebar.divider()

enable_ai = st.sidebar.checkbox(
    "🤖 Enable Qwen + RAG",
    value=False,
)

run_pipeline = st.button(
    "🚀 Run Pipeline",
    type="primary",
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if run_pipeline:

    try:

        # ------------------------------------------------------
        # Load
        # ------------------------------------------------------

        with st.spinner("Loading dataset..."):

            raw_df = pd.read_csv(
                dataset_path
            )

        st.subheader("1. Dataset Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            f"{len(raw_df):,}",
        )

        c2.metric(
            "Columns",
            f"{len(raw_df.columns):,}",
        )

        if "Price" in raw_df.columns:
            c3.metric(
                "Median Sale Price",
                f"${raw_df['Price'].median():,.0f}",
            )

        st.dataframe(
            raw_df.head(20),
            use_container_width=True,
        )

        # ------------------------------------------------------
        # Remove identifiers / leakage-prone columns
        # ------------------------------------------------------

        drop_cols = [
            "Address",
            "SellerG",
            "Date",
            "Postcode",
            "CouncilArea",
        ]

        df_filtered = raw_df.drop(
            columns=[
                column
                for column in drop_cols
                if column in raw_df.columns
            ]
        )

        # ------------------------------------------------------
        # Split BEFORE fitting transformations
        # ------------------------------------------------------

        train_df, test_df = train_test_split(
            df_filtered,
            test_size=0.20,
            random_state=42,
        )

        # ------------------------------------------------------
        # K-MEANS
        # ------------------------------------------------------

        st.subheader("2. K-Means Clustering")

        cluster_stage = ClusterStage(
            data_path=str(dataset_path),
            n_clusters=n_clusters,
            random_state=42,
        )

        with st.spinner(
            "Fitting K-Means on training data..."
        ):

            cluster_stage.fit(
                train_df
            )

            train_clustered = (
                cluster_stage.transform(
                    train_df
                )
            )

            test_clustered = (
                cluster_stage.transform(
                    test_df
                )
            )

        st.success(
            "K-Means was fitted using training data only."
        )

        # ------------------------------------------------------
        # REGRESSION DATA
        # ------------------------------------------------------

        X_train = train_clustered.drop(
            columns=["Price"]
        )

        y_train = train_clustered["Price"]

        X_test = test_clustered.drop(
            columns=["Price"]
        )

        y_test = test_clustered["Price"]

        categorical_cols = (
            X_train
            .select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            )
            .columns
            .tolist()
        )

        numeric_cols = (
            X_train
            .select_dtypes(
                include=[
                    "int64",
                    "float64",
                    "Int64",
                ]
            )
            .columns
            .tolist()
        )

        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                )
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    numeric_cols,
                ),
                (
                    "cat",
                    categorical_transformer,
                    categorical_cols,
                ),
            ]
        )

        # ------------------------------------------------------
        # REGRESSION MODEL
        # ------------------------------------------------------

        model_class = MODEL_MAP[
            selected_model_name
        ]

        chosen_regressor = (
            model_class(
                data_path=str(dataset_path)
            ).model
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "regressor",
                    chosen_regressor,
                ),
            ]
        )

        # ------------------------------------------------------
        # TRAIN
        # ------------------------------------------------------

        st.subheader("3. Model Training")

        with st.spinner(
            f"Training {selected_model_name}..."
        ):

            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(
                X_test
            )

        # ------------------------------------------------------
        # EVALUATION
        # ------------------------------------------------------

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

        st.subheader("4. Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"${mae:,.0f}",
        )

        col2.metric(
            "RMSE",
            f"${rmse:,.0f}",
        )

        col3.metric(
            "R²",
            f"{r2:.4f}",
        )

        # ------------------------------------------------------
        # ACTUAL VS PREDICTED TABLE
        # ------------------------------------------------------

        prediction_df = test_clustered.copy()

        prediction_df["Predicted"] = predictions

        prediction_df["ErrorPct"] = (
            (
                prediction_df["Predicted"]
                - prediction_df["Price"]
            )
            / prediction_df["Price"].replace(
                0,
                pd.NA,
            )
            * 100
        )

        st.subheader(
            "5. Predictions"
        )

        display_columns = [
            column
            for column in [
                "Suburb",
                "Type",
                "Rooms",
                "Bathroom",
                "Distance",
                "Price",
                "Predicted",
                "ErrorPct",
                "Cluster",
                "Lattitude",
                "Longtitude",
            ]
            if column in prediction_df.columns
        ]

        st.dataframe(
            prediction_df[
                display_columns
            ].head(100),
            use_container_width=True,
        )

        # ------------------------------------------------------
        # MAP
        # ------------------------------------------------------

        st.subheader(
            "6. Melbourne Property Map"
        )

        if not {
            "Lattitude",
            "Longtitude",
        }.issubset(prediction_df.columns):

            st.warning(
                "The dataset does not contain "
                "Lattitude/Longtitude columns, "
                "so the map cannot be displayed."
            )

        else:

            color_by = "price"
            map_df = prediction_df.copy()

            # The map generator expects actual sale-price information.
            try:

                housing_map = MapGenerator(
                    map_df,
                    dataset_label=(
                        f"{selected_model_name} "
                        f"test set"
                    ),
                )

                map_html = (
                    housing_map.generate_html(
                        color_by=color_by,
                        cluster=True,
                        heatmap=True,
                        zoom_start=11,
                    )
                )

                st.components.v1.html(
                    map_html,
                    height=700,
                    scrolling=False,
                )

            except Exception as map_error:

                st.error(
                    "Map generation failed: "
                    f"{map_error}"
                )

        # ------------------------------------------------------
        # MODEL ERROR SUMMARY
        # ------------------------------------------------------

        st.subheader(
            "7. Prediction Error Summary"
        )

        valid_errors = (
            prediction_df["ErrorPct"]
            .dropna()
        )

        if not valid_errors.empty:

            ec1, ec2, ec3 = st.columns(3)

            ec1.metric(
                "Median Error",
                f"{valid_errors.median():.2f}%",
            )

            ec2.metric(
                "Mean Absolute Error %",
                f"{valid_errors.abs().mean():.2f}%",
            )

            ec3.metric(
                "Within ±10%",
                f"{(valid_errors.abs() <= 10).mean() * 100:.1f}%",
            )

        # ------------------------------------------------------
        # AI ASSISTANT
        # ------------------------------------------------------

        if enable_ai:

            st.subheader(
                "🤖 AI Housing Assistant"
            )

            st.caption(
                "Qwen extracts the property inputs, "
                "XGBoost performs the numerical prediction, "
                "and RAG supplies project documentation."
            )

            query = st.text_area(
                "Ask about a property",
                placeholder=(
                    "Estimate the price of a 4 bedroom "
                    "property with 2 bathrooms and "
                    "600 square metres of land."
                ),
            )

            if st.button(
                "Ask AI",
                key="ask_ai_button",
            ):

                if not query.strip():

                    st.warning(
                        "Enter a housing question."
                    )

                else:

                    with st.spinner(
                        "Loading Qwen..."
                    ):

                        qwen = load_qwen()

                    with st.spinner(
                        "Loading RAG knowledge base..."
                    ):

                        rag = load_rag(
                            str(
                                PROJECT_ROOT
                                / "knowledge_base"
                            )
                        )

                    # --------------------------------------------------
                    # Dedicated seven-feature AI training data
                    # --------------------------------------------------

                    ai_train = train_clustered.copy()

                    for feature in AI_FEATURES:

                        if feature not in ai_train.columns:
                            raise ValueError(
                                f"AI feature missing from dataset: {feature}"
                            )

                        ai_train[feature] = pd.to_numeric(
                            ai_train[feature],
                            errors="coerce",
                        )

                    ai_train = ai_train.dropna(
                        subset=["Price"]
                    )

                    # XGBoost can handle remaining feature NaNs.
                    feature_medians = {
                        feature: float(
                            ai_train[feature].median()
                        )
                        for feature in AI_FEATURES
                    }

                    with st.spinner(
                        "Preparing AI prediction model..."
                    ):

                        ai_model = train_ai_xgboost(
                            ai_train[
                                AI_FEATURES
                                + ["Price"]
                            ].copy()
                        )

                    assistant = (
                        HousingAIAssistant(
                            xgboost_model=ai_model,
                            qwen_model=qwen,
                            rag=rag,
                            feature_medians=(
                                feature_medians
                            ),
                        )
                    )

                    with st.spinner(
                        "Qwen is analysing the request..."
                    ):

                        result = assistant.answer(
                            query
                        )

                    # --------------------------------------------------
                    # Prediction
                    # --------------------------------------------------

                    st.markdown(
                        "### 💰 Estimated Historical Price"
                    )

                    st.success(
                        f"AUD ${result['prediction']:,.0f}"
                    )

                    st.caption(
                        "This is a machine-learning estimate "
                        "from historical Melbourne housing data, "
                        "not a professional property valuation."
                    )

                    # --------------------------------------------------
                    # Features
                    # --------------------------------------------------

                    st.markdown(
                        "### Extracted Model Inputs"
                    )

                    feature_table = pd.DataFrame(
                        [
                            result["features"]
                        ]
                    )

                    st.dataframe(
                        feature_table,
                        use_container_width=True,
                    )

                    if result[
                        "missing_filled"
                    ]:

                        st.warning(
                            "Missing inputs were filled using "
                            "training-data medians: "
                            + ", ".join(
                                result[
                                    "missing_filled"
                                ]
                            )
                        )

                    # --------------------------------------------------
                    # RAG
                    # --------------------------------------------------

                    st.markdown(
                        "### 📚 Retrieved Knowledge"
                    )

                    for document in result[
                        "retrieved"
                    ]:

                        with st.expander(
                            document["source"]
                        ):

                            st.write(
                                document["text"]
                            )

                            st.caption(
                                "Similarity: "
                                f"{document['score']:.3f}"
                            )

                    # --------------------------------------------------
                    # Explanation
                    # --------------------------------------------------

                    st.markdown(
                        "### 🧠 AI Explanation"
                    )

                    st.write(
                        result[
                            "explanation"
                        ]
                    )

    except Exception as error:

        st.error(
            f"An error occurred: {error}"
        )