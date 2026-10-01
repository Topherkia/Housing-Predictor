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

from src.clustering_stage import ClusterStage
from src.map_generator import MapGenerator

from src.model_decision_tree import DecisionTreeModel
from src.model_gradient_boosting import GradientBoostingModel
from src.model_linear_regression import LinearRegressionModel
from src.model_random_forest import RandomForestModel
from src.model_xgboost import XGBoostModel


# -------------------------------------------------------------
# Page
# -------------------------------------------------------------

st.set_page_config(
    page_title="Melbourne Housing AI",
    layout="wide",
)

st.title(
    "🏡 Melbourne Housing Price Predictor"
)

st.write(
    """
    Melbourne housing price prediction using
    machine learning, XGBoost, K-Means clustering,
    RAG and Qwen.
    """
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent


# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------

st.sidebar.header(
    "Pipeline Configuration"
)

input_path_str = st.sidebar.text_input(
    "Dataset Relative Path",
    value="data/raw/melb_data.csv",
)

dataset_path = (
    PROJECT_ROOT
    / Path(input_path_str)
)


MODEL_MAP = {
    "Linear Regression":
        LinearRegressionModel,

    "Decision Tree Regressor":
        DecisionTreeModel,

    "Random Forest Regressor":
        RandomForestModel,

    "Gradient Boosting Regressor":
        GradientBoostingModel,

    "XGBoost Regressor":
        XGBoostModel,
}


selected_model_name = (
    st.sidebar.selectbox(
        "Select Regression Model",
        options=list(
            MODEL_MAP.keys()
        ),
        index=2,
    )
)


n_clusters = st.sidebar.slider(
    "Number of K-Means Clusters",
    min_value=2,
    max_value=10,
    value=3,
)


# -------------------------------------------------------------
# AI Assistant
# -------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader(
    "🤖 AI Assistant"
)

enable_ai = st.sidebar.checkbox(
    "Enable Qwen + RAG",
    value=False,
)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if st.button(
    "🚀 Run Pipeline"
):

    try:

        # -----------------------------------------------------
        # Load dataset
        # -----------------------------------------------------

        with st.spinner(
            "Loading dataset..."
        ):

            raw_df = pd.read_csv(
                dataset_path
            )

        st.subheader(
            "1. Dataset Overview"
        )

        st.write(
            f"Rows: **{raw_df.shape[0]}**"
        )

        st.write(
            f"Columns: **{raw_df.shape[1]}**"
        )

        st.dataframe(
            raw_df.head(),
            use_container_width=True,
        )

        # -----------------------------------------------------
        # Remove non-predictive columns
        # -----------------------------------------------------

        drop_cols = [
            "Address",
            "SellerG",
            "Date",
            "Postcode",
            "CouncilArea",
        ]

        df_filtered = raw_df.drop(
            columns=[
                col
                for col in drop_cols
                if col in raw_df.columns
            ]
        )

        # -----------------------------------------------------
        # Split FIRST
        # -----------------------------------------------------

        train_df, test_df = train_test_split(
            df_filtered,
            test_size=0.2,
            random_state=42,
        )

        # -----------------------------------------------------
        # K-Means
        # -----------------------------------------------------

        st.subheader(
            "2. K-Means Clustering"
        )

        cluster_stage = ClusterStage(
            data_path=str(
                dataset_path
            ),
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
            "K-Means fitted using training data only."
        )

        st.write(
            f"Training rows: "
            f"**{len(train_clustered)}**"
        )

        st.write(
            f"Test rows: "
            f"**{len(test_clustered)}**"
        )

        # -----------------------------------------------------
        # Model features
        # -----------------------------------------------------

        X_train = train_clustered.drop(
            columns=["Price"]
        )

        y_train = train_clustered[
            "Price"
        ]

        X_test = test_clustered.drop(
            columns=["Price"]
        )

        y_test = test_clustered[
            "Price"
        ]

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

        # -----------------------------------------------------
        # Preprocessing
        # -----------------------------------------------------

        numeric_transformer = (
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    )
                ]
            )
        )

        categorical_transformer = (
            Pipeline(
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
        )

        preprocessor = (
            ColumnTransformer(
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
        )

        # -----------------------------------------------------
        # Regression model
        # -----------------------------------------------------

        model_class = MODEL_MAP[
            selected_model_name
        ]

        chosen_regressor = (
            model_class(
                data_path=str(
                    dataset_path
                )
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

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        st.subheader(
            "3. Model Training"
        )

        with st.spinner(
            f"Training {selected_model_name}..."
        ):

            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = (
                pipeline.predict(
                    X_test
                )
            )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

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

        st.subheader(
            "4. Model Performance"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "MAE",
            f"${mae:,.2f}",
        )

        col2.metric(
            "RMSE",
            f"${rmse:,.2f}",
        )

        col3.metric(
            "R²",
            f"{r2:.4f}",
        )

        # -----------------------------------------------------
        # AI assistant
        # -----------------------------------------------------

        if enable_ai:

            st.subheader(
                "🤖 AI Housing Assistant"
            )

            st.info(
                "Qwen is loaded locally. "
                "The first run can take several minutes."
            )

            query = st.text_area(
                "Ask about a property",
                placeholder=(
                    "Example: Estimate the price "
                    "of a 4 bedroom property with "
                    "2 bathrooms and 600 square "
                    "metres of land."
                ),
            )

            if st.button(
                "Ask AI"
            ):

                if selected_model_name != (
                    "XGBoost Regressor"
                ):

                    st.warning(
                        "The AI assistant uses "
                        "XGBoost for its numerical "
                        "prediction. Select "
                        "XGBoost Regressor."
                    )

                elif not query.strip():

                    st.warning(
                        "Enter a housing question."
                    )

                else:

                    from src.qwen import (
                        QwenModel,
                    )

                    from src.rag import (
                        HousingRAG,
                    )

                    from src.ai_housing_assistant import (
                        HousingAIAssistant,
                    )

                    with st.spinner(
                        "Loading Qwen..."
                    ):

                        qwen = QwenModel()

                    with st.spinner(
                        "Building RAG knowledge base..."
                    ):

                        rag = HousingRAG(
                            knowledge_dir=(
                                PROJECT_ROOT
                                / "knowledge_base"
                            )
                        )

                    # -------------------------------------------------
                    # Use raw training data medians.
                    # -------------------------------------------------

                    required_features = [
                        "Rooms",
                        "Distance",
                        "Bedroom2",
                        "Bathroom",
                        "Car",
                        "Landsize",
                        "BuildingArea",
                    ]

                    feature_medians = {}

                    for feature in (
                        required_features
                    ):

                        feature_medians[
                            feature
                        ] = pd.to_numeric(
                            train_clustered[
                                feature
                            ],
                            errors="coerce",
                        ).median()

                    # -------------------------------------------------
                    # Get XGBoost estimator.
                    # -------------------------------------------------

                    xgb_model = (
                        pipeline.named_steps[
                            "regressor"
                        ]
                    )

                    # The AI assistant uses the original
                    # XGBoost feature schema.
                    #
                    # It therefore needs a model trained
                    # directly on these seven numerical
                    # features.

                    from xgboost import (
                        XGBRegressor,
                    )

                    xgb_features = required_features

                    xgb_model_direct = (
                        XGBRegressor(
                            n_estimators=(
                                xgb_model.n_estimators
                            ),
                            learning_rate=(
                                xgb_model.learning_rate
                            ),
                            max_depth=(
                                xgb_model.max_depth
                            ),
                            random_state=42,
                            objective=(
                                "reg:squarederror"
                            ),
                            n_jobs=-1,
                        )
                    )

                    xgb_model_direct.fit(
                        train_clustered[
                            xgb_features
                        ],
                        train_clustered[
                            "Price"
                        ],
                    )

                    assistant = (
                        HousingAIAssistant(
                            xgboost_model=(
                                xgb_model_direct
                            ),
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

                        result = (
                            assistant.answer(
                                query
                            )
                        )

                    # -------------------------------------------------
                    # Prediction
                    # -------------------------------------------------

                    st.markdown(
                        "### 💰 Estimated Price"
                    )

                    st.success(
                        f"AUD ${result['prediction']:,.0f}"
                    )

                    # -------------------------------------------------
                    # Extracted features
                    # -------------------------------------------------

                    st.markdown(
                        "### Extracted Features"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            [
                                result[
                                    "features"
                                ]
                            ]
                        ),
                        use_container_width=True,
                    )

                    if result[
                        "missing_filled"
                    ]:

                        st.warning(
                            "These inputs were "
                            "not supplied and were "
                            "filled using training "
                            "medians: "
                            + ", ".join(
                                result[
                                    "missing_filled"
                                ]
                            )
                        )

                    # -------------------------------------------------
                    # RAG results
                    # -------------------------------------------------

                    st.markdown(
                        "### 📚 Retrieved Knowledge"
                    )

                    for document in (
                        result[
                            "retrieved"
                        ]
                    ):

                        with st.expander(
                            document[
                                "source"
                            ]
                        ):

                            st.write(
                                document[
                                    "text"
                                ]
                            )

                            st.caption(
                                "Similarity: "
                                f"{document['score']:.3f}"
                            )

                    # -------------------------------------------------
                    # Qwen explanation
                    # -------------------------------------------------

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