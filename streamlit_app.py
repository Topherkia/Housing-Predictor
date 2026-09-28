from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Import classes from src
from src.clustering_stage import ClusterStage
from src.model_decision_tree import DecisionTreeModel
from src.model_gradient_boosting import GradientBoostingModel
from src.model_linear_regression import LinearRegressionModel
from src.model_random_forest import RandomForestModel
from src.model_xgboost import XGBoostModel

# Map feature
from map_demo import COLOUR_MODES, TYPE_LABELS, build_map, clean_coordinates, load_dataset

st.set_page_config(page_title="Melbourne Housing Price Predictor", layout="wide")

st.title("🏡 Melbourne Housing Price Predictor")
st.write("A pipeline combining **K-Means Clustering** and **Machine Learning Regressors**.")

# Project root directory reference
PROJECT_ROOT = Path(__file__).resolve().parent

# Internal column used to link model results back to the original rows (never used as a feature)
ROW_ID = "_row_id"

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.header("Pipeline Configuration")

# Dataset relative path input defaulting to data/raw/melb_data.csv
input_path_str = st.sidebar.text_input("Dataset Relative Path", value="data/raw/melb_data.csv")

# Resolve path relative to project root without allowing absolute OS paths
clean_rel_path = Path(input_path_str).relative_to(Path(input_path_str).anchor) if Path(input_path_str).is_absolute() else Path(input_path_str)
dataset_path = (PROJECT_ROOT / clean_rel_path).relative_to(PROJECT_ROOT)

# Regression model mapping
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
    index=2  # Default to Random Forest
)

n_clusters = st.sidebar.slider("Number of K-Means Clusters", min_value=2, max_value=10, value=3)

current_config = {"dataset": str(dataset_path), "model": selected_model_name, "n_clusters": n_clusters}


# -------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------
def run_pipeline(dataset_path: Path, model_name: str, n_clusters: int) -> dict:
    """Run clustering + regression and return everything the page needs to display."""
    # Convert path to string for modules requiring string representation
    str_dataset_path = str(dataset_path)

    # 1. Load Raw Data
    with st.spinner(f"Loading dataset from `{dataset_path}`..."):
        raw_df = pd.read_csv(dataset_path)
    raw_df[ROW_ID] = range(len(raw_df))

    # 2. Data Cleaning
    drop_cols = ['Address', 'SellerG', 'Date', 'Postcode', 'CouncilArea']
    df_filtered = raw_df.drop(columns=[col for col in drop_cols if col in raw_df.columns])

    # 3. Clustering Stage
    with st.spinner("Running K-Means Clustering..."):
        cluster_stage = ClusterStage(data_path=str_dataset_path, n_clusters=n_clusters, random_state=42)
        df_clustered, kmeans_model, cluster_scaler = cluster_stage.preprocess_and_cluster(df_filtered)

    cluster_counts = df_clustered['Cluster'].value_counts().sort_index().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    # 4. Prepare Features & Model
    X = df_clustered.drop(columns=['Price', ROW_ID])
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

    # Instantiate chosen src class model
    model_class = MODEL_MAP[model_name]
    chosen_regressor = model_class(data_path=str_dataset_path).model

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', chosen_regressor)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Model Training & Evaluation
    with st.spinner(f"Training `{model_name}`..."):
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    # 6. Results for the map: original rows + cluster + test-set predictions
    map_df = raw_df.set_index(ROW_ID).loc[df_clustered[ROW_ID].values].reset_index()
    map_df['Cluster'] = df_clustered['Cluster'].values
    predicted = pd.Series(y_pred, index=X_test.index)
    map_df['Predicted'] = predicted.reindex(df_clustered.index).round(0).values
    map_df['ErrorPct'] = ((map_df['Predicted'] - map_df['Price']) / map_df['Price'] * 100).round(1)
    map_df = clean_coordinates(map_df.drop(columns=[ROW_ID])).reset_index(drop=True)

    return {
        "config": {"dataset": str(dataset_path), "model": model_name, "n_clusters": n_clusters},
        "raw_shape": raw_df.drop(columns=[ROW_ID]).shape,
        "raw_head": raw_df.drop(columns=[ROW_ID]).head(5),
        "clustered_shape": df_clustered.drop(columns=[ROW_ID]).shape,
        "clustered_head": df_clustered[['Rooms', 'Price', 'Distance', 'Cluster']].head(5),
        "cluster_counts": cluster_counts,
        "metrics": metrics,
        "map_df": map_df,
    }


# -------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------
if st.button("🚀 Run Pipeline"):
    try:
        st.session_state["results"] = run_pipeline(dataset_path, selected_model_name, n_clusters)
    except Exception as e:
        st.session_state.pop("results", None)
        st.error(f"An error occurred: {e}")

# Results are kept in session_state so they stay visible while the map filters are used
results = st.session_state.get("results")

if results:
    cfg = results["config"]
    if cfg != current_config:
        st.info("The sidebar settings changed since the last run. Press **Run Pipeline** to "
                "update the results and the cluster / error colours on the map.")

    st.subheader("1. Dataset Overview")
    st.write(f"Raw shape: **{results['raw_shape'][0]} rows, {results['raw_shape'][1]} columns**")
    st.dataframe(results["raw_head"], width="stretch")

    st.subheader("2. Clustering Results")
    st.write(f"Clustered shape: **{results['clustered_shape'][0]} rows, {results['clustered_shape'][1]} columns**")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(results["clustered_head"], width="stretch")
    with col2:
        st.bar_chart(results["cluster_counts"].set_index('Cluster'))

    st.subheader(f"3. Model Performance ({cfg['model']})")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${results['metrics']['MAE']:,.2f}")
    m2.metric("RMSE", f"${results['metrics']['RMSE']:,.2f}")
    m3.metric("R² Score", f"{results['metrics']['R2']:.4f}")


# -------------------------------------------------------------
# Property Map (Folium) – with colour modes and sidebar filters
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_map_data(path_str: str) -> pd.DataFrame:
    return load_dataset(str(PROJECT_ROOT / path_str))


@st.cache_data(show_spinner=False)
def render_property_map(map_df: pd.DataFrame, color_by: str, heatmap: bool, label: str) -> str:
    """Build the Folium map and return it as a full HTML document (cached)."""
    folium_map = build_map(map_df, dataset_label=label, heatmap=heatmap, color_by=color_by)
    return folium_map.get_root().render()


def sidebar_map_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Draw the map filters in the sidebar and return the filtered rows."""
    st.sidebar.divider()
    st.sidebar.header("🗺️ Map Filters")
    mask = pd.Series(True, index=df.index)

    if "Type" in df.columns:
        types = sorted(df["Type"].dropna().unique())
        chosen = st.sidebar.multiselect("Property type", types, default=types,
                                        format_func=lambda t: TYPE_LABELS.get(t, t))
        mask &= df["Type"].isin(chosen)

    if "Regionname" in df.columns:
        regions = sorted(df["Regionname"].dropna().unique())
        chosen = st.sidebar.multiselect("Region", regions, placeholder="All regions")
        if chosen:
            mask &= df["Regionname"].isin(chosen)

    def range_slider(col, label, step, fmt=None, as_int=True):
        nonlocal mask
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            return
        lo, hi = values.min(), values.max()
        if as_int:
            lo, hi = int(lo), int(hi)
        if lo == hi:
            return
        sel = st.sidebar.slider(label, lo, hi, (lo, hi), step=step, format=fmt)
        # Rows with an unknown value are kept only while the slider is untouched
        if sel != (lo, hi):
            mask &= values.between(*sel)

    if "Price" in df.columns:
        range_slider("Price", "Price (AUD)", 50_000, fmt="$%d")
    if "Rooms" in df.columns:
        range_slider("Rooms", "Rooms", 1)
    if "Distance" in df.columns:
        range_slider("Distance", "Distance to CBD (km)", 0.5, fmt="%.1f km", as_int=False)
    if "YearBuilt" in df.columns:
        range_slider("YearBuilt", "Year built", 1)

    return df[mask]


st.divider()
st.subheader("🗺️ Property Map")
st.write("Properties of the selected dataset, placed by latitude and longitude. "
         "Zoom in to split the clusters and click a marker for its details.")

# Use the pipeline output (with clusters and predictions) when it matches the chosen dataset
has_model_output = bool(results) and results["config"]["dataset"] == str(dataset_path)

try:
    base_df = results["map_df"] if has_model_output else load_map_data(str(dataset_path))
except Exception as e:
    base_df = None
    st.error(f"Could not load the map data: {e}")

if base_df is not None:
    filtered_df = sidebar_map_filters(base_df)

    available_modes = ["price", "cluster", "error"] if has_model_output else ["price"]
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        color_by = st.radio("Colour markers by", available_modes, horizontal=True,
                            format_func=lambda k: COLOUR_MODES[k])
    with ctrl2:
        map_heatmap = st.checkbox("Price heat-map layer", value=False)
    with ctrl3:
        show_map = st.toggle("Show map", value=True)

    if not has_model_output:
        st.caption("Run the pipeline to colour the map by **K-Means cluster** or **prediction error**.")

    map_df = filtered_df
    if color_by == "error":
        map_df = filtered_df.dropna(subset=["ErrorPct"])
        st.caption(f"Prediction error is only available for the test set "
                   f"({results['map_df']['ErrorPct'].notna().sum():,} properties, "
                   f"model: {results['config']['model']}).")

    st.caption(f"Showing **{len(map_df):,}** of {len(base_df):,} properties from `{dataset_path}`")

    if color_by == "error" and not map_df.empty:
        abs_err = map_df["ErrorPct"].abs()
        e1, e2, e3 = st.columns(3)
        e1.metric("Within ±10%", f"{(abs_err <= 10).mean():.0%}")
        e2.metric("Median absolute error", f"{abs_err.median():.1f}%")
        e3.metric("Off by more than 30%", f"{(abs_err > 30).mean():.0%}")

    if show_map:
        if map_df.empty:
            st.warning("No properties match the current filters.")
        else:
            try:
                with st.spinner("Building the map..."):
                    label = f"{dataset_path.name} – {COLOUR_MODES[color_by].lower()}"
                    map_html = render_property_map(map_df, color_by, map_heatmap, label)
                if hasattr(st, "iframe"):          # Streamlit >= 1.52
                    st.iframe(map_html, height=650)
                else:                              # older Streamlit versions
                    import streamlit.components.v1 as components
                    components.html(map_html, height=650)
            except Exception as e:
                st.error(f"Could not build the map: {e}")

    if color_by == "error" and not map_df.empty:
        with st.expander("Suburbs where the model is least accurate"):
            worst = (map_df.assign(AbsErrorPct=map_df["ErrorPct"].abs())
                     .groupby("Suburb")
                     .agg(Properties=("AbsErrorPct", "size"),
                          MeanAbsErrorPct=("AbsErrorPct", "mean"),
                          MeanErrorPct=("ErrorPct", "mean"))
                     .query("Properties >= 5")
                     .sort_values("MeanAbsErrorPct", ascending=False)
                     .head(10).round(1))
            st.dataframe(worst, width="stretch")
            st.caption("Suburbs with at least 5 test-set properties. "
                       "A positive mean error means the model tends to over-estimate there.")
