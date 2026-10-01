import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusterStage:
    """
    Handles preprocessing and K-Means clustering.
    
    """

    CLUSTER_FEATURES = [
        "Rooms",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "Distance",
    ]

    def __init__(
        self,
        data_path: str,
        n_clusters: int = 3,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.kmeans = None
        self.scaler = None

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the dataset.

        This method does NOT fit K-Means.
        """

        df = df.copy()

        # ---------------------------------------------------------
        # Standardize column names
        # ---------------------------------------------------------

        df.columns = df.columns.str.strip()

        column_mapping = {
            "Bedroom": "Bedroom2",
            "Latitude": "Lattitude",
            "Longitude": "Longtitude",
        }

        df.rename(columns=column_mapping, inplace=True)

        # ---------------------------------------------------------
        # Convert numeric columns
        # ---------------------------------------------------------

        numeric_cols = [
            "Price",
            "Rooms",
            "Bedroom2",
            "Bathroom",
            "Car",
            "Landsize",
            "BuildingArea",
            "YearBuilt",
            "Distance",
            "Propertycount",
            "Lattitude",
            "Longtitude",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

        # ---------------------------------------------------------
        # Price
        # ---------------------------------------------------------

        if "Price" in df.columns:
            df = df.dropna(
                subset=["Price"]
            ).reset_index(drop=True)

        # ---------------------------------------------------------
        # Outliers
        # ---------------------------------------------------------

        if "Landsize" in df.columns:
            df.loc[
                df["Landsize"] >= 10000,
                "Landsize",
            ] = np.nan

        if "BuildingArea" in df.columns:
            df.loc[
                df["BuildingArea"] <= 0,
                "BuildingArea",
            ] = np.nan

            df.loc[
                df["BuildingArea"] > 1500,
                "BuildingArea",
            ] = np.nan

        # ---------------------------------------------------------
        # Date feature engineering
        # ---------------------------------------------------------

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                dayfirst=True,
                errors="coerce",
            )

            df["SaleYear"] = df["Date"].dt.year
            df["SaleMonth"] = df["Date"].dt.month

        # ---------------------------------------------------------
        # Numerical imputation
        # ---------------------------------------------------------

        impute_cols = [
            "Bedroom2",
            "Bathroom",
            "Car",
            "Landsize",
            "BuildingArea",
            "YearBuilt",
            "Distance",
            "Propertycount",
        ]

        for col in impute_cols:

            if col in df.columns:

                median = df[col].median()

                df[col] = df[col].fillna(median)

        # ---------------------------------------------------------
        # Property age
        # ---------------------------------------------------------

        if (
            "YearBuilt" in df.columns
            and "SaleYear" in df.columns
        ):

            df["PropertyAge"] = (
                df["SaleYear"]
                - df["YearBuilt"]
            ).clip(lower=0)

        return df

    def fit(self, df: pd.DataFrame):
        """
        Fit scaler and K-Means using training data only.
        """

        df = self.preprocess_data(df)

        available_features = [
            col
            for col in self.CLUSTER_FEATURES
            if col in df.columns
        ]

        if len(available_features) != len(
            self.CLUSTER_FEATURES
        ):
            missing = set(
                self.CLUSTER_FEATURES
            ) - set(available_features)

            raise ValueError(
                f"Missing clustering columns: {missing}"
            )

        X_cluster = df[
            available_features
        ].copy()

        # ---------------------------------------------------------
        # Fit scaler ONLY on training data
        # ---------------------------------------------------------

        self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(
            X_cluster
        )

        # ---------------------------------------------------------
        # Fit K-Means ONLY on training data
        # ---------------------------------------------------------

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )

        self.kmeans.fit(X_scaled)

        return self

    def transform(self, df: pd.DataFrame):
        """
        Apply the already-fitted scaler and K-Means model.
        """

        if self.scaler is None:
            raise RuntimeError(
                "ClusterStage must be fitted before transform()."
            )

        if self.kmeans is None:
            raise RuntimeError(
                "K-Means model has not been fitted."
            )

        df = self.preprocess_data(df)

        available_features = [
            col
            for col in self.CLUSTER_FEATURES
            if col in df.columns
        ]

        X_cluster = df[
            available_features
        ].copy()

        X_scaled = self.scaler.transform(
            X_cluster
        )

        labels = self.kmeans.predict(
            X_scaled
        )

        df["Cluster"] = labels

        return df

    def fit_transform(self, df: pd.DataFrame):
        """
        Fit K-Means and transform the same training dataset.
        """

        self.fit(df)

        return self.transform(df)

    def preprocess_and_cluster(
        self,
        df: pd.DataFrame = None,
    ):
        """
        Backwards-compatible method.

        This method is intended for exploratory use.
        For proper model evaluation, use fit() on training data
        followed by transform() on test data.
        """

        if df is None:
            df = self.load_data()

        df = self.fit_transform(df)

        return (
            df,
            self.kmeans,
            self.scaler,
        )