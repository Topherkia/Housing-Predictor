import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class ClusterStage:
    """Handles preprocessing, cleaning, and K-Means clustering."""
    
    def __init__(self, data_path: str = None, n_clusters: int = 3, random_state: int = 42):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.scaler = StandardScaler()
        self.cluster_features = [
            'Rooms', 'Bathroom', 'Car', 
            'Landsize', 'BuildingArea', 'Distance'
        ]

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and imputes raw data prior to splitting."""
        df = df.copy()

        # Standardize column names
        df.columns = df.columns.str.strip()
        column_mapping = {
            'Bedroom': 'Bedroom2',
            'Latitude': 'Lattitude',
            'Longitude': 'Longtitude'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Convert numeric columns
        numeric_cols = [
            'Price', 'Bedroom2', 'Bathroom', 'Car', 
            'Landsize', 'BuildingArea', 'YearBuilt', 
            'Distance', 'Propertycount'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows missing target variable 'Price'
        if 'Price' in df.columns:
            df = df.dropna(subset=['Price']).reset_index(drop=True)

        # Outlier handling
        if 'Landsize' in df.columns:
            df = df[df['Landsize'] < 10000]

        if 'BuildingArea' in df.columns:
            df.loc[df['BuildingArea'] == 0, 'BuildingArea'] = np.nan
            df.loc[df['BuildingArea'] > 1500, 'BuildingArea'] = np.nan

        # Date handling and feature engineering
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df['SaleYear'] = df['Date'].dt.year
            df['SaleMonth'] = df['Date'].dt.month

        # Impute missing numerical attributes
        num_impute_cols = [
            'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
            'BuildingArea', 'YearBuilt', 'Distance', 'Propertycount'
        ]
        for col in num_impute_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        if 'YearBuilt' in df.columns and 'SaleYear' in df.columns:
            df['PropertyAge'] = (df['SaleYear'] - df['YearBuilt']).clip(lower=0)

        return df

    def fit_transform_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fits StandardScaler & KMeans strictly on train features and adds Cluster column."""
        X_train = X_train.copy()
        features = [col for col in self.cluster_features if col in X_train.columns]
        if 'Lattitude' in X_train.columns and 'Longtitude' in X_train.columns:
            features.extend(['Lattitude', 'Longtitude'])

        X_cluster = X_train[features]
        X_scaled = self.scaler.fit_transform(X_cluster)
        X_train['Cluster'] = self.kmeans.fit_predict(X_scaled)
        return X_train

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Transforms test set using fitted StandardScaler & KMeans."""
        X_test = X_test.copy()
        features = [col for col in self.cluster_features if col in X_test.columns]
        if 'Lattitude' in X_test.columns and 'Longtitude' in X_test.columns:
            features.extend(['Lattitude', 'Longtitude'])

        X_cluster = X_test[features]
        X_scaled = self.scaler.transform(X_cluster)
        X_test['Cluster'] = self.kmeans.predict(X_scaled)
        return X_test