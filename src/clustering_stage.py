import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class ClusterStage:
    """Handles preprocessing, cleaning, and K-Means clustering on raw input data."""
    
    def __init__(self, data_path: str, n_clusters: int = 3, random_state: int = 42):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def preprocess_and_cluster(self, df: pd.DataFrame = None):
        if df is None:
            df = self.load_data()
        else:
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

        # Clustering Stage
        cluster_features = [
            'Rooms', 'Bathroom', 'Car', 
            'Landsize', 'BuildingArea', 'Distance'
        ]
        if 'Lattitude' in df.columns and 'Longtitude' in df.columns:
            cluster_features.extend(['Lattitude', 'Longtitude'])

        valid_cluster_idx = df[cluster_features].dropna().index
        X_cluster = df.loc[valid_cluster_idx, cluster_features]

        # Standardize & Fit KMeans
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_cluster)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Assign Cluster assignments back
        df['Cluster'] = np.nan
        df.loc[valid_cluster_idx, 'Cluster'] = cluster_labels
        
        df = df.dropna(subset=['Cluster']).reset_index(drop=True)
        df['Cluster'] = df['Cluster'].astype(int)

        return df, self.kmeans, self.scaler