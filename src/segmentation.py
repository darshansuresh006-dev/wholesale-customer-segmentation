"""
Reusable utilities for the Wholesale Customer Segmentation project.
Author: Darshan Suresh S
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RANDOM_STATE = 42
SPENDING_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
VALIDATION_COLS = ['Channel', 'Region']
CHOSEN_K = 3

CHANNEL_LABELS = {1: 'Horeca', 2: 'Retail'}
REGION_LABELS = {1: 'Lisbon', 2: 'Oporto', 3: 'Other'}


def load_data(path):
    return pd.read_csv(path)


def transform_and_scale(df):
    """Apply log1p to spending columns, then standard-scale. Returns (X_scaled_df, scaler, df_log)."""
    df_log = df.copy()
    for col in SPENDING_COLS:
        df_log[col] = np.log1p(df_log[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log[SPENDING_COLS])
    X_scaled = pd.DataFrame(X_scaled, columns=SPENDING_COLS)
    return X_scaled, scaler, df_log


def fit_kmeans(X_scaled, k=CHOSEN_K, random_state=RANDOM_STATE):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_clusters(df, labels, cluster_col='cluster'):
    """Build original-unit average/median/count profile table for each cluster."""
    df = df.copy()
    df[cluster_col] = labels
    profile = df.groupby(cluster_col)[SPENDING_COLS].agg(['mean', 'median'])
    counts = df[cluster_col].value_counts().sort_index()
    return profile.round(1), counts