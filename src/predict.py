"""
Reusable customer assignment function for the Wholesale Customer Segmentation project.
Author: Darshan Suresh S
"""
import numpy as np
import pandas as pd
import joblib

SPENDING_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

CLUSTER_LABELS = {
    0: 'Grocery & Household Retailers',
    1: 'High-Volume Bulk Buyers',
    2: 'Fresh-Focused Food Service (Horeca)',
}


def load_artifacts(model_path='models/cluster_model.joblib', scaler_path='models/preprocessor.joblib'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def assign_cluster(customer_record: dict, model, scaler):
    """
    customer_record: dict with keys Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
    (raw annual spending values, not yet transformed)
    """
    record_df = pd.DataFrame([customer_record])[SPENDING_COLS]
    record_log = np.log1p(record_df)
    record_scaled = scaler.transform(record_log)
    cluster_id = int(model.predict(record_scaled)[0])
    return {'cluster': cluster_id, 'label': CLUSTER_LABELS.get(cluster_id, 'Unknown')}


if __name__ == '__main__':
    model, scaler = load_artifacts('models/cluster_model.joblib', 'models/preprocessor.joblib')
    sample = {'Fresh': 3000, 'Milk': 8000, 'Grocery': 15000, 'Frozen': 500,
              'Detergents_Paper': 6000, 'Delicassen': 700}
    print(assign_cluster(sample, model, scaler))