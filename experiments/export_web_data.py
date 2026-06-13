import os
import sys
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.data_loader import load_csv
from config import PROCESSED_DATA_PATH

def main():
    raw_data = load_csv(PROCESSED_DATA_PATH)
    X = np.array(raw_data)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X)
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X)
    
    # Align labels
    labels_km = kmeans.predict(X)
    labels_gmm = gmm.predict(X)
    
    mean_km_0 = kmeans.cluster_centers_[0]
    mean_gmm_0 = gmm.means_[0]
    mean_gmm_1 = gmm.means_[1]
    
    if np.linalg.norm(mean_km_0 - mean_gmm_1) < np.linalg.norm(mean_km_0 - mean_gmm_0):
        labels_gmm = 1 - labels_gmm
        probs = gmm.predict_proba(X)
        probs = probs[:, [1, 0]]
        means = gmm.means_[[1, 0]]
        covs = gmm.covariances_[[1, 0]]
    else:
        probs = gmm.predict_proba(X)
        means = gmm.means_
        covs = gmm.covariances_

    data = {
        'points': [
            {
                'x': float(X[i, 0]),
                'y': float(X[i, 1]),
                'km_label': int(labels_km[i]),
                'gmm_label': int(labels_gmm[i]),
                'gmm_prob_0': float(probs[i, 0]),
                'gmm_prob_1': float(probs[i, 1])
            } for i in range(len(X))
        ],
        'kmeans': {
            'centroids': [
                {'x': float(kmeans.cluster_centers_[0, 0]), 'y': float(kmeans.cluster_centers_[0, 1])},
                {'x': float(kmeans.cluster_centers_[1, 0]), 'y': float(kmeans.cluster_centers_[1, 1])}
            ]
        },
        'gmm': {
            'means': [
                {'x': float(means[0, 0]), 'y': float(means[0, 1])},
                {'x': float(means[1, 0]), 'y': float(means[1, 1])}
            ],
            'covs': [
                [[float(covs[0, 0, 0]), float(covs[0, 0, 1])], [float(covs[0, 1, 0]), float(covs[0, 1, 1])]],
                [[float(covs[1, 0, 0]), float(covs[1, 0, 1])], [float(covs[1, 1, 0]), float(covs[1, 1, 1])]]
            ]
        }
    }
    
    os.makedirs(os.path.join(BASE_DIR, 'web-vis'), exist_ok=True)
    out_path = os.path.join(BASE_DIR, 'web-vis', 'data.js')
    with open(out_path, 'w') as f:
        f.write("const VIS_DATA = " + json.dumps(data, indent=2) + ";")
    print(f"Data exported to {out_path}")

if __name__ == '__main__':
    main()
