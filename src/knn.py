import numpy as np

def fit_predict_knn(X_train, y_train, X_test, k=3):
    """
    Manual implementation of K-Nearest Neighbors classifier.
    """
    import numpy as np
    
    y_pred = np.zeros(X_test.shape[0], dtype=int)
    for i, test_point in enumerate(X_test):
        # Compute Euclidean distance from test_point to all training points
        distances = np.linalg.norm(X_train - test_point, axis=1)
        
        # Get indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        # Get the labels of the nearest neighbors
        nearest_labels = y_train[nearest_indices]
        
        # Majority voting
        counts = np.bincount(nearest_labels)
        y_pred[i] = np.argmax(counts)
        
    return y_pred
