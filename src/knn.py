"""
knn.py — K-Nearest Neighbors classifier from scratch.

Since we are treating the Iris dataset as unsupervised (ignoring ground-truth labels),
we use KNN to test clustering stability/consistency.:
1. Generate pseudo-labels from KMeans
2. For each point, predict its label using KNN (leave-one-out)
3. Measure agreement between KNN predictions and KMeans labels

High consistency = clusters are well-separated and locally coherent.
KNN is an instance-based (lazy) learner vs GMM's generative approach.
"""
import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two points.
    
    Args:
        a (numpy.ndarray): First point of shape (D,).
        b (numpy.ndarray): Second point of shape (D,).
        
    Returns:
        float: Euclidean distance.
    """
    diff = a - b
    return np.sqrt(np.sum(diff * diff))


def predict_single(X_train, y_train, x_query, k):
    """
    Predict label for a single query point using KNN.
    
    Algorithm:
    1. Compute distance from query to all training points
    2. Find K nearest neighbors
    3. Majority vote among neighbors' labels
    
    Args:
        X_train (numpy.ndarray): Training data of shape (N, D).
        y_train (numpy.ndarray): Training labels of shape (N,).
        x_query (numpy.ndarray): Query point of shape (D,).
        k (int): Number of neighbors.
        
    Returns:
        int: Predicted label.
    """
    n_train = X_train.shape[0]
    
    # Compute distances to all training points
    distances = np.zeros(n_train)
    for i in range(n_train):
        distances[i] = euclidean_distance(X_train[i], x_query)
    
    # Find K nearest neighbors (indices of K smallest distances)
    neighbor_indices = np.argsort(distances)[:k]
    neighbor_labels = y_train[neighbor_indices]
    
    # Majority vote
    unique_labels = np.unique(neighbor_labels)
    best_label = unique_labels[0]
    best_count = 0
    
    for label in unique_labels:
        count = np.sum(neighbor_labels == label)
        if count > best_count:
            best_count = count
            best_label = label
    
    return best_label


def predict_knn(X_train, y_train, X_test, k):
    """
    Predict labels for multiple query points.
    
    Args:
        X_train (numpy.ndarray): Training data of shape (N_train, D).
        y_train (numpy.ndarray): Training labels of shape (N_train,).
        X_test (numpy.ndarray): Test data of shape (N_test, D).
        k (int): Number of neighbors.
        
    Returns:
        numpy.ndarray: Predicted labels of shape (N_test,).
    """
    n_test = X_test.shape[0]
    predictions = np.zeros(n_test, dtype=int)
    
    for i in range(n_test):
        predictions[i] = predict_single(X_train, y_train, X_test[i], k)
    
    return predictions


def evaluate_consistency(X, kmeans_labels, k):
    """
    Evaluate clustering consistency using leave-one-out KNN.
    
    For each point:
    1. Remove it from the dataset
    2. Predict its label using remaining points + KNN
    3. Check if prediction matches KMeans label
    
    High accuracy = clusters are locally consistent and well-separated.
    
    Args:
        X (numpy.ndarray): Data of shape (N, D).
        kmeans_labels (numpy.ndarray): Labels from KMeans of shape (N,).
        k (int): Number of neighbors for KNN.
        
    Returns:
        float: Consistency score (fraction of agreements).
    """
    n_samples = X.shape[0]
    correct = 0
    
    for i in range(n_samples):
        # Leave-one-out: exclude point i
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        
        X_train = X[mask]
        y_train = kmeans_labels[mask]
        
        # Predict label for excluded point
        predicted = predict_single(X_train, y_train, X[i], k)
        
        if predicted == kmeans_labels[i]:
            correct += 1
    
    consistency = correct / n_samples
    print(f"    KNN consistency (k={k}): {consistency:.4f} "
          f"({correct}/{n_samples} agreements)")
    
    return consistency
