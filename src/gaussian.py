"""
gaussian.py — Multivariate Gaussian probability density function from scratch.

Implements the full multivariate normal distribution PDF:
    N(x|mu, Sigma) = (1 / ((2*pi)^(D/2) * |Sigma|^(1/2))) * 
                     exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))

All components (determinant, inverse, exponent) use NumPy only for
basic linear algebra operations — no scipy or stats libraries.
"""
import numpy as np


def compute_covariance_matrix(X):
    """
    Compute the covariance matrix of dataset X manually.
    
    Formula: Sigma = (1/N) * sum_i (x_i - mean)(x_i - mean)^T
    
    This measures how features co-vary. For the Iris data:
    - Diagonal = individual variances (spread)
    - Off-diagonal = how features like sepal length and width correlate
    
    Args:
        X (numpy.ndarray): Data matrix of shape (N, D).
        
    Returns:
        numpy.ndarray: Covariance matrix of shape (D, D).
    """
    n_samples, n_features = X.shape
    
    # Compute mean of each feature
    mean = np.zeros(n_features)
    for i in range(n_samples):
        mean += X[i]
    mean /= n_samples
    
    # Compute covariance: (1/N) * sum((x_i - mean)(x_i - mean)^T)
    cov = np.zeros((n_features, n_features))
    for i in range(n_samples):
        diff = (X[i] - mean).reshape(-1, 1)  # Column vector (D, 1)
        cov += np.dot(diff, diff.T)           # Outer product (D, D)
    cov /= n_samples
    
    return cov


def multivariate_gaussian_pdf(x, mean, cov):
    """
    Compute the multivariate Gaussian PDF at point x.
    
    N(x|mu, Sigma) = (1 / ((2*pi)^(D/2) * |Sigma|^(1/2))) *
                     exp(-0.5 * (x-mu)^T * Sigma^(-1) * (x-mu))
    
    Components:
    - |Sigma|: determinant of covariance (volume scaling factor)
    - Sigma^(-1): precision matrix (inverse covariance)
    - (x-mu)^T * Sigma^(-1) * (x-mu): Mahalanobis distance squared
    - (2*pi)^(D/2): normalization constant for D dimensions
    
    Args:
        x (numpy.ndarray): Data point of shape (D,).
        mean (numpy.ndarray): Mean vector of shape (D,).
        cov (numpy.ndarray): Covariance matrix of shape (D, D).
        
    Returns:
        float: Probability density at point x.
    """
    d = len(mean)  # Number of dimensions
    
    # Difference vector: (x - mu)
    diff = x - mean
    
    # Determinant of covariance matrix
    det = np.linalg.det(cov)
    
    # Safety check: determinant should be positive
    if det <= 0:
        det = 1e-300  # Prevent log(0) or division by zero
    
    # Inverse of covariance matrix (precision matrix)
    cov_inv = np.linalg.inv(cov)
    
    # Normalization constant: 1 / ((2*pi)^(D/2) * |Sigma|^(1/2))
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
    
    # Exponent: -0.5 * (x-mu)^T * Sigma^(-1) * (x-mu)
    # This is the squared Mahalanobis distance
    mahalanobis_sq = np.dot(diff, np.dot(cov_inv, diff))
    exponent = -0.5 * mahalanobis_sq
    
    # Full PDF value
    pdf_value = norm_const * np.exp(exponent)
    
    return pdf_value


def gaussian_pdf_batch(X, mean, cov):
    """
    Compute multivariate Gaussian PDF for all data points at once.
    
    More efficient than calling multivariate_gaussian_pdf in a loop.
    
    Args:
        X (numpy.ndarray): Data matrix of shape (N, D).
        mean (numpy.ndarray): Mean vector of shape (D,).
        cov (numpy.ndarray): Covariance matrix of shape (D, D).
        
    Returns:
        numpy.ndarray: PDF values of shape (N,).
    """
    n_samples = X.shape[0]
    d = len(mean)
    
    det = np.linalg.det(cov)
    if det <= 0:
        det = 1e-300
    
    cov_inv = np.linalg.inv(cov)
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
    
    # Vectorized Mahalanobis distance for all points
    diff = X - mean  # (N, D)
    # (N, D) @ (D, D) -> (N, D), then element-wise multiply and sum
    mahalanobis_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
    
    pdf_values = norm_const * np.exp(-0.5 * mahalanobis_sq)
    
    return pdf_values
