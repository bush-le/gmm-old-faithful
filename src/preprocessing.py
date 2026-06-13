"""
preprocessing.py — Manual data cleaning and standardization pipeline.

Implements z-score normalization from scratch: x' = (x - mean) / std
No pandas, no sklearn preprocessing — pure Python + NumPy arrays.
"""
import numpy as np
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_loader import load_csv


def clean_data(raw_data):
    """
    Remove invalid rows (NaN, negative, or zero values).
    
    Iris data should have all positive features.
    
    Args:
        raw_data (list): List of feature rows.
        
    Returns:
        numpy.ndarray: Cleaned data array of shape (N, D).
    """
    cleaned = []
    removed_count = 0
    
    for row in raw_data:
        # Check for valid numeric values
        if all(val > 0 for val in row):
            cleaned.append(row)
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"  Removed {removed_count} invalid rows")
    
    print(f"  Clean data: {len(cleaned)} rows")
    return np.array(cleaned)


def compute_mean(data):
    """
    Compute column-wise mean manually.
    """
    n_samples = data.shape[0]
    mean = np.zeros(data.shape[1])
    for i in range(n_samples):
        mean += data[i]
    return mean / n_samples


def compute_std(data, mean):
    """
    Compute column-wise standard deviation manually.
    """
    n_samples = data.shape[0]
    variance = np.zeros(data.shape[1])
    for i in range(n_samples):
        diff = data[i] - mean
        variance += diff * diff
    variance = variance / n_samples
    return np.sqrt(variance)


def standardize(data):
    """
    Apply z-score standardization: x' = (x - mean) / std
    """
    mean = compute_mean(data)
    std = compute_std(data, mean)
    
    print(f"  Feature means: " + ", ".join([f"f{i}={mean[i]:.4f}" for i in range(len(mean))]))
    print(f"  Feature stds:  " + ", ".join([f"f{i}={std[i]:.4f}" for i in range(len(std))]))
    
    standardized = (data - mean) / std
    
    return standardized, mean, std


def save_csv(data, filepath, header="sepal_length,sepal_width,petal_length,petal_width"):
    """
    Save numpy array to CSV file using manual file I/O.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for row in data:
            f.write(",".join([f"{val:.6f}" for val in row]) + "\n")
    
    print(f"  Saved {len(data)} rows to {filepath}")


def run_pipeline():
    """
    Execute the full preprocessing pipeline:
    1. Load raw CSV
    2. Clean invalid rows
    3. Standardize features (z-score)
    4. Save processed data
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    print("\n[Step 1] Loading raw data...")
    raw_data = load_csv(RAW_DATA_PATH)
    
    print("\n[Step 2] Cleaning data...")
    clean = clean_data(raw_data)
    
    print("\n[Step 3] Standardizing features (z-score)...")
    standardized, mean, std = standardize(clean)
    
    print("\n[Step 4] Saving processed data...")
    save_csv(standardized, PROCESSED_DATA_PATH)
    
    print("\n[DONE] Preprocessing complete.")
    return standardized, mean, std


if __name__ == "__main__":
    run_pipeline()
