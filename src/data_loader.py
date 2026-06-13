"""
data_loader.py — Manual CSV file parser using Python file I/O.

Loads the Iris dataset from CSV format without any external libraries.
Parses each line manually, handling edge cases (empty lines, invalid values).
"""


def load_csv(filepath):
    """
    Load a CSV file and return data as a list of floats.
    
    Parses the CSV manually using open() and string splitting.
    Skips the header row and any rows with invalid/missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        list: List of features [sepal_length, sepal_width, petal_length, petal_width] as floats.
    """
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Skip header (first line)
    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        parts = line.split(',')
        
        # Expect 4 columns for Iris
        if len(parts) < 4:
            print(f"  [SKIP] Row {i}: expected at least 4 columns, got {len(parts)}")
            continue
        
        try:
            row_data = [float(p) for p in parts[:4]]
        except ValueError:
            print(f"  [SKIP] Row {i}: non-numeric value '{parts}'")
            continue
        
        data.append(row_data)
    
    print(f"  Loaded {len(data)} valid rows from {filepath}")
    return data
