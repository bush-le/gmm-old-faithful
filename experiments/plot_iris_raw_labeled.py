import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # 1. Load Iris Data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "iris_clean.csv")
    X = np.loadtxt(data_path, delimiter=',', skiprows=1)
    
    # Use first two features: Sepal Length vs Sepal Width
    X_2d = X[:, :2]
    
    # Ground truth labels: 50 Setosa (0), 50 Versicolor (1), 50 Virginica (2)
    y_true = np.concatenate([np.zeros(50), np.ones(50), np.ones(50) * 2]).astype(int)
    
    # 2. Setup Plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#E74C3C', '#2ECC71', '#3498DB']
    markers = ['o', 's', '^']
    labels = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    
    # 3. Scatter Plot
    for i in range(3):
        mask = (y_true == i)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=colors[i], marker=markers[i], s=80, 
                   edgecolors='black', linewidths=1.0, 
                   alpha=0.8, label=labels[i])
    
    # 4. Styling
    ax.set_title("Iris Dataset - Ground Truth Labels\n(Sepal Length vs Sepal Width)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Sepal Length (cm)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sepal Width (cm)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Species', title_fontsize='12', fontsize='11', loc='upper right', framealpha=0.9)
    
    # 5. Save Plot
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iris_raw_labeled.png")
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved plot to: {out_path}")

if __name__ == "__main__":
    main()