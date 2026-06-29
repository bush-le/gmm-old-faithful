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
    
    # Read headers
    with open(data_path, 'r') as f:
        headers = f.readline().strip().split(',')
        
    X = np.loadtxt(data_path, delimiter=',', skiprows=1)
    
    # Ground truth labels: 50 Setosa (0), 50 Versicolor (1), 50 Virginica (2)
    y_true = np.concatenate([np.zeros(50), np.ones(50), np.ones(50) * 2]).astype(int)
    
    # 2. Setup Plot
    plt.style.use('default')
    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    
    colors = ['#E74C3C', '#2ECC71', '#3498DB']
    markers = ['o', 's', '^']
    labels = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    
    # 3. Scatter Plot Matrix
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:
                # Histogram for diagonal
                for c in range(3):
                    mask = (y_true == c)
                    ax.hist(X[mask, i], color=colors[c], alpha=0.5, bins=15, edgecolor='black', linewidth=0.5)
            else:
                # Scatter for off-diagonal
                for c in range(3):
                    mask = (y_true == c)
                    ax.scatter(X[mask, j], X[mask, i], 
                               c=colors[c], marker=markers[c], s=40, 
                               edgecolors='black', linewidths=0.5, 
                               alpha=0.8)
            
            if i == n_features - 1:
                ax.set_xlabel(headers[j].replace('_', ' ').title(), fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(headers[i].replace('_', ' ').title(), fontsize=12, fontweight='bold')
            
            ax.grid(True, linestyle='--', alpha=0.3)
                
    fig.suptitle("Iris Dataset - Full Feature Scatter Matrix", fontsize=20, fontweight='bold', y=0.92)
    
    # Add a single legend for the whole figure
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=markers[c], color='w', label=labels[c],
                              markerfacecolor=colors[c], markersize=12, markeredgecolor='black')
                       for c in range(3)]
    fig.legend(handles=legend_elements, loc='upper center', title='Species', fontsize='12', title_fontsize='14', bbox_to_anchor=(0.5, 0.90), ncol=3)
    
    # 5. Save Plot
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iris_raw_labeled.png")
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved plot to: {out_path}")

if __name__ == "__main__":
    main()
