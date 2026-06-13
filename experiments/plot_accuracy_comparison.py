import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

# Thêm đường dẫn gốc của project vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.em import fit_gmm
from src.kmeans import fit_kmeans

def main():
    # 1. Tải dữ liệu Iris
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "iris_clean.csv")
    X = np.loadtxt(data_path, delimiter=',', skiprows=1)
    
    # Tạo nhãn gốc (Ground Truth) cho 150 mẫu (50 Setosa, 50 Versicolor, 50 Virginica)
    y_true = np.concatenate([np.zeros(50), np.ones(50), np.ones(50) * 2]).astype(int)

    # 2. Huấn luyện thuật toán K-Means
    km_labels, km_centroids = fit_kmeans(X, K=3, max_iters=100, seed=42)
    
    # 3. Huấn luyện thuật toán GMM
    gmm_params, gmm_resp, _, _ = fit_gmm(X, K=3, max_iters=100, tol=1e-6, reg_covar=1e-6, init_method="kmeans", seed=42)
    gmm_labels = np.argmax(gmm_resp, axis=1)

    # 4. Căn chỉnh nhãn (Bởi vì thuật toán không giám sát có thể bị đảo ngược nhãn ngẫu nhiên)
    def align_labels(labels, y_true):
        best_acc = 0
        best_labels = labels
        for perm in itertools.permutations([0, 1, 2]):
            mapped_labels = np.zeros_like(labels)
            for i, p in enumerate(perm):
                mapped_labels[labels == i] = p
            acc = np.mean(mapped_labels == y_true)
            if acc > best_acc:
                best_acc = acc
                best_labels = mapped_labels
        return best_labels

    km_labels = align_labels(km_labels, y_true)
    gmm_labels = align_labels(gmm_labels, y_true)

    # 5. Tính toán tỉ lệ chính xác (Accuracy %)
    km_accuracy = np.mean(km_labels == y_true) * 100
    gmm_accuracy = np.mean(gmm_labels == y_true) * 100

    # 6. Trực quan hoá biểu đồ (Bar Chart)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['K-Means', 'GMM']
    accuracies = [km_accuracy, gmm_accuracy]
    colors = ['#E74C3C', '#27AE60']  # Đỏ cho K-Means, Xanh cho GMM
    
    bars = ax.bar(models, accuracies, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    
    # Cấu hình hiển thị trục và tiêu đề
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy Comparison:\nK-Means vs Gaussian Mixture Model (GMM)', fontsize=14, fontweight='bold', pad=20)
    
    # Đính kèm phần trăm ngay trên từng cột
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # Chú thích giải thích nguyên nhân
    caption = "Performance on the Iris dataset (4 features, 3 classes).\nGMM provides a more flexible probabilistic model compared to K-Means,\nbetter capturing the overlapping clusters and varying densities of the classes."
    plt.figtext(0.5, -0.06, caption, ha="center", fontsize=10, style='italic',
                bbox=dict(facecolor='#F8F9F9', edgecolor='#BDC3C7', boxstyle='round,pad=0.8'))

    # 7. Xuất và lưu hình ảnh biểu đồ
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accuracy_comparison.png")
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved chart to: {out_path}")
    print(f"K-Means Accuracy: {km_accuracy:.2f}%")
    print(f"GMM Accuracy:     {gmm_accuracy:.2f}%")

if __name__ == "__main__":
    main()