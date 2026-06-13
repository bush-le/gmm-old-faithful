import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc của project vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.em import fit_gmm
from src.kmeans import fit_kmeans

def main():
    # 1. Tạo dữ liệu giả lập (Synthetic Data) với 2 cụm hình elip chồng lấp
    np.random.seed(42)
    # Cụm 0: Hình elip kéo dài
    mean0 = np.array([-1.0, -1.0])
    cov0 = np.array([[6.0, 4.5], [4.5, 5.0]])
    X0 = np.random.multivariate_normal(mean0, cov0, 350)
    
    # Cụm 1: Hình elip kéo dài hướng khác
    mean1 = np.array([4.0, -2.0])
    cov1 = np.array([[2.5, -1.5], [-1.5, 3.0]])
    X1 = np.random.multivariate_normal(mean1, cov1, 350)
    
    X = np.vstack([X0, X1])
    # Tạo nhãn gốc (Ground Truth)
    y_true = np.concatenate([np.zeros(350), np.ones(350)]).astype(int)

    # 2. Huấn luyện thuật toán K-Means
    km_labels, km_centroids = fit_kmeans(X, K=2, max_iters=100, seed=42)
    
    # 3. Huấn luyện thuật toán GMM
    gmm_params, gmm_resp, _, _ = fit_gmm(X, K=2, max_iters=100, tol=1e-6, reg_covar=1e-6, init_method="kmeans", seed=42)
    gmm_labels = np.argmax(gmm_resp, axis=1)

    # 4. Căn chỉnh nhãn (Bởi vì thuật toán không giám sát có thể bị đảo ngược nhãn ngẫu nhiên)
    if np.mean(km_labels == y_true) < 0.5:
        km_labels = 1 - km_labels
        
    if np.mean(gmm_labels == y_true) < 0.5:
        gmm_labels = 1 - gmm_labels

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
    ax.set_ylabel('Độ chính xác / Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('So sánh độ chính xác phân loại:\nK-Means vs Gaussian Mixture Model (GMM)', fontsize=14, fontweight='bold', pad=20)
    
    # Đính kèm phần trăm ngay trên từng cột
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # Chú thích giải thích nguyên nhân
    caption = "Dữ liệu được tạo mô phỏng phân bố Gaussian đa biến (hình elip) chồng lấp.\nGMM đạt độ chính xác cao hơn hẳn nhờ mô hình hóa được ma trận hiệp phương sai (covariance)\ntrong khi K-Means chỉ dùng khoảng cách cứng (Euclidean)."
    plt.figtext(0.5, -0.06, caption, ha="center", fontsize=10, style='italic',
                bbox=dict(facecolor='#F8F9F9', edgecolor='#BDC3C7', boxstyle='round,pad=0.8'))

    # 7. Xuất và lưu hình ảnh biểu đồ
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accuracy_comparison.png")
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Đã lưu biểu đồ thành công tại: {out_path}")
    print(f"Độ chính xác của K-Means: {km_accuracy:.2f}%")
    print(f"Độ chính xác của GMM:     {gmm_accuracy:.2f}%")

if __name__ == "__main__":
    main()