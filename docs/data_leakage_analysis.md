# 📊 Báo Cáo Phân Tích Rò Rỉ Dữ Liệu (Data Leakage Analysis)

Báo cáo này đánh giá quy trình tiền xử lý, chuẩn hóa (standardization) và khả năng phân tách dữ liệu (train/validation/test split) trong dự án Gaussian Mixture Model (GMM) Old Faithful hiện tại nhằm phát hiện và ngăn chặn hiện tượng **Rò rỉ dữ liệu (Data Leakage)**.

---

## 🔍 1. Hiện Trạng Quy Trình Hiện Tại (Current Pipeline State)

Trong mã nguồn hiện tại của [main.py](file:///home/bush/Desktop/gmm-old-faithful/main.py) và [src/preprocessing.py](file:///home/bush/Desktop/gmm-old-faithful/src/preprocessing.py):
1. **Không thực hiện phân tách dữ liệu (No Train/Test Split)**: Mô hình GMM được huấn luyện trực tiếp trên toàn bộ 272 mẫu dữ liệu đã qua chuẩn hóa.
2. **Chuẩn hóa toàn bộ (Global Standardization)**: Toàn bộ dữ liệu thô được đưa vào hàm `standardize()` để tính toán trung bình ($\mu$) và độ lệch chuẩn ($\sigma$), tạo ra file [faithful_clean.csv](file:///home/bush/Desktop/gmm-old-faithful/data/processed/faithful_clean.csv).

### ⚠️ Rủi ro Data Leakage nếu phân tách dữ liệu sau này:
Nếu bạn lấy file [faithful_clean.csv](file:///home/bush/Desktop/gmm-old-faithful/data/processed/faithful_clean.csv) (hoặc mảng `X_standardized` trong bộ nhớ) rồi chia thành các tập **Train / Validation / Test**, bạn sẽ gặp lỗi **Data Leakage (Rò rỉ thông tin phân phối)** rất nghiêm trọng. 

Lý do là vì các thông số chuẩn hóa ($\mu$ và $\sigma$) đã được tính toán trên **tất cả các dòng dữ liệu** (bao gồm cả dữ liệu mà lẽ ra mô hình không được biết ở tập validation/test).

---

## 🛑 2. Chi Tiết Lỗi Rò Rỉ Dữ Liệu Trong Mã Nguồn

### Lỗi 1: Tính toán Z-score trên toàn bộ dữ liệu trước khi chia
Tại [src/preprocessing.py:L286-L343](file:///home/bush/Desktop/gmm-old-faithful/src/preprocessing.py#L286-L343):

```python
def standardize(data, logs_dir=None):
    mean = compute_mean(data)          # Lấy trung bình của TOÀN BỘ tập dữ liệu
    std = compute_std(data, mean)      # Lấy độ lệch chuẩn của TOÀN BỘ tập dữ liệu
    
    standardized = (data - mean) / std # Áp dụng chuẩn hóa toàn cục
    return standardized, mean, std
```

**Hậu quả:** 
* Tập huấn luyện (Train set) sẽ mang thông tin gián tiếp về phân phối (trung bình và phương sai) của tập kiểm thử (Test set).
* Khi đánh giá mô hình trên tập Test, kết quả metric (Log-Likelihood, Silhouette, v.v.) sẽ bị thổi phồng, không phản ánh chính xác khả năng tổng quát hóa thực tế của mô hình trên dữ liệu mới hoàn toàn.
* Vi phạm trực tiếp quy tắc trong tài liệu tham khảo [ML_PIPELINE_REFERENCE.md:L524](file:///home/bush/Desktop/gmm-old-faithful/docs/ML_PIPELINE_REFERENCE.md#L524):
  > **AI Agent Rule:** Always fit the scaler **only** on training data. Apply (transform only) to validation and test data. Never fit on the full dataset — that causes data leakage (Step 10).

---

### Lỗi 2: Phát hiện ngoại lai (Outlier Detection) toàn cục
Tại [src/preprocessing.py:L108-L150](file:///home/bush/Desktop/gmm-old-faithful/src/preprocessing.py#L108-L150):
* Phép tính IQR (Q1, Q3) được thực hiện trên toàn bộ dữ liệu thô. 
* Mặc dù trong dự án này, quyết định cuối cùng là **giữ lại toàn bộ ngoại lai (Treatment: KEEP)**, nhưng nếu trong tương lai bạn chuyển sang phương án loại bỏ ngoại lai hoặc gán giá trị biên (Capping/Winsorization), việc tính toán ngưỡng IQR trên toàn bộ dữ liệu cũng sẽ gây rò rỉ dữ liệu. Ngưỡng ngoại lai phải được xác định dựa trên tập Train và áp dụng lên tập Test.

---

## 📊 3. Sơ Đồ Quy Trình: Sai (Có Leak) vs Đúng (Không Leak)

```mermaid
graph TD
    subgraph Quy trình SAI (Có Data Leakage)
        A1[Dữ liệu thô] --> B1(Tính Mean/Std toàn cục)
        B1 --> C1(Chuẩn hóa toàn bộ dữ liệu)
        C1 --> D1[Chia Train / Test Split]
        D1 --> E1[Huấn luyện trên Train]
        D1 --> F1[Đánh giá trên Test - Bị Leak!]
    end

    subgraph Quy trình ĐÚNG (Ngăn chặn Leakage)
        A2[Dữ liệu thô] --> B2[Chia Train / Test Split]
        B2 --> C2_train[Tập Train]
        B2 --> C2_test[Tập Test]
        C2_train --> D2(Tính Mean/Std trên Train ONLY)
        D2 --> E2(Chuẩn hóa tập Train bằng Mean/Std của Train)
        D2 --> F2(Chuẩn hóa tập Test bằng Mean/Std của Train)
        E2 --> G2[Huấn luyện mô hình]
        F2 --> H2[Đánh giá mô hình - Khách quan!]
    end
```

---

## 🛠️ 4. Giải Pháp Khắc Phục Bằng Pure NumPy (Không Dùng Sklearn)

Để khắc phục triệt để lỗi này khi thực hiện Train/Test split, bạn cần viết lại luồng tiền xử lý. Dưới đây là cách triển khai chuẩn bằng **Pure NumPy** tuân thủ nghiêm ngặt các ràng buộc của dự án.

### Bước 1: Hàm phân tách dữ liệu không thiên vị (Train/Test Split)
Thêm hàm chia dữ liệu ngẫu nhiên có kiểm soát hạt giống (random seed) trong [src/preprocessing.py](file:///home/bush/Desktop/gmm-old-faithful/src/preprocessing.py):

```python
def train_test_split_numpy(X, test_size=0.2, seed=42):
    """
    Chia tập dữ liệu thành Train và Test bằng NumPy thuần.
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    
    test_set_size = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return X[train_indices], X[test_indices]
```

### Bước 2: Chuẩn hóa đúng cách (Fit trên Train, Transform trên Train/Test)
Tách biệt hàm tính toán tham số chuẩn hóa (`fit`) và hàm áp dụng chuẩn hóa (`transform`):

```python
def fit_standard_scaler(X_train):
    """
    Chỉ tính toán Mean và Std từ tập Train.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # Tránh chia cho 0 nếu std = 0
    std = np.where(std == 0, 1.0, std)
    return mean, std

def transform_standard_scaler(X, mean, std):
    """
    Áp dụng thông số chuẩn hóa lên bất kỳ tập dữ liệu nào (Train, Val, Test).
    """
    return (X - mean) / std
```

### Bước 3: Áp dụng vào luồng chạy chính ([main.py](file:///home/bush/Desktop/gmm-old-faithful/main.py))
Thay đổi thứ tự thực hiện: **Split trước -> Scale sau**:

```python
# 1. Load và làm sạch dữ liệu thô
raw_data = load_csv(RAW_DATA_PATH)
X_raw = np.array([[r[0], r[1]] for r in raw_data if r[0] > 0 and r[1] > 0])

# 2. Chia tập Train và Test TRƯỚC khi chuẩn hóa
X_train_raw, X_test_raw = train_test_split_numpy(X_raw, test_size=0.2, seed=RANDOM_SEED)

# 3. Tính toán tham số chuẩn hóa CHỈ trên tập Train
mean_train, std_train = fit_standard_scaler(X_train_raw)

# 4. Chuẩn hóa độc lập tập Train và Test
X_train_scaled = transform_standard_scaler(X_train_raw, mean_train, std_train)
X_test_scaled = transform_standard_scaler(X_test_raw, mean_train, std_train)

# 5. Huấn luyện GMM trên X_train_scaled
params, responsibilities, log_likelihoods, n_iters = fit_gmm(
    X_train_scaled, K, MAX_ITERS, TOL, REG_COVAR,
    init_method=INIT_METHOD, seed=RANDOM_SEED
)

# 6. Đánh giá khách quan trên X_test_scaled sử dụng tham số đã học
from src.em import compute_log_likelihood
test_ll = compute_log_likelihood(X_test_scaled, params)
print(f"Log-Likelihood trên tập kiểm thử độc lập (Test Set LL): {test_ll:.6f}")
```

---

## 💡 5. Đánh Giá Đối Với Bài Toán Unsupervised GMM hiện tại

* **Nếu chỉ phân cụm mô tả (Descriptive Clustering)**: Đối với tập dữ liệu Old Faithful kinh điển, nếu mục tiêu của bạn chỉ là tìm ra sự phân bổ tự nhiên của 2 cụm (bimodal structure) trên chính tập dữ liệu đó mà không cần dự báo cho dữ liệu tương lai, việc sử dụng toàn bộ dữ liệu để học tham số chuẩn hóa và huấn luyện là chấp nhận được.
* **Nếu muốn kiểm thử khả năng tổng quát (Generative Generalization)**: Nếu mục tiêu là đánh giá xem mô hình GMM học từ Old Faithful có tổng quát tốt cho các vụ phun trào mới hay không, việc chia tập dữ liệu theo cách sửa đổi ở phần 4 là **bắt buộc** để tránh hiện tượng lạc quan thái quá do rò rỉ dữ liệu.
* **Đối với mô hình MLP (Supervised)**: Nếu bạn refactor dự án để chạy cả MLP (như mô tả trong [REFACTOR_PROMPTS.md](file:///home/bush/Desktop/gmm-old-faithful/docs/REFACTOR_PROMPTS.md)), việc sửa quy trình chuẩn hóa theo giải pháp ở phần 4 là **tối quan trọng** để mô hình không bị mất tác dụng thực tế khi triển khai.
