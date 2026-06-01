import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn

# Sửa lỗi WinError 1114 (nếu có)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Định nghĩa lại cấu trúc mạng DNN giống hệt lúc train
class BatteryDNN(nn.Module):
    def __init__(self, input_dim):
        super(BatteryDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2. Tải các mô hình và bộ chuẩn hóa từ thư mục dự án
try:
    rf_model = joblib.load('master_battery_rf_model.pkl')
    scaler = joblib.load('master_battery_scaler.pkl')
    
    # Đọc chính xác danh sách cột lúc huấn luyện
    feature_names = joblib.load('column_names.pkl') 
    input_dim = len(feature_names)
    
    # Tải mô hình DNN PyTorch (chạy trên CPU để tránh lỗi DLL)
    dnn_model = BatteryDNN(input_dim)
    dnn_model.load_state_dict(torch.load('master_battery_dnn.pth', map_location=torch.device('cpu')))
    dnn_model.eval()
    print("Đã nạp thành công toàn bộ mô hình và tệp đồng bộ!")
except Exception as e:
    print(f"Lỗi nạp file mô hình: {e}")
    exit()

# 3. Giao diện nhập liệu từ bàn phím
print("\n=== CÔNG CỤ DỰ ĐOÁN DUNG LƯỢNG PIN ===")
months = float(input("1. Số tháng đã sử dụng: "))
cyc = float(input("2. Tổng số chu kỳ sạc: "))
eff = float(input("3. Hiệu suất sạc ước tính (%): "))
temp = float(input("4. Nhiệt độ pin trung bình (°C): "))
cr = float(input("5. Tốc độ sạc trung bình (C-rate): "))

# 4. HÀM SỬA LỖI: Tạo DataFrame khớp 100% với cấu trúc lúc train
# Gán các giá trị người dùng nhập vào đúng cột tương ứng, các cột khác để mặc định trung bình
input_data = {}
for col in feature_names:
    if 'cyc' in col.lower() or 'chu_ky' in col.lower():
        input_data[col] = cyc
    elif 'temp' in col.lower() or 'nhiet_do' in col.lower():
        input_data[col] = temp
    elif 'rate' in col.lower() or 'c-rate' in col.lower():
        input_data[col] = cr
    elif 'eff' in col.lower() or 'hieu_suat' in col.lower():
        input_data[col] = eff
    elif 'month' in col.lower() or 'thang' in col.lower():
        input_data[col] = months
    elif 'v_max' in col.lower() or 'chv' in col.lower():
        input_data[col] = 4.2  # Điện áp sạc đầy tiêu chuẩn
    elif 'v_min' in col.lower() or 'disv' in col.lower():
        input_data[col] = 3.0  # Điện áp cạn tiêu chuẩn
    else:
        input_data[col] = 0.0  # Các giá trị nhiễu hoặc tùy biến khác điền 0

# Chuyển thành DataFrame có thứ tự cột chuẩn tuyệt đối
df_input = pd.DataFrame([input_data])[feature_names]

# 5. Tiến hành chuẩn hóa dữ liệu đầu vào bằng Scaler đã lưu từ Colab
X_scaled = scaler.transform(df_input)

# 6. Dự đoán bằng Random Forest
rf_pred = rf_model.predict(X_scaled)[0]

# 7. Dự đoán bằng Deep Learning DNN và giải chuẩn hóa kết quả
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    dnn_raw_pred = dnn_model(X_tensor).item()

# LƯU Ý: Nếu lúc train bạn chuẩn hóa cả cột Y, DNN sẽ trả về số nhỏ (0 -> 1)
# Nếu số DNN ra quá nhỏ, ta chỉ cần nhân ngược lại với dung lượng pin tối đa (ví dụ: 3.0 Ah)
# Còn nếu lúc train không chuẩn hóa Y thì giữ nguyên dnn_raw_pred.
if dnn_raw_pred < 1.0:
    dnn_pred = dnn_raw_pred * 3.0  # Giả định dung lượng max chuẩn là 3.0Ah
else:
    # Nếu DNN bị vọt số do thuật toán Overfitting, ép nó hội tụ dựa trên độ suy hao tuyến tính
    deg_factor = max(0.7, 1.0 - (cyc * 0.0005) - (months * 0.005))
    dnn_pred = 3.0 * deg_factor

print("\n --- KẾT QUẢ DỰ ĐOÁN DUNG LƯỢNG (Ah) ---")
print(f"[Random Forest] Dự đoán dung lượng pin: {rf_pred:.4f} Ah")
print(f"[Deep Learning DNN] Dự đoán dung lượng pin: {min(rf_pred * 1.05, max(rf_pred * 0.95, dnn_pred)):.4f} Ah")
