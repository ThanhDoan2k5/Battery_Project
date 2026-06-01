import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# =======================================================
# 1. ĐỊNH NGHĨA CẤU TRÚC MẠNG DNN (Chuẩn xác 100% theo file .pth)
# =======================================================
class BatteryDNN(nn.Module):
    def __init__(self, input_dim):
        super(BatteryDNN, self).__init__()
        # Cấu trúc chuẩn: Input -> 64 -> 32 -> 1 (Khớp hoàn toàn với checkpoint)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Tầng cuối cùng kết nối trực tiếp từ 32 ra 1
        )
    def forward(self, x):
        return self.net(x)

# =======================================================
# 2. NẠP MÔ HÌNH VÀ ÉP CHẠY BẰNG CPU (Chống lỗi c10.dll)
# =======================================================
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."

print("Đang nạp các bộ não mô hình...")
rf_model = joblib.load(os.path.join(base_dir, 'master_battery_rf_model.pkl'))
scaler = joblib.load(os.path.join(base_dir, 'master_battery_scaler.pkl'))
feature_names = joblib.load(os.path.join(base_dir, 'column_names.pkl'))

# Khởi tạo mô hình mạng và nạp trọng số, ép map sang CPU để bypass WinError 1114
dnn_model = BatteryDNN(input_dim=len(feature_names))
dnn_model.load_state_dict(torch.load(os.path.join(base_dir, 'master_battery_dnn.pth'), map_location=torch.device('cpu')))
dnn_model.eval()

print("Đã nạp xong tất cả mô hình thực tế!")

# =======================================================
# 3. HÀM DỊCH THÔNG SỐ ĐẦU VÀO
# =======================================================
def estimate_features(months, cyc, eff, temp, cr):
    estimated_deg = (cyc * 0.03) + (months * 0.5)
    return [4.2, 3.2, 3.7, cr+0.5, 0.1, cr, temp+10, temp, estimated_deg, eff, cyc, 90.0, 25.0, cr, 0.0]

# =======================================================
# 4. GIAO DIỆN NHẬP LIỆU VÀ DỰ ĐOÁN
# =======================================================
print("\n=== CÔNG CỤ DỰ ĐOÁN DUNG LƯỢNG PIN ===")
m = float(input("1. Số tháng đã sử dụng [VD: 12]: "))
c = float(input("2. Tổng số chu kỳ sạc [VD: 350]: "))
e = float(input("3. Hiệu suất sạc ước tính (%) [VD: 90.0]: "))
t = float(input("4. Nhiệt độ pin trung bình (°C) [VD: 35.0]: "))
cr = float(input("5. Tốc độ sạc trung bình (C-rate) [VD: 1.0]: "))

raw_features = estimate_features(m, c, e, t, cr)
df_input = pd.DataFrame([raw_features], columns=feature_names)
scaled_input = scaler.transform(df_input)

print("\n --- KẾT QUẢ DỰ ĐOÁN DUNG LƯỢNG (Ah) ---")

# 4.1 Dự đoán bằng Random Forest
rf_pred = rf_model.predict(scaled_input)
print(f"[Random Forest] Dự đoán dung lượng pin: {rf_pred[0]:.4f} Ah")

# 4.2 Dự đoán bằng Deep Learning DNN
with torch.no_grad():
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
    dnn_pred = dnn_model(tensor_input).numpy()
print(f"[Deep Learning DNN] Dự đoán dung lượng pin: {dnn_pred[0][0]:.4f} Ah")
print("==========================================")
