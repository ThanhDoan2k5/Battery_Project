import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# 1. Định nghĩa cấu trúc mạng DNN
class BatteryDNN(nn.Module):
    def __init__(self, input_dim):
        super(BatteryDNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2), 
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

# 2. Nạp mô hình và tên cột
base_dir = os.path.dirname(os.path.abspath(__file__))
rf_model = joblib.load(os.path.join(base_dir, 'master_battery_rf_model.pkl'))
scaler = joblib.load(os.path.join(base_dir, 'master_battery_scaler.pkl'))
dnn_model = BatteryDNN(input_dim=15)
dnn_model.load_state_dict(torch.load(os.path.join(base_dir, 'master_battery_dnn.pth')))
dnn_model.eval()
feature_names = joblib.load(os.path.join(base_dir, 'column_names.pkl'))

# 3. Hàm "phiên dịch" 5 thông số thành 15 đặc trưng của mô hình
def estimate_features(months, cyc, eff, temp, cr):
    # Logic: Ước tính độ chai pin dựa trên chu kỳ và thời gian sử dụng
    estimated_deg = (cyc * 0.03) + (months * 0.5) 
    # Trả về list 15 phần tử khớp thứ tự với feature_names
    # Lưu ý: Nếu thứ tự cột trong 'column_names.pkl' khác, bạn cần map lại cho đúng
    return [4.2, 3.2, 3.7, cr+0.5, 0.1, cr, temp+10, temp, estimated_deg, eff, cyc, 90.0, 25.0, cr, 0.0]

# 4. Giao diện nhập liệu
print("=== CÔNG CỤ DỰ ĐOÁN DUNG LƯỢNG PIN ===")
m = float(input("1. Số tháng đã sử dụng [VD: 12]: "))
c = float(input("2. Tổng số chu kỳ sạc [VD: 350]: "))
e = float(input("3. Hiệu suất sạc ước tính (%) [VD: 90.0]: "))
t = float(input("4. Nhiệt độ trung bình (°C) [VD: 30.0]: "))
a = float(input("5. Dòng sạc trung bình (A) [VD: 2.0]: "))

# 5. Xử lý và Dự đoán
user_inputs = estimate_features(m, c, e, t, a)
new_battery_df = pd.DataFrame([user_inputs], columns=feature_names)
new_battery_scaled = scaler.transform(new_battery_df)

pred_rf = rf_model.predict(new_battery_scaled)
pred_dnn = dnn_model(torch.tensor(new_battery_scaled.astype(np.float32)))

print("\n🚀 KẾT QUẢ DỰ ĐOÁN DUNG LƯỢNG:")
print(f"-> Theo mô hình Random Forest: {pred_rf[0]:.3f} Ah")
print(f"-> Theo mô hình Mạng Nơ-ron (DNN): {pred_dnn.item():.3f} Ah")