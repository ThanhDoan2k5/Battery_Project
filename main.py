import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class BatteryDNN(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

try:
    rf = joblib.load('master_battery_rf_model.pkl')
    scaler = joblib.load('master_battery_scaler.pkl')
    model = BatteryDNN(input_dim=14)
    model.load_state_dict(torch.load('master_battery_dnn.pth'))
    model.eval()
except Exception as e:
    print(f"Lỗi nạp mô hình: {e}")
    exit()

def predict():
    print("\n--- DỰ BÁO DUNG LƯỢNG & TUỔI THỌ PIN (mAh) ---")
    try:
        cap_goc = float(input("1. Dung lượng gốc (mAh): "))
        thang_sd = float(input("2. Số tháng đã sử dụng: "))
        chu_ky = float(input("3. Số chu kỳ sạc: "))
        nhiet_do = float(input("4. Nhiệt độ trung bình (°C): "))

        # 1. TÍNH DUNG LƯỢNG (Khóa chặt bằng công thức Vật lý, không để AI cào bằng)
        # Giảm 1.5% mỗi 100 chu kỳ + 1.5% mỗi 10 tháng. Nhiệt độ > 25°C phạt thêm 3% mỗi độ.
        cycle_drop = 0.00015 * chu_ky
        time_drop = 0.0015 * thang_sd
        temp_penalty = 1.0 + max(0, nhiet_do - 25) * 0.03
        
        total_drop = (cycle_drop + time_drop) * temp_penalty
        
        # Kết quả dung lượng (Chặn tối đa hỏng 80%)
        ket_qua = cap_goc * (1 - min(total_drop, 0.8))

        # 2. TÍNH RUL (CHU KỲ CÒN LẠI) CHUẨN XÁC
        eol_cap = cap_goc * 0.7 # Hỏng ở 70%
        
        if ket_qua > eol_cap:
            # Tốc độ hao mòn = (Dung lượng đã mất) / (Số chu kỳ)
            loss_per_cycle = (cap_goc - ket_qua) / max(chu_ky, 1)
            # Chặn lỗi chia cho số 0 với pin quá mới
            loss_per_cycle = max(loss_per_cycle, cap_goc * 0.0001) 
            
            rul = (ket_qua - eol_cap) / loss_per_cycle
        else:
            rul = 0

        print(f"\n✅ DUNG LƯỢNG CÒN LẠI: {int(ket_qua)} mAh")
        print(f"✅ SỐ CHU KỲ CÒN DÙNG ĐƯỢC: {int(rul)} chu kỳ")
        print("==========================================")
        
    except ValueError:
        print("❌ Lỗi: Vui lòng nhập số hợp lệ!")

if __name__ == "__main__":
    predict()