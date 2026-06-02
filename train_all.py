import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Cấu trúc mạng khớp với file .pth của bạn (64-32-1)
class BatteryDNN(nn.Module):
    def __init__(self, input_dim):
        super(BatteryDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def run_training():
    df = pd.read_csv('Master_Battery_Dataset.csv')
    features = ['V_max', 'V_min', 'V_mean', 'C_rate_max', 'C_rate_min', 'C_rate_mean', 
                'Temp_max', 'Temp_mean', 'chI', 'disI', 'Current (A)', 'Ambient Temp (°C)', 
                'Efficiency (%)', 'Optimal Charging Duration Class']
    
    X = df[features].fillna(0)
    y = df['Target_Capacity_Ah']
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # Lưu scaler và model
    joblib.dump(scaler, 'master_battery_scaler.pkl')
    
    rf = RandomForestRegressor(n_estimators=100).fit(X_scaled, y)
    joblib.dump(rf, 'master_battery_rf_model.pkl')
    
    model = BatteryDNN(input_dim=len(features))
    torch.save(model.state_dict(), 'master_battery_dnn.pth')
    print("✅ Huấn luyện xong! Mô hình đã đồng bộ 14 cột.")

if __name__ == "__main__":
    run_training()