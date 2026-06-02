https://drive.google.com/drive/folders/19mpYVv_SEXilw3iv1eR-NWUietI_gj6Z?usp=sharing
Dataset của dự án
# Hệ Thống Dự Báo Dung Lượng & Tuổi Thọ Pin (Battery Health & RUL Prediction)
- Đây là ứng dụng phân tích chuyên sâu về sức khỏe pin (Battery Health) và dự báo tuổi thọ hữu dụng còn lại (RUL - Remaining Useful Life) được xây dựng bằng Python. Hệ thống kết hợp giữa mô hình học máy (Machine Learning) và các công thức vật lý để đưa ra dự báo chính xác về dung lượng còn lại của pin (mAh) và số chu kỳ sạc khả dụng.
## Tính năng chính
- Đa nguồn dữ liệu (Multi-Source Data Integration): Tự động thu thập và chuẩn hóa dữ liệu từ nhiều nguồn tiêu chuẩn như Oxford, LG và các tệp dữ liệu CSV riêng biệt.
- Dự báo AI mạnh mẽ: Sử dụng kết hợp Random Forest (cho dữ liệu cấu trúc) và Deep Neural Networks (DNN) (cho dự báo phi tuyến tính) để xác định dung lượng pin.
- Logic tính toán RUL chuyên sâu: Không chỉ dựa vào AI, hệ thống áp dụng các công thức vật lý để tính toán sự suy giảm dung lượng theo chu kỳ (cycle), thời gian sử dụng (time) và ảnh hưởng nhiệt độ (temperature penalty).
- Trực quan hóa: Hỗ trợ biểu đồ hóa mức độ ảnh hưởng của các thông số đầu vào (Feature Importance), giúp người dùng hiểu rõ yếu tố nào đang tác động mạnh nhất đến sức khỏe viên pin.
## Công nghệ sử dụng
- Ngôn ngữ: Python 3.x
- Xử lý dữ liệu: pandas, numpy, scipy.io (xử lý file .mat)
- Học máy: scikit-learn (Random Forest), PyTorch (Deep Neural Networks)
- Trực quan hóa: matplotlib, seaborn
## Nguyên lý hoạt động
 Hệ thống hoạt động theo mô hình lai (Hybrid Model) để đảm bảo độ tin cậy:
### 1. Dự đoán dung lượng (Capacity Prediction)
 Thay vì sử dụng các phương pháp đơn giản, chúng tôi huấn luyện mô hình dựa trên các biến đặc trưng (Features) quan trọng:
- Điện áp (V), Dòng điện (I), Nhiệt độ (T)
- Số chu kỳ sạc & Thời gian sạc
- Các thông số từ tập dữ liệu Oxford & LG.
### 2. Ước tính RUL (Remaining Useful Life)
 Hệ thống sử dụng phương pháp nội suy dựa trên tốc độ suy giảm thực tế:
- Độ suy giảm: total_drop = (cycle_drop + time_drop) * temp_penalty
- Tính toán RUL: Khi dung lượng dự đoán chạm ngưỡng giới hạn (EOL - End of Life, thường ở mức 70%), hệ thống tính toán dựa trên tốc độ mất dung lượng trung bình qua mỗi chu kỳ:
  
  ### $$RUL = \frac{\text{Dung lượng hiện tại} - \text{Dung lượng tới hạn}}{\text{Tốc độ mất dung lượng mỗi chu kỳ}}$$
## Hướng dẫn Cài đặt & Sử dụng
### 1. Chuẩn bị môi trường
- Cài đặt các thư viện cần thiết thông qua pip:

Bash
pip install pandas numpy scikit-learn torch joblib scipy matplotlib seaborn

### 2. Chuẩn bị Dữ liệu
Đặt các tệp dữ liệu vào thư mục tương ứng trong dự án:
- /Oxford_Data/: Chứa các tệp dữ liệu .mat từ nguồn Oxford.
- /LG_Data/: Chứa các tệp dữ liệu .mat từ pin LGHG2.
- Các file CSV (nếu có): Để hệ thống tự động gộp vào Master_Battery_Dataset.csv.
### 3. Khởi chạy
- Chạy file Jupyter Notebook battery_demo1 (4).ipynb để thực hiện quá trình gộp dữ liệu, huấn luyện mô hình (master_battery_rf_model.pkl, master_battery_dnn.pth).
- Chạy script dự báo (predict function) để nhập thông số thực tế của pin và xem kết quả:

Bash
python predict_battery.py
