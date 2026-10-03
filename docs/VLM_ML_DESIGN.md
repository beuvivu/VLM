# VLM: thiết kế dự báo xác suất có kiểm định

## Phạm vi

Dùng lịch sử đang lưu trong VLM khi người dùng chưa cung cấp tệp mới. Giữ bộ
`vietlott_engine.forecast` hiện có tương thích; thêm `vlm.forecast` cho ensemble ML.
Một mô hình cho mỗi component của cả tám sản phẩm, gồm Max 4D lưu trữ. Không coi
vị trí của số đã sắp tăng dần trong Mega/Power/Keno là vị trí quay độc lập.

## Kiến trúc

- Đặc trưng online, lấy snapshot **trước** khi nạp kết quả: tần suất, MA/EMA,
  khoảng cách mã kỳ, lag theo các quan sát có dữ liệu, lặp chu kỳ, cặp/ba đồng xuất hiện,
  cụm đồng xuất hiện, tổng, chẵn/lẻ, đầu/đuôi, khoảng ngày. Kỳ thiếu không được
  điền bằng một kết quả giả; báo censored và khoảng mã kỳ.
- Expert công bằng; hai logistic AdaGrad với L2; GRU dùng NumPy, BPTT đầy đủ qua
  cửa sổ tám snapshot, Adam, L2, gradient clipping; Random Forest giới hạn độ sâu.
  XGBoost và LightGBM là extra `ml`, được bật rõ ràng, lỗi thiếu dependency phải
  được báo. SGD/GRU và trọng số ensemble học từng kỳ; cây fit lại định kỳ trên
  buffer giới hạn, không gọi đây là partial_fit của cây.
- Mô hình đầu ra node scores. Luật tập số là conditional Bernoulli được chuẩn
  hóa bằng đa thức đối xứng bậc k, hỗ trợ bonus Power từ phần còn lại; luật chữ
  số là categorical theo vị trí. Trộn **luật xác suất**, không nhân các xác suất
  biên hay dùng softmax làm xác suất trúng bộ số.
- Học chính sách lựa chọn expert với reward = log likelihood thực tế, fixed-share
  exponential weights. Đây là học online có phản hồi đầy đủ; không giả lập RL có
  khả năng điều khiển máy quay. Các mức L2 định trước là các ứng viên
  siêu tham số được đánh giá trước-học-sau; giữ proper log loss, thêm Brier và
  reliability bins để theo dõi calibration.
- Bootstrap tối đa 1.000 kỳ gần nhất để không chiếm CPU crawler. Backtest bootstrap
  là nghiên cứu khám phá, không phải bằng chứng được đăng ký trước. Sau khởi tạo,
  chỉ forecast đã lưu trước thời điểm quay, có target/time xác minh được, mới
  góp vào bằng chứng live. Ngưỡng e-value 140 = 7 / 0,05, thêm tối thiểu 100 kỳ
  live và log-score gần đây dương; dùng **e-value hiện tại** cho gate. Max 4D
  ngừng phát hành, không có target tiếp theo.
- Chưa qua gate: xác suất sử dụng là luật công bằng, xác suất ML ghi rõ thử nghiệm.
  Không chuyển e-value hoặc điểm model thành phần trăm chắc chắn thắng. Confidence
  là trạng thái kiểm định, số kỳ, p-value anytime dưới null đã nêu và cỡ mẫu,
  không phải xác suất bộ số sẽ trúng.
- Top N chữ số enumerate chính xác. Tập số dùng tìm kiếm có bound và budget;
  nếu chưa chứng minh toàn cục thì `ranking_exact=false`. Probability của mỗi
  ứng viên vẫn tính đúng theo luật mô hình. Top N không được chuẩn hóa về 100%.
- Portfolio ưu tiên vé khác nhau và ít giao nhau, có trần ngân sách; chỉ xuất
  vé legal cho Mega/Power/Lotto (Lotto kèm số đặc biệt). Giảm giao nhau là đa dạng
  coverage, không nâng xác suất của một vé hay chứng minh giảm mọi dạng rủi ro.

## Vận hành

Checkpoint JSON gzip atomically thay thế, không pickle; version/model/config và
hash prefix lịch sử để phát hiện sửa dữ liệu hoặc backfill. Kỳ lặp không học lại.
Sửa một kỳ đã học buộc reset/replay bounded; reset bằng chứng live và pending
forecast. File hỏng phục hồi có cảnh báo và không giữ thành tích cũ.

Sau sync lưu journal, chụp Series dưới lock cập nhật rồi học trong thread, ngoài
event loop. Một lock chung bảo vệ forecast state/ledger giữa API, CLI cùng tiến
trình; khóa file bảo vệ writer giữa tiến trình. Lỗi model không làm mất kết quả
đã lưu. Luồng định kỳ, API và `vlm-update --once` học từ snapshot trước khi trả kết
quả; learner có deadline riêng, kết quả lưu thành công không bị đổi thành lỗi
nguồn khi model chậm. Job hết deadline vẫn được theo dõi đến khi thread dừng;
đợi các job này trước khi đóng CSDL. Trạng thái công khai learned count và lỗi
mô hình. Cache Actions
bao gồm state ML. Crawler giữ nguyên lịch và budget nguồn.

## Kiểm định bắt buộc

Causal-prefix invariance, luật xác suất tổng bằng một trên không gian nhỏ, GRU
gradient finite difference, signal có thể học/không tự chứng nhận noise, score
trước update, reload/chunk invariance, correction/gap/corrupt recovery,
feature shape/range, optional dependency adapters, Top N/cost/duplicate checks,
API/CLI và kết nối updater. Chạy toàn bộ pytest và Ruff, benchmark dữ liệu thật
từng sản phẩm và lưu báo cáo không khẳng định lợi thế từ backtest hồi cứu.

## Nguồn kỹ thuật

- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- https://scikit-learn.org/stable/modules/calibration.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html (công thức GRU;
  implementation này dùng NumPy và reset trước recurrent projection)
- https://xgboost.readthedocs.io/en/stable/python/python_api.html
- https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
- https://safestatistics.com/wp-content/uploads/2023/10/RamdasSAVIStatScience23.pdf
