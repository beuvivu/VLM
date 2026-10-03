# VLM: dự báo ML và học sau từng kỳ quay

Bộ mới nằm trong `vlm.forecast`, chạy song song với bộ dự báo Bayesian hiện có.
Mặc định dùng dữ liệu đã lưu trong repo, gồm seed và journal kết quả. Người dùng
có thể cung cấp JSONL khác; mọi dòng được validator kiểm tra trước khi học.

## Sử dụng

```bash
pip install -e ".[dev]"
vlm-forecast next --product mega645 --top-n 10
vlm-forecast next --product power655 --top-n 10 --budget 100000
vlm-forecast update --product all
vlm-forecast benchmark --product all --bootstrap 1000
```

Boosters là tùy chọn; GRU NumPy và Random Forest không cần PyTorch/GPU.

```bash
pip install -e ".[ml]"
vlm-forecast fit --product mega645 --backends rf,xgb,lgb --bootstrap 1000
vlm-forecast benchmark --product mega645 --backends rf,xgb,lgb --bootstrap 1000
```

`fit` khởi tạo lại trọng số và bằng chứng live của sản phẩm được chọn. `next`
và `update` sử dụng cấu hình đã lưu; đổi backend/bootstrap của model hiện có
bằng `fit`. `benchmark` tạo model độc lập, không ghi thành tích live và không
ghi đè checkpoint. Đường dẫn `--dir` là thư mục forecast, với state ML trong
`ml/<product>.json.gz`. Repo đã có lệnh `vietlott forecast` vẫn dùng được.

```bash
vlm-forecast next --product mega645 --input history.jsonl --top-n 5 --dir data/local/forecast
```

Ví dụ một dòng JSONL:

```json
{"id":1570,"date":"2026-10-02","result":[1,7,15,24,32,45]}
```

Đây là ví dụ định dạng, không phải kết quả kỳ #1570. Power cần `bonus_number`,
Lotto cần số đặc biệt. Có thể dùng schema canonical `DrawRecord` với
`game_type`, `draw_id`, `draw_date`, `winning_numbers`. Sai phạm vi số, số trùng,
ID trùng, ngày tương lai hoặc ngày đi lùi trong cửa sổ học sẽ bị từ chối.

API:

| Endpoint | Chức năng |
|---|---|
| `GET /ml/forecast/mega645?top_n=10&budget=100000` | Học kỳ mới, trả Top N, bằng chứng, metrics và danh mục trong ngân sách |
| `POST /ml/forecast/mega645/fit` | Khởi tạo lại với body theo `MLConfig`; reset bằng chứng live |
| `GET /updates/status` | Trạng thái nguồn, độ mới kết quả và `products.<game>.learning` |

`product` nhận `mega645`, `power655`, `lotto535`, `keno`, `bingo18`, `max3d`,
`max3dpro`, `max4d`. Max 3D/+ dùng chung kết quả. Max 4D là lưu trữ, không có
`target_id` tiếp theo.

## Cách đọc xác suất và confidence

| Trường | Ý nghĩa |
|---|---|
| `top[].p_model` | Xác suất của ứng viên theo luật ensemble thử nghiệm |
| `top[].p_fair` | Xác suất theo máy quay công bằng |
| `top[].p_deployed` | Xác suất được dùng: fair cho tới khi qua gate live |
| `marginals_model` | Xác suất xuất hiện của mỗi số/vị trí; không phải xác suất trúng toàn bộ |
| `ranking_exact` | Đã chứng minh Top N toàn cục; `false` nghĩa tìm kiếm hết budget, chỉ xếp hạng ứng viên |
| `feature_support` | Tần suất, EMA, gap, cặp/ba của các số trong ứng viên; chỉ là mô tả thống kê |
| `confidence.validated` | Đủ điều kiện kiểm định live của chính sản phẩm này |
| `confidence.confidence_score` | `null`: không chế tạo một phần trăm chắc thắng từ e-value/backtest |
| `confidence.family_adjusted_anytime_p` | Kiểm định null, đã tính bảy sản phẩm; không phải P(trúng vé) |
| `metrics.mean_log_gain_nats` | Log-score ensemble trừ fair, chấm trước-học-sau; âm nghĩa kém hơn fair |
| `metrics.brier_observed_rates`, `reliability` | Sai số xác suất và bins calibration; Max đo tỷ lệ chữ số trong nhiều giải |

Xác suất Top N không cộng về 100%; hàng triệu tổ hợp khác vẫn còn xác suất.
Với Power, `top` là tập **sáu số chính**, đã marginalize bonus; xác suất jackpot
1 của vé sáu số không cần khớp bonus. `Law.probability(numbers, bonus=...)`
tính joint khi cần bóng bonus. Lotto báo main và special riêng; danh mục vé
kèm special, giả định hai thành phần độc lập. Max báo một tuple chữ số, không
báo joint của toàn bộ 20 giải; không dùng tuple đó như xác suất trúng một tầng
giải cụ thể. Keno báo tập 20 số kết quả, không chuyển thành vé chơi 20 số.

Gate yêu cầu ít nhất 100 kỳ live, e-value **hiện tại** ≥ 140, tổng log-score
100 kỳ live gần nhất dương và dữ liệu không có gap/date anomaly. Thành tích chỉ
lấy phân phối được lưu trước thời điểm quay với target có thể xác minh. Cấu
hình clock mô phỏng `VQE_FORECAST_NOW` không được đăng ký bằng chứng live.
Keno/Bingo18 hiện dùng Series chỉ có ngày, nên không được chứng nhận live từng
kỳ dù bảng kết quả trông đủ một ngày. Timestamp từng kỳ là điều kiện cần để
mở rộng phần này. Đây là giới hạn kiểm định; crawler và học online vẫn chạy.
Core, checkpoint loader và scorer đều kiểm tra slot chính xác theo giờ Việt
Nam và margin trước giờ quay năm phút. Không thể đăng ký giờ tùy ý sau giờ
quay bằng cách gọi trực tiếp `issue()`.

## Thuật toán và cơ chế tự học

1. Snapshot lấy trước kết quả: hot/cold, MA 10/50, EMA 10/100, khoảng mã kỳ và
   censored gap, lag 1/2/10 **theo các quan sát có dữ liệu**, đồ thị cặp/ba,
   cụm đồng xuất hiện, tổng, tỷ lệ chẵn/lẻ, đầu/đuôi, số ngày và ID bị bỏ qua.
   Lags trên lịch sử thiếu kỳ không phải lags cách đều theo thời gian.
2. Hai logistic AdaGrad (L2 0,001/0,01), GRU thật với reset gate/update gate,
   BPTT tám snapshot, Adam/L2/gradient clipping. Cây giới hạn depth 4, 24 cây,
   buffer 64 kỳ; XGB/LGB thêm shrinkage và regularization. Cây fit lại định kỳ;
   SGD/GRU và trọng số policy học từng kỳ, không gọi cây là mô hình partial_fit.
3. Set games dùng conditional Bernoulli: `P(S) = product(w_i)/e_k(w)`.
   Power bonus được lấy từ phần còn lại của drum. Digit games dùng categorical
   theo vị trí. Ensemble trộn các **luật joint**; không nhân các marginal.
4. Reward là log likelihood của kết quả thực tế. Fixed-share exponential weights
   tự phân bổ trọng số giữa các expert sau khi chấm. Đây là học online có phản
   hồi đầy đủ; chính sách không điều khiển quá trình quay. Hai L2 là các ứng viên
   hyperparameter định trước; không tune bằng cách nhìn nhãn tương lai. Giữ
   proper loss thay vì thay loss để làm đẹp backtest.
5. Bootstrap tối đa 1.000 kỳ gần nhất, rồi chỉ nạp kỳ mới. Backtest bootstrap là
   thử nghiệm hồi cứu. Kiểm định live là một luồng riêng, bắt đầu sau khởi tạo.

## Vận hành và phục hồi

API/CLI updater tự gọi bộ học sau khi lưu kết quả, mặc định
`VQE_ML_AUTO_UPDATE_ENABLED=true`. CPU chạy trong thread, cùng một guard với
legacy forecast; writer giữa các tiến trình bị khóa bằng OS lease. Lỗi model
hiện trong `learning.error`; kết quả đã fsync vẫn được giữ. CLI forecast nên
chạy khi API không chiếm cùng file DuckDB, hoặc gọi API đang chạy.
Updater ghi trạng thái thành công của kết quả trước khi chạy learner với
deadline riêng. Timeout model trả `learning.error=TimeoutError`, `pending=true`;
job vẫn được theo dõi, không xếp trùng job cùng sản phẩm và được đợi trước khi
đóng CSDL. Trạng thái kết quả không bị đổi thành lỗi nguồn vì model chậm.

| Cấu hình | Mặc định |
|---|---:|
| `VQE_ML_BOOTSTRAP` | 1000 |
| `VQE_ML_LEARNING_TIMEOUT_S` | 180 giây |
| `VQE_ML_TREE_EVERY` | 64 |
| `VQE_ML_BACKENDS` | `["rf"]` |
| `VQE_ML_SEARCH_NODES` | 1000 |

Checkpoint JSON gzip được fsync rồi atomic replace, không pickle. Hash prefix
lịch sử nhận ra backfill hoặc sửa kết quả cũ: model reset/replay cửa sổ giới
hạn, xóa evidence/pending cũ. State hỏng được giữ với suffix `.invalid`, học
lại và báo `checkpoint_recovered`. File `ml-ledger.jsonl` lưu immutable issue/
score events; workflow commit ledger và cache state ML. Cache mất thì evidence
live không được dựng lại bằng một backtest giả.
Checkpoint version 2 không nhận evidence từ state version 1. Journal dự báo
phục hồi fragment cuối bị ghi dở (kể cả byte UTF-8 bị cắt), giữ bản `.corrupt`
và chuẩn hóa newline. Record hỏng giữa file hoặc record hoàn chỉnh sai định
dạng vẫn là lỗi cần kiểm tra, không bị bỏ qua âm thầm.

Ngày đi lùi ở phần lịch sử ngoài cửa sổ bootstrap được báo `date_anomalies`,
không tự sửa. Cửa sổ học có ngày đi lùi bị từ chối, và mọi anomaly/gap còn tồn
tại đều khóa chứng nhận. Sau kiểm tra/sửa nguồn, dùng `fit` nếu cần bắt đầu lại
luồng live đã bị mismatch hoặc clock mô phỏng.

Danh mục trong `--budget` chỉ hỗ trợ vé đơn Mega/Power/Lotto, tối đa 100 vé,
10.000 đồng/vé và không chi quá ngân sách. Greedy ưu tiên vé ít giao số trong
một candidate pool, không phải lời giải tối ưu toàn cục cho mọi mức giải. Các
sản phẩm còn lại trả ghi chú để tránh chuyển một tuple dự báo thành vé sai luật.
Không mua vé, không tự đặt cược, không dùng martingale/gấp thếp để đuổi lỗ.

## Tái lập kiểm định

```bash
python scripts/ml_report.py --bootstrap 1000 --out reports/ml-rf-diagnostics.json
python scripts/ml_report.py --bootstrap 128 --backends rf,xgb,lgb --out reports/ml-boosters-diagnostics.json
python -m pytest
ruff check --select F,B023 src scripts tests
```

Kết quả ngày 03/10/2026 và hash dữ liệu từng sản phẩm nằm trong
`reports/VLM_ML_2026-10-03.md` và `reports/ml-rf-diagnostics.json`.

Tài liệu kỹ thuật: [GRU](https://arxiv.org/abs/1406.1078),
[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[calibration](https://scikit-learn.org/stable/modules/calibration.html),
[XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html),
[LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html),
[anytime-valid inference](https://safestatistics.com/wp-content/uploads/2023/10/RamdasSAVIStatScience23.pdf).
