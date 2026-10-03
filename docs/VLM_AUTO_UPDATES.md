# Tự cập nhật kết quả Vietlott

API tự chạy bộ cập nhật khi khởi động (`VQE_AUTO_UPDATE_ENABLED=true`, mặc định). Không cần gọi sync thủ công. Mọi quyết định lịch, checkpoint và trạng thái dùng giờ Việt Nam UTC+7, không phụ thuộc timezone của máy.

## Lịch và nhịp kiểm tra

| Sản phẩm | Giờ quay công bố | Cửa sổ kiểm tra mỗi 2 phút |
|---|---|---|
| Mega 6/45 | Thứ 4, 6, Chủ nhật, 18:00–18:30 | 18:00–19:00 đúng ngày quay |
| Power 6/55 | Thứ 3, 5, 7, 18:00–18:30 | 18:00–19:00 đúng ngày quay |
| Max 3D / Max 3D+ | Thứ 2, 4, 6, 18:00–18:30; dùng chung kết quả | 18:00–19:00 đúng ngày quay |
| Max 3D Pro | Thứ 3, 5, 7, 18:00–18:30 | 18:00–19:00 đúng ngày quay |
| Lotto 5/35 | 13:00 và 21:00 hằng ngày | 13:00–14:00, 21:00–22:00 |
| Keno | Quay nhanh xuyên ngày; trang chủ hiện công bố 8 phút/kỳ | 06:00–22:15 hằng ngày |
| Bingo18 | 6 phút/kỳ, phát hành 06:00–21:53 | 06:00–22:15 hằng ngày |
| Max 4D | Đã ngừng phát hành | Không có kỳ mới để cập nhật |

Ngoài cửa sổ trên, kiểm tra mỗi giờ; nếu đang thiếu kết quả đến hạn, thử lại chậm nhất sau 5 phút cả ngoài cửa sổ. Weekly chỉ được coi đã đến hạn công bố sau 18:30; Lotto có khoảng chờ đến 13:10/21:10. Đây là khoảng chờ kiểm định, không phải lời hứa thời điểm Vietlott/mirror công bố. Không suy ra ID kỳ quay hoặc bộ số từ thời gian.

Nguồn lịch: [Mega](https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/645), [Power](https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/655), [Max 3D](https://media.vietlott.vn/vi/04.2019/system/archivedate/the-le-max-3d.pdf), [Max 3D Pro](https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/max-3DPro), [Lotto](https://vietlott.vn/vi/tin-tuc/tin-hoat-dong/20235-thong-bao-chinh-thuc-phat-hanh-san-pham-xo-so-tu-chon-lotto-535/), [Keno](https://vietlott.vn/vi/home), [Bingo18](https://vietlott.vn/vi/choi/bingo/cach-choi). Đối chiếu ngày 03/10/2026. Polling nhanh hơn nhịp quay và không khóa cứng số kỳ/ngày, nên vẫn dùng được khi cadence được điều chỉnh.

## Chạy thường trực

```bash
docker compose build
docker compose up -d api
curl http://localhost:8000/updates/status
```

Docker đã cài `curl_cffi`, dùng browser transport và giữ CSDL/journal/checkpoint trong volume `vqe-data`. Một API worker sở hữu DuckDB. Service `scheduler` cũ vẫn có thể chạy để làm mới bảng giải và dự báo mỗi 6 giờ, nhưng kết quả quay được bộ cập nhật trong API lấy theo lịch trên.

Cài native: `pip install -e ".[crawler]"`, rồi `vietlott serve`; API tự cập nhật. Nếu không chạy API, có thể chạy `vlm-update` liên tục qua service manager của máy. `vlm-update --once` dành cho cron và `vlm-update --status` chỉ đọc trạng thái. Không chạy CLI writer cùng CSDL với API: dùng API sync endpoint; khóa OS từ chối writer thứ hai. Tiến trình cần chạy thường trực để đạt nhịp kiểm tra 2 phút.

```bash
# Đồng bộ đến hạn một lần khi API đang tắt
vlm-update --once
# Buộc thử ngay một sản phẩm
vlm-update --once --product power655 --force
# Đọc checkpoint mà không mở DuckDB
vlm-update --status
```

## Recovery trên GitHub

`results.yml` chạy mỗi 10 phút từ 06:00–22:50 và mỗi giờ ban đêm, dùng cron UTC đã quy đổi. Push thay đổi bộ cập nhật lên `main` cũng khởi động một lần chạy. GitHub có thể trì hoãn cron; đây là luồng cào bù khi không có API thường trực, không bảo đảm realtime.

Kết quả đã validate được lưu vào `data/results/results.jsonl` và commit vào repo. Checkpoint/kho nhanh được cache; mất cache thì nạp seed rồi replay journal. Workflow dự báo/trang hiện có vẫn chạy hai lần/ngày, cũng replay và giữ journal trước khi tạo dự báo. Lỗi nguồn xuất hiện trong artifact `vietlott-results-status` và làm workflow báo lỗi sau khi đã lưu phần thành công; lần chạy sau tiếp tục thử các game bị lỗi.

## Trạng thái và toàn vẹn

`GET /updates/status` có `last_attempt`, `last_success`, `last_new_result`, `next_attempt`, ID/ngày cuối, số lỗi liên tiếp, lỗi theo loại, freshness và các conflict khi replay. `last_success` chỉ nghĩa lần đồng bộ hoàn tất, không chứng minh đã có kết quả mới. Chỉ `freshness.verified=true` mới xác nhận các slot có thể kiểm bằng ngày/số kỳ đã lưu. Keno/Bingo18 lưu ngày nhưng không đủ thời gian chính xác từng kỳ nên trả `date_only`, `verified=false`; không được đổi thành “đã cập nhật đến hiện tại”. Nếu ID đứng yên ít nhất 20 phút trong giờ phát hành 06:00–21:53, trạng thái chuyển `not_advancing` để báo nguồn có thể trễ. Ngày ngoài lịch quay cũng không được chứng nhận. CLI trả mã lỗi khi nguồn/worker lỗi, quá lịch hoặc không tiến triển; GitHub hiển thị lỗi sau khi giữ phần kết quả đã lấy được.

Mỗi game có task và timeout riêng, không chặn tick của game khác. Lỗi được retry sau 30/60/120/240/300 giây; không có task trùng cho cùng game. Mỗi nguồn có budget 20 giây; HTTP 200 nhưng không có kỳ mới vẫn chuyển fallback. Kết quả được lưu trước bước bảng giải tùy chọn; lỗi bảng giải không xóa kết quả. API sync thủ công và worker dùng chung khóa từng game.

Journal append/fsync, checkpoint và file lịch sử JSONL/gzip thay file nguyên tử; ghi thất bại vẫn giữ file cũ. Replay không ghi đè kết quả đã có khi khác với journal; conflict phải đối chiếu nguồn. Kết quả sửa được ghi thành revision mới trong journal. Một JSON tail chưa hoàn tất được giữ vào `.corrupt`, phần hợp lệ được phục hồi; lỗi giữa file/record hoàn chỉnh vẫn được báo để tránh bỏ dữ liệu âm thầm.

Các khoảng trống lịch sử đã ghi trong [Audit Matrix](../reports/VLM_AUDIT_2026-10-03.md) vẫn cần `vlm-audit --repair`; cập nhật tăng dần không chứng nhận mọi ID từ kỳ đầu đã đủ. Cloudflare hoặc mirror trễ có thể làm kết quả chậm hơn nhịp polling; trạng thái thể hiện điều này.

## Cấu hình

- `VQE_AUTO_UPDATE_ENABLED`: bật/tắt worker API, mặc định `true`.
- `VQE_AUTO_UPDATE_TIMEOUT_S`: budget mỗi game, mặc định 180 giây.
- `VQE_SOURCE_TIMEOUT_S`: budget mỗi nguồn, mặc định 20 giây.
- `VQE_AUTO_UPDATE_DIR`: mặc định `<data_dir>/updates`; GitHub dùng `data/results`.
- `VQE_HTTP_BACKEND=curl_cffi`: cần extra `crawler`, Docker/GitHub đã cài.
- Proxy/solver do người vận hành cung cấp theo [VLM_INTEGRITY](VLM_INTEGRITY.md).
