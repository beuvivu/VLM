# VLM: dữ liệu có nguồn và tính thưởng xác định

Mục tiêu: thực hiện năm hạng mục người dùng đã yêu cầu trên repo VLM, giữ tương thích với `vietlott_engine` và bổ sung API `vlm` trong cấu trúc `src/` hiện có.

## Kiến trúc

- Tái sử dụng danh mục game, công thức Bao và bộ phân tích HTML/AjaxPro hiện có. `vlm.rules.calculator` bổ sung đối soát vé với kết quả thật và chia quỹ chính xác bằng `Fraction`.
- `vlm.crawler.async_scraper` thay thế transport khi được chọn: curl_cffi, profile trình duyệt, proxy được cấu hình, retry 1/2/4/8/16 giây, giới hạn tốc độ và thời gian. Session clearance gắn với proxy và User-Agent. FlareSolverr là dịch vụ tùy chọn; lỗi challenge đi qua fallback, không được coi HTML challenge là dữ liệu.
- `vlm.database.schema`: mô hình Pydantic thống nhất; SQLAlchemy cho SQLite/PostgreSQL, backend DuckDB độc lập và xuất/nhập Parquet. Không suy diễn số người trúng/doanh thu còn thiếu thành 0; không ghi đè kết quả mâu thuẫn.
- `vlm.tools.gap_analyzer`: đọc seed trực tiếp, kiểm tra từ kỳ 1, duplicate, dòng không hợp lệ, ngày đảo thứ tự, kỳ loại trừ và trạng thái cuối dải. Chỉ endpoint latest/giới hạn xác nhận mới cho phép kết luận đầy đủ đến hiện tại. Keno/Bingo không suy diễn lịch sử draw_id từ lịch quay hiện tại.
- Repair chỉ nhận kỳ trong báo cáo, kiểm tra đúng game/id, ghi nguyên tử và lưu thất bại để chạy lại; không điền kỳ không được Hội đồng xác nhận.

## Phạm vi và tính đúng

Mega/Power: vé đơn, Bao 5, 7–15, 18; Jackpot chia theo lượt trúng, không nhân cả quỹ cho từng tổ hợp. Power chỉ chuyển phần vượt 300 tỷ khi không có JP1 và có JP2.

Max3D/+: chung kết quả; Pro riêng. Bao bộ số khử permutation trùng chữ số, Bao nhiều số phân biệt cặp có thứ tự với cặp không thứ tự. Cách chơi chưa xác minh chính thức phải gắn nhãn nghiên cứu, không mặc định được bán.

Keno: 1–10, các cửa phụ, hạn mức tổng trả thưởng từng bậc 8/9/10. Ô bậc 5 trùng 4 số mâu thuẫn giữa trang giới thiệu và trang chi tiết phải thể hiện nguồn. Bingo18: đủ 38 cửa (6 một số, 6 đôi, 6 ba cụ thể, 1 ba bất kỳ, 16 tổng, 3 lớn/hòa/nhỏ).

Lotto5/35 và Max4D đã có trong repo được giữ trong audit; không đánh đồng Max4D đã ngừng với dữ liệu trễ.

## Kiểm chứng

Unit tests đối chiếu ma trận Bao với vét cạn, xét Bao 5/7/18, bonus trong/ngoài bộ, phân chia Jackpot và tiền nguyên; schema round-trip SQLite/DuckDB/Parquet, SQL PostgreSQL; retry/block HTML 200/fallback sai id; leading/internal/trailing gaps và loại trừ. Chạy toàn bộ suite hiện có và Ruff; kiểm đếm lại seed, báo cáo nguồn chưa truy cập được.
