# Trang chính VLM: kết quả, dự báo, đối chiếu

## Phạm vi
Trang `index.html` là bảng theo dõi các sản phẩm đang hoạt động: Mega 6/45, Power 6/55, Lotto 5/35, Max 3D / Max 3D+, Max 3D Pro, Keno, Bingo18. Trang phân tích hiện tại chuyển thành `forecast.html`; giữ các JSON cũ để VLA tiếp tục đọc. Max 4D chỉ xuất hiện ở trang phân tích lưu trữ.

## Thiết kế
VLA: nền #F4F5FF / #E8ECFF, tím #6554D9, xanh #356AE6, teal #087F83, chữ slate; Inter/Poppins và system fallback. Header cố định nhẹ, logo VLM, điều hướng kết quả/dự báo/đối chiếu, nút đổi sáng tối lưu trên máy. Nội dung tối đa 1440px, thẻ bo 20px, khoảng cách 24px desktop / 16px mobile, điều khiển tối thiểu 44px. Grid desktop 12 cột, mobile một cột; bảng giải và thẻ dự báo đọc được ở 375px/414px. Tôn trọng reduced motion; dùng text kèm màu để biểu thị trùng số. Không dựng thành tựu/giải thưởng tổ chức chưa có bằng chứng.

## Hợp đồng dữ liệu
`data/dashboard.json` có `generated_at`, `products` (latest results, recent draw archive, full prize catalogue, next forecast, comparison archive), `stats`, `warnings`. Nguồn là seed, cache kết quả mới, DuckDB (nếu có), journal đã xác thực. Loại kỳ không xác nhận. Giá trị và số người trúng khuyết giữ null; tiền hiển thị VND. Bảng giải đúng mã kỳ, không lấy Jackpot hay số người trúng của kỳ trước cho kỳ mới.

Dự báo đã đăng ký dùng phân phối ML bất biến trong `ml-ledger.jsonl` để tái dựng Top 5 bằng chính config tìm kiếm đã lưu; legacy ledger là dự phòng. Chọn bản đầu tiên hợp lệ cho mỗi product/target_id/engine, ưu tiên ML. Chỉ hợp lệ khi target = based_on + 1, made_at trước thời gian kỳ ít nhất 5 phút, lịch kỳ khớp sản phẩm, dự báo không mô phỏng. Không dự báo lại quá khứ bằng checkpoint hiện tại. Đối chiếu với đúng product + draw_id + ngày; thiếu kết quả thì pending, sai ngày thì mismatch. Không giả lập kết quả hoặc giải thưởng thực nhận. Matrix: số chính, bonus, hạng giải. Max: khớp toàn bộ chuỗi 3 chữ số, xác định các nhóm giải, không dùng chữ số khớp vị trí như trúng cả số. Bingo: thứ tự và số trùng/tổng tách riêng. Keno: dãy tham chiếu 20 số, không gọi là vé 20 số hợp lệ.

Khi chưa có dự báo ML đã đăng ký, dùng báo cáo checkpoint hiện tại (ML hoặc legacy) chỉ cho latest+1 nếu dữ liệu checkpoint đồng bộ; gắn trạng thái tham khảo, không đưa vào thành tích đối chiếu. Keno/Bingo18 thiếu thời điểm từng kỳ nên không chứng nhận live. Không đồng nhất điểm mô hình với xác suất thắng đảm bảo.

## Vận hành
Cả luồng results thường xuyên và batch update đồng bộ main trước khi cập nhật, rồi xác nhận journal và hai sổ dự báo đã lưu vào repo trước khi dựng/triển khai hoặc lưu cache. Pull/rebase/push lỗi chặn xuất bản; một nguồn hoặc mô hình lỗi vẫn giữ kết quả đã xác thực của các sản phẩm khác. Khi hai nguồn cung cấp cùng kết quả của cùng kỳ, ưu tiên nguồn chính thức và giữ các giá trị tài chính đã biết; dữ liệu xung đột tạo cảnh báo. Site poll dashboard JSON 60 giây khi tab hiển thị; nút làm mới; lỗi mạng/JSON sai cấu trúc hoặc thời gian tương lai giữ bản tốt cuối cùng và phục hồi khi có bản hợp lệ, hiện trạng thái cũ và thời gian cập nhật. Cron GitHub có thể trễ; UI không hứa realtime. Dựng từ ledger và seed vẫn chạy khi mất cache. Bộ lọc sản phẩm, chọn kỳ kết quả, mở bảng cơ cấu giải và danh sách đối chiếu phải hoạt động; không mất focus/lựa chọn khi auto refresh.
