# Danh sách trang vietlott.vn nên kiểm tra

Engine không vượt bước *xác minh bạn là người* (Cloudflare) của vietlott.vn. Bạn mở các trang dưới đây bằng trình duyệt bình thường, tự qua bước xác minh, rồi lưu trang: **Ctrl+S → "Trang web, một tệp" (.mhtml)**. Hoặc mở DevTools (F12) ▸ Network *trước* khi tải trang, chọn bộ lọc **All**, duyệt các trang, rồi bấm nút mũi tên tải xuống **"Export HAR (sanitized)…"** trên thanh công cụ của tab Network (.har; Chrome/Edge bản mới — bản cũ: chuột phải ▸ "Save all as HAR with content"). Bỏ tất cả vào một thư mục, rồi chạy:

```bash
vietlott products import-pages --path <thư-mục> --dry-run   # chỉ so sánh
vietlott products import-pages --path <thư-mục>             # so sánh và nhập
```

Nên mở chậm, như người đọc bình thường. Nhóm 1 là quan trọng nhất (vài trang); nhóm 2–4 là mẫu ngẫu nhiên (hạt giống 20261002) để ước lượng tỷ lệ sai của nguồn phụ: 30 kỳ khớp cả ⇒ tỷ lệ sai ≤ 9.5% (độ tin cậy 95%).

| Nhóm | Sản phẩm | Kỳ | Lý do | Trang |
|---|---|---:|---|---|
| 1 · ưu tiên | lotto535 | 920 | kết quả và bảng giải chỉ có từ trang phụ (xosominhngoc) | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/535?id=00920&nocatche=1 |
| 1 · ưu tiên | lotto535 | 874 | Jackpot lệch giữa nguồn: 30.22 tỷ (bản ghi chính thức) vs 35.96 tỷ (trang phụ) | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/535?id=00874&nocatche=1 |
| 1 · ưu tiên | keno | 297782 | trang chi tiết Keno: đọc ô bậc 5 trùng 4 số (150.000 đ hay 0 đ?) và các cửa phụ | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297782 |
| 1 · ưu tiên | keno | 297778 | kỳ gần đây chỉ có từ trang phụ (xosominhngoc) | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297778 |
| 1 · ưu tiên | keno | 297779 | kỳ gần đây chỉ có từ trang phụ (xosominhngoc) | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297779 |
| 1 · ưu tiên | keno | 297780 | kỳ gần đây chỉ có từ trang phụ (xosominhngoc) | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297780 |
| 1 · ưu tiên | keno | 297781 | kỳ gần đây chỉ có từ trang phụ (xosominhngoc) | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297781 |
| 1 · ưu tiên | bingo18 | 189249 | kỳ gần đây chỉ có từ trang phụ (xosominhngoc); trang danh sách Bingo18 lưu bằng HAR phủ được nhiều kỳ | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0189249 |
| 2 · mẫu Keno 2019–2022 | keno | 4073 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0004073 |
| 2 · mẫu Keno 2019–2022 | keno | 11662 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0011662 |
| 2 · mẫu Keno 2019–2022 | keno | 21549 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0021549 |
| 2 · mẫu Keno 2019–2022 | keno | 23302 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0023302 |
| 2 · mẫu Keno 2019–2022 | keno | 29314 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0029314 |
| 2 · mẫu Keno 2019–2022 | keno | 31785 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0031785 |
| 2 · mẫu Keno 2019–2022 | keno | 33457 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0033457 |
| 2 · mẫu Keno 2019–2022 | keno | 44132 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0044132 |
| 2 · mẫu Keno 2019–2022 | keno | 45482 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0045482 |
| 2 · mẫu Keno 2019–2022 | keno | 53945 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0053945 |
| 2 · mẫu Keno 2019–2022 | keno | 58631 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0058631 |
| 2 · mẫu Keno 2019–2022 | keno | 71075 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0071075 |
| 2 · mẫu Keno 2019–2022 | keno | 87236 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0087236 |
| 2 · mẫu Keno 2019–2022 | keno | 129264 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0129264 |
| 2 · mẫu Keno 2019–2022 | keno | 134156 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0134156 |
| 2 · mẫu Keno 2019–2022 | keno | 137627 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0137627 |
| 2 · mẫu Keno 2019–2022 | keno | 142293 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0142293 |
| 2 · mẫu Keno 2019–2022 | keno | 160250 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0160250 |
| 2 · mẫu Keno 2019–2022 | keno | 162282 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0162282 |
| 2 · mẫu Keno 2019–2022 | keno | 165051 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0165051 |
| 2 · mẫu Keno 2019–2022 | keno | 167849 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0167849 |
| 2 · mẫu Keno 2019–2022 | keno | 181644 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0181644 |
| 2 · mẫu Keno 2019–2022 | keno | 203796 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0203796 |
| 2 · mẫu Keno 2019–2022 | keno | 204027 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0204027 |
| 2 · mẫu Keno 2019–2022 | keno | 207728 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0207728 |
| 2 · mẫu Keno 2019–2022 | keno | 211787 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0211787 |
| 2 · mẫu Keno 2019–2022 | keno | 213396 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0213396 |
| 2 · mẫu Keno 2019–2022 | keno | 215861 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0215861 |
| 2 · mẫu Keno 2019–2022 | keno | 226824 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0226824 |
| 2 · mẫu Keno 2019–2022 | keno | 226830 | mẫu ngẫu nhiên phần chỉ có ở kho xoso.com.vn | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0226830 |
| 3 · mẫu Bingo18 | bingo18 | 135672 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0135672 |
| 3 · mẫu Bingo18 | bingo18 | 136069 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0136069 |
| 3 · mẫu Bingo18 | bingo18 | 136807 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0136807 |
| 3 · mẫu Bingo18 | bingo18 | 138225 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0138225 |
| 3 · mẫu Bingo18 | bingo18 | 139399 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0139399 |
| 3 · mẫu Bingo18 | bingo18 | 141000 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0141000 |
| 3 · mẫu Bingo18 | bingo18 | 141489 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0141489 |
| 3 · mẫu Bingo18 | bingo18 | 143779 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0143779 |
| 3 · mẫu Bingo18 | bingo18 | 144913 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0144913 |
| 3 · mẫu Bingo18 | bingo18 | 145469 | mẫu ngẫu nhiên phần kho ghi nguồn 'unknown' | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-bingo18-result?nocatche=1&id=0145469 |
| 4 · khoảng trống Keno | keno | 32998 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0032998 |
| 4 · khoảng trống Keno | keno | 295698 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0295698 |
| 4 · khoảng trống Keno | keno | 297355 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297355 |
| 4 · khoảng trống Keno | keno | 297458 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297458 |
| 4 · khoảng trống Keno | keno | 297485 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297485 |
| 4 · khoảng trống Keno | keno | 297502 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297502 |
| 4 · khoảng trống Keno | keno | 297525 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297525 |
| 4 · khoảng trống Keno | keno | 297715 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297715 |
| 4 · khoảng trống Keno | keno | 297725 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297725 |
| 4 · khoảng trống Keno | keno | 297740 | mã kỳ không có trong dữ liệu: có kỳ quay này không? | https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/view-detail-keno-result?id=0297740 |
