# Dự báo tự học cho mọi sản phẩm (v3.5) — kết quả trên dữ liệu thật

`vietlott forecast next` · `python scripts/forecast_report.py`. Mỗi *chuyên gia* (ngẫu nhiên đều, số nóng/lạnh toàn bộ lịch sử và gần đây, số lâu chưa về / vừa về, lặp kỳ trước, hồi quy logistic học trực tuyến, Dirichlet cho trò chữ số, và một rổ giả thuyết "một số / một chữ số lệch ±10–25%") đưa ra một phân phối xác suất đầy đủ cho kỳ tới, chỉ từ các kỳ trước. Hỗn hợp Bayes *fixed-share* tự chỉnh trọng số sau mỗi kỳ theo mức mỗi chuyên gia đoán đúng. Tỷ số hợp lý của hỗn hợp so với máy quay công bằng là một **e-value hợp lệ ở mọi thời điểm**: vượt 20 (10^1,30) mới là bằng chứng (α = 5%).

| Sản phẩm | Kỳ đã học | e-value cao nhất / hiện tại | Bằng chứng? | Trọng số "ngẫu nhiên" | Chuyên gia khác nặng nhất | Chấm lùi: trùng × ngẫu nhiên (z) | Hệ số mạnh nhất mô hình tự cho |
|---|---:|---|---|---:|---|---|---:|
| Mega 6/45 | 1.569 | 10^0,20 / 10^-0,46 | không | 50,8% | cold_all_a500 (3,1%) | ×0,969 (-1,26) | ×1,003 |
| Power 6/55 | 1.405 | 10^0,33 / 10^-0,29 | không | 35,2% | tilt[6]-0.25 (9,3%) | ×0,982 (-0,61) | ×1,013 |
| Lotto 5/35 | 920 | 10^0,12 / 10^-0,46 | không | 37,1% | tilt[34]+0.25 (9,0%) | ×1,014 (0,40) | ×1,023 |
| Keno | 297.367 | 10^0,18 / 10^-0,83 | không | 99,2% | logistic (0,1%) | ×1,000 (0,28) | ×1,000 |
| Bingo18 | 105.563 | 10^0,12 / 10^-0,65 | không | 98,4% | dirichlet_pooled_a1000 (0,4%) | ×1,003 (0,76) | ×1,000 |
| Max 3D / Max 3D+ | 1.139 | 10^2,57 / 10^1,67 | **có** | 5,7% | tilt[đơn vị=6]+0.10 (73,8%) | ×1,039 (3,39) | ×1,075 |
| Max 3D Pro | 786 | 10^2,84 / 10^1,45 | **có** | 12,0% | tilt[đơn vị=6]+0.10 (78,1%) | ×1,026 (1,87) | ×1,070 |
| Max 4D (đã ngừng) | 722 | 10^0,72 / 10^-0,15 | không | 32,7% | tilt[trăm=8]-0.10 (16,9%) | ×1,028 (1,25) | ×1,014 |

**Kiểm tra trên dữ liệu mô phỏng** (200 lịch sử mỗi loại, 500 kỳ): máy quay công bằng 6/45 → e-value vượt 20 ở 1,5% số lịch sử; kiểu Max 3D → 2,0% (giới hạn lý thuyết 5%). Khi chữ số 6 hàng đơn vị ra 11% thay vì 10% (cỡ lệch thấy ở Max 3D), mô hình tìm ra ở 88% số lịch sử dài 1.139 kỳ và 52% số lịch sử dài 786 kỳ (100 lịch sử mỗi loại).

## Kỳ tới theo từng sản phẩm

### Mega 6/45 — kỳ #1570

Mô hình tự học CHƯA tìm thấy tín hiệu vượt ngẫu nhiên (e-value cao nhất 10^0,20 < 20): mọi bộ số vẫn có cùng xác suất; các hệ số × dưới đây là mô hình tự đánh giá, chưa được dữ liệu xác nhận. Bộ đề xuất 03 14 15 17 38 39: P(trúng cả bộ) theo mô hình 1/7.973.677 (×1,021; ngẫu nhiên 1/8.145.060).

- Xác suất cao nhất: 39 (13,372%, ×1,0029), 3 (13,370%, ×1,0027), 15 (13,365%, ×1,0024), 14 (13,363%, ×1,0022), 17 (13,361%, ×1,0021), 38 (13,361%, ×1,0021) — ngẫu nhiên 13,33%.
- Bộ đề xuất 03 14 15 17 38 39: P(trúng cả bộ) 1/7.973.677 theo mô hình, 1/8.145.060 ngẫu nhiên.

### Power 6/55 — kỳ #1406

Mô hình tự học CHƯA tìm thấy tín hiệu vượt ngẫu nhiên (e-value cao nhất 10^0,33 < 20): mọi bộ số vẫn có cùng xác suất; các hệ số × dưới đây là mô hình tự đánh giá, chưa được dữ liệu xác nhận. Bộ đề xuất 12 20 22 30 40 50: P(trúng cả bộ) theo mô hình 1/26.833.707 (×1,080; ngẫu nhiên 1/28.989.675).

- Xác suất cao nhất: 12 (11,046%, ×1,0126), 20 (10,998%, ×1,0081), 22 (10,995%, ×1,0079), 50 (10,988%, ×1,0072), 30 (10,984%, ×1,0068), 40 (10,979%, ×1,0064) — ngẫu nhiên 10,91%.
- Bộ đề xuất 12 20 22 30 40 50: P(trúng cả bộ) 1/26.833.707 theo mô hình, 1/28.989.675 ngẫu nhiên.

### Lotto 5/35 — kỳ #921

Mô hình tự học CHƯA tìm thấy tín hiệu vượt ngẫu nhiên (e-value cao nhất 10^0,12 < 20): mọi bộ số vẫn có cùng xác suất; các hệ số × dưới đây là mô hình tự đánh giá, chưa được dữ liệu xác nhận. Bộ đề xuất 10 20 22 28 34: P(trúng cả bộ) theo mô hình 1/311.055 (×1,044; ngẫu nhiên 1/324.632); mạnh nhất là 11 ở vị trí số đặc biệt (×1,017); số đề xuất 11 ×1,017.

- Xác suất cao nhất: 34 (14,613%, ×1,0229), 22 (14,345%, ×1,0041), 28 (14,334%, ×1,0034), 10 (14,333%, ×1,0033), 20 (14,328%, ×1,0029), 15 (14,327%, ×1,0029) — ngẫu nhiên 14,29%.
- Bộ đề xuất 10 20 22 28 34: P(trúng cả bộ) 1/311.055 theo mô hình, 1/324.632 ngẫu nhiên.
- số đặc biệt: 11 (8,48%, ×1,017), 4 (8,41%, ×1,009), 6 (8,39%, ×1,007)
- Số đề xuất: 11 (×1,017), 4 (×1,009), 6 (×1,007), 5 (×1,006), 8 (×0,999)

### Keno — kỳ #297783

Mô hình tự học CHƯA tìm thấy tín hiệu vượt ngẫu nhiên (e-value cao nhất 10^0,18 < 20): mọi bộ số vẫn có cùng xác suất; các hệ số × dưới đây là mô hình tự đánh giá, chưa được dữ liệu xác nhận. Vé Keno tốt nhất theo mô hình: bậc 10, RTP 0,5775 (ngẫu nhiên 0,5775). Mọi cửa vẫn có kỳ vọng âm (RTP < 1).

- Xác suất cao nhất: 11 (25,002%, ×1,0001), 70 (25,000%, ×1,0000), 3 (25,000%, ×1,0000), 41 (25,000%, ×1,0000), 59 (25,000%, ×1,0000), 71 (25,000%, ×1,0000) — ngẫu nhiên 25,00%.
- Keno, vé tốt nhất mỗi bậc theo mô hình (RTP mô hình / ngẫu nhiên): bậc 1: 0,5000 / 0,5000; bậc 2: 0,5412 / 0,5411; bậc 3: 0,5551 / 0,5550; bậc 4: 0,5515 / 0,5514; bậc 5: 0,5492 / 0,5491; bậc 6: 0,5446 / 0,5445; bậc 7: 0,5513 / 0,5512; bậc 8: 0,5468 / 0,5468; bậc 9: 0,5473 / 0,5473; bậc 10: 0,5775 / 0,5775.

### Bingo18 — kỳ #189250

Mô hình tự học CHƯA tìm thấy tín hiệu vượt ngẫu nhiên (e-value cao nhất 10^0,12 < 20): mọi bộ số vẫn có cùng xác suất; các hệ số × dưới đây là mô hình tự đánh giá, chưa được dữ liệu xác nhận. Mạnh nhất là 3 ở vị trí bóng 1 (×1,000); số đề xuất 3-3-5 ×1,000; cửa tốt nhất theo mô hình: Lớn (12–18), RTP 0,5625 (ngẫu nhiên 0,5625). Mọi cửa vẫn có kỳ vọng âm (RTP < 1).

- bóng 1: 3 (16,67%, ×1,000), 5 (16,67%, ×1,000), 6 (16,67%, ×1,000)
- bóng 2: 3 (16,67%, ×1,000), 2 (16,67%, ×1,000), 5 (16,67%, ×1,000)
- bóng 3: 5 (16,67%, ×1,000), 2 (16,67%, ×1,000), 6 (16,67%, ×1,000)
- Số đề xuất: 3-3-5 (×1,000), 3-3-2 (×1,000), 3-2-5 (×1,000), 3-3-6 (×1,000), 3-5-5 (×1,000)
- Lớn (12–18): RTP 0,5625 theo mô hình, 0,5625 ngẫu nhiên.
- Nhỏ (3–9): RTP 0,5625 theo mô hình, 0,5625 ngẫu nhiên.
- Tổng 14: RTP 0,5556 theo mô hình, 0,5556 ngẫu nhiên.

### Max 3D / Max 3D+ — kỳ #1140

Mô hình tự học ĐÃ tìm thấy độ lệch có ý nghĩa thống kê so với máy quay công bằng (e-value cao nhất 10^2,57 ≥ 20, α = 5%); lợi thế nhỏ, xem RTP. Mạnh nhất là 6 ở vị trí đơn vị (×1,075); số đề xuất 426 ×1,084; cửa tốt nhất theo mô hình: Max 3D, số 426 (RTP mô hình ≈ RTP × hệ số), RTP 0,5908 (ngẫu nhiên 0,5451). Mọi cửa vẫn có kỳ vọng âm (RTP < 1).

E-value vượt 20 lần đầu ở kỳ #961 (2025-08-08); từ đó trên ngưỡng ở 81% trong 179 kỳ, liên tục trên ngưỡng từ #1032 (2026-01-21). Cao nhất 10^2,57 ở #1104 (2026-07-10), nay 10^1,67. Mức giảm từ đỉnh (0,90) không chứng tỏ độ lệch yếu đi: với một độ lệch *không đổi* cùng cỡ, 17% lịch sử mô phỏng giảm ít nhất chừng ấy.

- trăm: 4 (10,03%, ×1,003), 1 (10,01%, ×1,001), 3 (10,01%, ×1,001)
- chục: 2 (10,06%, ×1,006), 5 (10,04%, ×1,004), 8 (10,01%, ×1,001)
- đơn vị: 6 (10,75%, ×1,075), 9 (9,96%, ×0,996), 0 (9,96%, ×0,996)
- Số đề xuất: 426 (×1,084), 456 (×1,083), 126 (×1,082), 326 (×1,082), 026 (×1,082)
- Max 3D, số 426 (RTP mô hình ≈ RTP × hệ số): RTP 0,5908 theo mô hình, 0,5451 ngẫu nhiên.

### Max 3D Pro — kỳ #787

Mô hình tự học ĐÃ tìm thấy độ lệch có ý nghĩa thống kê so với máy quay công bằng (e-value cao nhất 10^2,84 ≥ 20, α = 5%); lợi thế nhỏ, xem RTP. Mạnh nhất là 6 ở vị trí đơn vị (×1,070); số đề xuất 316 ×1,070; cửa tốt nhất theo mô hình: Max 3D Pro, cặp 316 616 (RTP mô hình ≤ RTP × hệ số₁ × hệ số₂), RTP 0,6265 (ngẫu nhiên 0,5469). Mọi cửa vẫn có kỳ vọng âm (RTP < 1).

E-value vượt 20 lần đầu ở kỳ #316 (2023-09-23); từ đó trên ngưỡng ở 47% trong 471 kỳ, liên tục trên ngưỡng từ #670 (2026-01-01). Cao nhất 10^2,84 ở #359 (2024-01-02), nay 10^1,45. Mức giảm từ đỉnh (1,39) khá hiếm nếu độ lệch không đổi (chỉ 3% lịch sử mô phỏng giảm chừng ấy): gợi ý độ lệch yếu hơn sau đỉnh — chỉ là gợi ý, vì mốc đỉnh được chọn sau khi xem dữ liệu.

- trăm: 6 (10,00%, ×1,000), 3 (10,00%, ×1,000), 5 (10,00%, ×1,000)
- chục: 1 (10,00%, ×1,000), 6 (10,00%, ×1,000), 8 (10,00%, ×1,000)
- đơn vị: 6 (10,70%, ×1,070), 1 (9,93%, ×0,993), 0 (9,93%, ×0,993)
- Số đề xuất: 316 (×1,070), 616 (×1,070), 816 (×1,070), 516 (×1,070), 366 (×1,070)
- Max 3D Pro, cặp 316 616 (RTP mô hình ≤ RTP × hệ số₁ × hệ số₂): RTP 0,6265 theo mô hình, 0,5469 ngẫu nhiên.

### Max 4D (đã ngừng) — kỳ #None

Mô hình tự học CHƯA tìm thấy tín hiệu vượt ngẫu nhiên (e-value cao nhất 10^0,72 < 20): mọi bộ số vẫn có cùng xác suất; các hệ số × dưới đây là mô hình tự đánh giá, chưa được dữ liệu xác nhận. Mạnh nhất là 1 ở vị trí trăm (×1,014); số đề xuất 6107 ×1,029.

- nghìn: 6 (10,05%, ×1,005), 0 (10,04%, ×1,004), 4 (10,03%, ×1,003)
- trăm: 1 (10,14%, ×1,014), 4 (10,04%, ×1,004), 0 (10,03%, ×1,002)
- chục: 0 (10,03%, ×1,003), 5 (10,02%, ×1,002), 9 (10,01%, ×1,001)
- đơn vị: 7 (10,06%, ×1,006), 6 (10,04%, ×1,004), 0 (10,00%, ×1,000)
- Số đề xuất: 6107 (×1,029), 6157 (×1,029), 0107 (×1,028), 6197 (×1,028), 0157 (×1,028)

## Đọc kết quả thế nào

- **Không có bằng chứng** (Mega, Power, Lotto, Keno, Bingo18, Max 4D): trọng số dồn về chuyên gia "ngẫu nhiên" và các giả thuyết lệch nhẹ (Keno, Bingo18 — nhiều dữ liệu nhất — 98–99% cho "ngẫu nhiên"); các hệ số × mô hình tự cho (thường ×1,00x–×1,02) là dao động của dữ liệu, không phải lợi thế. Chấm lùi trên toàn bộ lịch sử: lựa chọn của mô hình trùng trong phạm vi ngẫu nhiên (|z| < 2).
- **Có bằng chứng** (Max 3D, Max 3D Pro): mô hình chỉ ra chữ số 6 hàng đơn vị mà không được lập trình để tìm nó (giả thuyết lệch được đặt đều cho mọi chữ số ở mọi vị trí). Nhưng nó chạy trên cùng dữ liệu đã cho phát hiện ở mục 1.6, nên đây là cùng bằng chứng nhìn theo cách khác, không phải xác nhận độc lập; trên Max 4D nó không thấy độ lệch này. Lợi thế ≈ ×1,075 cho chữ số 6 (×1,08 cho số đề xuất, phần nhỉnh thêm từ hai chữ số kia chưa có bằng chứng), đưa RTP Max 3D từ 0,545 lên ≈ 0,59: **vẫn mất trung bình ~41% tiền vé**.
- Thành tích thật chỉ được tính từ các dự báo ghi *trước* kỳ quay (`forecast next` → `forecast update` → `forecast scoreboard`). Sổ là tệp cục bộ để tự theo dõi, không phải bằng chứng chống sửa; với Keno/Bingo18, hãy đồng bộ ngay trước khi dự báo.

## Các cách chọn số phổ biến, xem như mô hình xác suất (Keno, 297 nghìn kỳ)

log10 tỷ số hợp lý so với máy quay công bằng: mô hình gán xác suất cho kết quả thật kém máy công bằng bao nhiêu bậc 10. Đây là độ khớp của *mô hình*; một vé chọn theo số nóng vẫn có xác suất trúng như mọi vé khác nếu máy công bằng. Độ lớn chủ yếu phản ánh mô hình nghiêng mạnh đến đâu.

| Mô hình | log10 tỷ số hợp lý |
|---|---:|
| hot_ewma10 | -33.309,5 |
| overdue | -22.152,3 |
| repeat | -7.451,6 |
| cold_all_a50 | -160,1 |
| hot_all_a50 | -131,1 |
| logistic | -51,8 |
