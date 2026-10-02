# Vietlott v3.3 — dữ liệu chính thức, hành vi người chơi, EV, bao, độ phủ, chia giải

## 1. Dữ liệu chính thức (trang chi tiết kỳ quay vietlott.vn)

| Game | Kỳ quay | Kỳ có số người trúng từng giải | Kỳ có giá trị Jackpot | So với dữ liệu v3.1 (bên thứ ba) |
|---|---:|---:|---:|---|
| Mega 6/45 | 1569 (#1 → #1569, 2026-09-30) | 1569 | 1569 | 161 kỳ chung: số người trúng khớp 161; +197 kỳ mới |
| Power 6/55 | 1405 (#1 → #1405, 2026-10-01) | 1405 | 1405 | 1394 kỳ chung: số người trúng khớp 1381; Jackpot khớp 1353/1376 (lệch lớn nhất ×6.2); +1 kỳ mới |
| Lotto 5/35 | 920 (#1 → #920, 2026-10-01) | 920 | 920 | 917 kỳ chung: số người trúng khớp 917; +2 kỳ mới |

| Sản phẩm | Kỳ quay | Từ – đến | Mã kỳ thiếu trong khoảng | Nguồn |
|---|---:|---|---:|---|
| Keno | 297,367 | 2019-08-23 → 2026-10-02 | 415 | vietlott.vn/list+nhanaz+v130 |
| Bingo18 | 105,563 | 2024-12-03 → 2026-10-02 | 564 | vietlott.vn/list+nhanaz+v130 |
| Max 3D / 3D+ | 1,139 | 2019-04-22 → 2026-09-30 | 0 | vietlott.vn/detail+nhanaz+vietlott.vn/list+v130 |
| Max 3D Pro | 786 | 2021-09-14 → 2026-10-01 | 0 | vietlott.vn/detail+nhanaz+vietlott.vn/list+v130 |
| Max 4D (đã ngừng) | 722 | 2016-11-19 → 2021-08-31 | 0 | nhanaz+v130 |

Từ v3.2 dữ liệu giải thưởng Mega/Power/Lotto là bảng giải trên trang chi tiết từng kỳ của vietlott.vn (giá trị Jackpot × số vé trúng = quỹ Jackpot kỳ đó). Dữ liệu Power 6/55 của v3.1 sai ở một số kỳ (xem cột cuối) và đã được thay. Từ v3.3, kỳ còn thiếu trong bản ghi chính thức được bổ sung từ kho cộng đồng NhanAZ-Data/vietlott-research (Lotto #920, nguồn xosominhngoc.net.vn, đánh dấu `secondary_source`); khi hai nguồn lệch nhau, bản ghi chính thức được giữ (vd. Jackpot kỳ chia giải Lotto #874: 30,22 tỷ chính thức so với 35,96 tỷ ở trang phụ) — chi tiết `data/seed/DATA_MERGE_REPORT.json`.

## 2. Thị trường & hành vi người chơi

### Power 6/55
- Tỷ lệ doanh thu vào quỹ Jackpot + giải cố định: s = 0.418 (method of moments on tiers 2–3 (identity E[y_m] = N·H_m)).
- Vé bán mỗi kỳ: trung vị 1,033,659 (p10 653,937 – p90 2,901,799); jackpot accounting identity ΔJ + F = s·R.
- Doanh số theo Jackpot: độ co giãn 0.84 ± 0.02 tại Jackpot trung vị, độ cong +0.011 ± 0.033 (HAC); dự báo 50 tỷ → 1.26 tr vé, 150 tỷ → 3.18 tr vé.
- Quick pick ≈ 51% ± 14%; số được chọn nhiều nhất [5, 9, 8, 7, 12, 3, 19, 11], ít nhất [46, 50, 47, 55, 49, 40, 41, 53].
- Kiểm tra ngoài mẫu (281 kỳ cuối): +0.201 log-lik/kỳ so với đám đông chọn đều, p = 7.1e-05.
- Kiểm tra mức (không cần doanh số): tương quan 0.43, hệ số góc 0.81, p hoán vị 0.0002.
- Độ phổ biến của vé ngẫu nhiên theo mô hình: p01 0.66, p10 0.75, p50 0.94, p90 1.28, p99 1.80.
- Tier estimates of s agree (first: 0.417, second: 0.419, third: 0.418), validating the identity.
- Sales elasticity to the advertised jackpot: 0.84 ± 0.02 at the median jackpot, curvature +0.011 ± 0.033 (HAC).

### Mega 6/45
- Tỷ lệ doanh thu vào quỹ Jackpot + giải cố định: s = 0.465 (method of moments on tiers 2–3 (identity E[y_m] = N·H_m)).
- Vé bán mỗi kỳ: trung vị 671,883 (p10 462,055 – p90 1,819,363); jackpot accounting identity ΔJ + F = s·R.
- Doanh số theo Jackpot: độ co giãn 0.65 ± 0.03 tại Jackpot trung vị, độ cong +0.198 ± 0.050 (HAC); dự báo 50 tỷ → 1.54 tr vé, 150 tỷ → 5.79 tr vé.
- Quick pick ≈ 51% ± 14%; số được chọn nhiều nhất [5, 9, 8, 7, 12, 3, 19, 11], ít nhất [40, 41, 31, 42, 44, 34, 30, 1].
- Kiểm tra ngoài mẫu (toàn bộ 1569 kỳ — mô hình chuyển từ Power, không ước lượng trên game này): +0.001 log-lik/kỳ so với đám đông chọn đều, p = 0.36.
- Chọn mô hình đám đông: chuyển từ Power 6/55 (mô hình riêng: ngoài mẫu p = 0.61, kiểm tra mức 0.24; chuyển từ Power: kiểm tra mức 0.30). Không mô hình nào vượt đám đông chọn đều trong kiểm định likelihood, nên con số ×r của Mega chỉ mang tính tham khảo.
- Kiểm tra mức (không cần doanh số): tương quan 0.30, hệ số góc 0.74, p hoán vị 0.0002.
- Độ phổ biến của vé ngẫu nhiên theo mô hình: p01 0.70, p10 0.79, p50 0.95, p90 1.23, p99 1.71.
- Tier estimates of s agree (first: 0.462, second: 0.464, third: 0.465), validating the identity.
- Sales elasticity to the advertised jackpot: 0.65 ± 0.03 at the median jackpot, curvature +0.198 ± 0.050 (HAC).
- Crowd model: Mega's own fit does not beat a uniform crowd out of sample (holdout p = 0.61); the Power 6/55 crowd transferred to Mega explains third-prize winner levels better (corr 0.30 vs 0.24), so the transferred model is used for popularity and EV.

### Lotto 5/35
- Vé bán mỗi kỳ: trung vị 170,230 (p10 115,693 – p90 266,546); NB profile from winner counts (E[y_g] = N·P_g(W_t)).
- Quick pick ≈ 47% ± 7%; số được chọn nhiều nhất [19, 9, 12, 8, 11, 25, 7, 20], ít nhất [1, 35, 4, 2, 29, 30, 31, 21].
- Kiểm tra ngoài mẫu (184 kỳ cuối): +3.143 log-lik/kỳ so với đám đông chọn đều, p = 2.8e-15.
- Kiểm tra mức (không cần doanh số): tương quan 0.23, hệ số góc 0.97, p hoán vị 0.0002.
- Độ phổ biến của vé ngẫu nhiên theo mô hình: p01 0.65, p10 0.75, p50 0.94, p90 1.22, p99 1.49.
- Sales are estimated from winner counts (no sales figures are published per draw). Official jackpot series: 17 rolldowns executed, 6 pre-empted; accrual 0.078 (slow) / 0.375 (fast) per đồng; slow phase refunds ≈5.63 tỷ per jackpot restart.
- Reconstruction without the jackpot series (v3 method, kept as a check): 21 rolldown draws announced, 17 executed; jackpot accrual c = 0.155 per đồng of sales (rolldown day 0.48); mean realised RTP of rolldown draws 1.48 (pre-tax), falling as rolldown sales grow. Official series: mean realised RTP 1.42.

## 3. EV với mô hình đã hiệu chỉnh (sau thuế, 1 lượt 10.000 đ)

(×r = độ phổ biến so với vé chọn đều; số vé bán lấy từ mô hình doanh số khi có)

| Game · Jackpot · vé bán (mô hình) | Vé mẫu hình | Vé "ngày sinh" | Vé ít người chọn (tối ưu) | Jackpot hòa vốn (vé ít người chọn, có phản ứng doanh số) |
|---|---|---|---|---|
| Mega 6/45 · 30.0 tỷ · 0.98 tr (sales_model) | RTP 22.8% (×29.6) | RTP 44.0% (×1.52) | [30, 31, 40, 41, 42, 44]: RTP 45.7% (×0.59) | 86.3 tỷ |
| Mega 6/45 · 100.0 tỷ · 3.36 tr (sales_model) | RTP 22.7% (×29.6) | RTP 95.8% (×1.52) | [30, 31, 40, 41, 42, 44]: RTP 111.8% (×0.59) | 86.3 tỷ |
| Mega 6/45 · 158.8 tỷ · 6.28 tr (sales_model) | RTP 21.4% (×29.6) | RTP 117.1% (×1.52) | [30, 31, 40, 41, 42, 44]: RTP 154.8% (×0.59) | 86.3 tỷ |
| Power 6/55 · 60.0 tỷ · 1.46 tr (sales_model) | RTP 24.8% (×40.2) | RTP 38.7% (×2.13) | [40, 41, 46, 49, 50, 55]: RTP 39.7% (×0.56) | 274.8 tỷ |
| Power 6/55 · 200.0 tỷ · 4.07 tr (sales_model) | RTP 27.1% (×40.2) | RTP 78.6% (×2.13) | [40, 41, 46, 49, 50, 55]: RTP 85.8% (×0.56) | 252.9 tỷ |
| Lotto 5/35 · 6.0 tỷ · 0.17 tr (default) | RTP 27.7% (×14.0) | RTP 30.7% (×1.80) | [1, 2, 33, 34, 35] + 1: RTP 31.1% (×0.52) | 36.2 tỷ ở doanh số cố định — vượt ngưỡng chia giải 12 tỷ ⇒ kỳ thường luôn âm |
| Lotto 5/35 · 11.0 tỷ · 0.17 tr (default) | RTP 36.4% (×14.0) | RTP 41.8% (×1.80) | [1, 2, 33, 34, 35] + 1: RTP 42.5% (×0.52) | 36.2 tỷ ở doanh số cố định — vượt ngưỡng chia giải 12 tỷ ⇒ kỳ thường luôn âm |

## 4. Chơi bao vs vé lẻ (chính xác, không mô phỏng)

| Game | Kiểu | Số lượt | Chi phí | P(trúng ≥ 1 giải) | Cùng tiền, vé lẻ độc lập | Lượt trúng kỳ vọng |
|---|---|---:|---:|---:|---:|---:|
| Mega 6/45 | Bao 5 | 40 | 400,000 | 12.48% | 61.90% | 0.95 |
| Mega 6/45 | Bao 7 | 7 | 70,000 | 3.94% | 15.54% | 0.17 |
| Mega 6/45 | Bao 8 | 28 | 280,000 | 5.94% | 49.11% | 0.67 |
| Mega 6/45 | Bao 10 | 210 | 2,100,000 | 11.29% | 99.37% | 5.01 |
| Power 6/55 | Bao 5 | 50 | 500,000 | 8.64% | 48.88% | 0.67 |
| Power 6/55 | Bao 7 | 7 | 70,000 | 2.23% | 8.97% | 0.09 |
| Lotto 5/35 | Bao 4 + 1 số ĐB | 31 | 310,000 | 16.48% | 95.63% | 2.98 |
| Lotto 5/35 | Bao 6 + 1 số ĐB | 6 | 60,000 | 10.75% | 45.44% | 0.58 |
| Lotto 5/35 | Bao 6 số đặc biệt | 6 | 60,000 | 50.69% | 45.44% | 0.58 |

Kỳ vọng tiền thưởng trên mỗi đồng là như nhau; bao gom giải thành từng cụm nên xác suất trúng *ít nhất một* giải thấp hơn nhiều so với cùng số tiền mua vé lẻ.

## 5. Danh mục tối đa hóa P(trúng ít nhất 1 giải)

| Game | Số vé | P(≥1 giải) tối ưu [95%] | Vé ngẫu nhiên | Cận trên B·p | Số giải kỳ vọng (mọi danh mục) |
|---|---:|---:|---:|---:|---:|
| Mega 6/45 | 10 | 23.0% [22.8%, 23.1%] | 21.4% | 23.8% | 0.24 |
| Mega 6/45 | 28 | 55.2% [54.9%, 55.4%] | 49.0% | 66.7% | 0.67 |
| Power 6/55 | 10 | 13.0% [12.9%, 13.2%] | 12.6% | 13.3% | 0.13 |
| Power 6/55 | 28 | 33.9% [33.7%, 34.1%] | 31.5% | 37.3% | 0.37 |
| Lotto 5/35 | 6 | 54.1% [53.9%, 54.4%] | 45.5% | 57.6% | 0.58 |
| Lotto 5/35 | 12 | 100.0% [100.0%, 100.0%] | 70.2% | 100.0% | 1.15 |

## 6. Lotto 5/35 — chia giải Độc đắc (chuỗi Jackpot chính thức)

17 kỳ chia giải đã thực hiện, 6 lần công bố nhưng có người trúng Độc đắc trước. RTP thực tế trung bình của kỳ chia giải 1.42 (trước thuế); xu hướng giảm theo thời gian (Spearman ρ = -0.90, p = 1e-06) vì càng về sau càng nhiều người dồn vào kỳ chia giải.

**Hai pha tích lũy Jackpot** (suy ra từ giá trị công bố, không phải quy định đã công bố):

- Pha chậm: Jackpot tăng 0.078 × doanh thu (cùng giải cố định: 0.251).
- Pha nhanh: 0.375 × doanh thu (cùng giải cố định: 0.548).
- Chuyển pha ở mức Jackpot trung vị 7.9 tỷ (7.3 tỷ – 10.1 tỷ).
- Phần chênh lệch trong pha chậm ≈ 5.6 tỷ cho mỗi lần Jackpot khởi động lại (28 đoạn) — khớp với việc bù lại 6 tỷ khởi điểm.

**Dự báo** (sau kỳ #920, 2026-10-01): Jackpot 7.6 tỷ, pha slow, còn ≈ 6.6 kỳ đến ngưỡng 12 tỷ → khoảng 3–5 ngày nữa (kỳ 21:00 của ngày sau khi vượt ngưỡng), nếu không ai trúng Độc đắc trước.
Kiểm định lại dự báo trên 48 thời điểm: sai số tuyệt đối trung vị 2.3 kỳ, 62% trong ±3 kỳ, Spearman 0.87.

| Kỳ | Ngày | Jackpot chia (công bố) | Vé bán (ước tính) | ×doanh số | RTP thực (trước thuế) | RTP mô hình (sau thuế) |
|---|---|---:|---:|---:|---:|---:|
| #38 | 2025-07-17 | 16.3 tỷ | 0.72 tr | 3.2 | 2.44 | 2.32 |
| #54 | 2025-07-25 | 19.6 tỷ | 1.12 tr | 4.7 | 1.94 | 1.82 |
| #70 (Độc đắc trúng trước khi chia) | 2025-08-02 | 19.4 tỷ | 1.37 tr | 5.6 | — | — |
| #96 | 2025-08-15 | 18.9 tỷ | 1.28 tr | 5.6 | 1.64 | 1.56 |
| #146 | 2025-09-09 | 21.4 tỷ | 1.62 tr | 8.3 | 1.46 | 1.41 |
| #168 | 2025-09-20 | 18.6 tỷ | 1.42 tr | 7.7 | 1.48 | 1.40 |
| #192 | 2025-10-02 | 22.1 tỷ | 1.69 tr | 9.3 | 1.47 | 1.40 |
| #266 (Độc đắc trúng trước khi chia) | 2025-11-08 | 21.4 tỷ | 1.80 tr | 9.5 | — | — |
| #290 | 2025-11-20 | 19.4 tỷ | 1.72 tr | 11.0 | 1.28 | 1.23 |
| #340 | 2025-12-15 | 20.6 tỷ | 1.65 tr | 9.1 | 1.44 | 1.34 |
| #392 (Độc đắc trúng trước khi chia) | 2026-01-10 | 19.9 tỷ | 1.89 tr | 11.1 | — | — |
| #430 | 2026-01-29 | 20.8 tỷ | 1.93 tr | 11.5 | 1.23 | 1.18 |
| #462 | 2026-02-14 | 23.7 tỷ | 2.25 tr | 10.4 | 1.24 | 1.15 |
| #492 | 2026-03-01 | 24.4 tỷ | 2.17 tr | 9.9 | 1.30 | 1.22 |
| #523 (Độc đắc trúng trước khi chia) | 2026-03-17 | 14.6 tỷ | 0.32 tr | 1.6 | — | — |
| #546 | 2026-03-28 | 21.5 tỷ | 2.07 tr | 11.3 | 1.20 | 1.14 |
| #584 | 2026-04-16 | 22.4 tỷ | 1.95 tr | 12.0 | 1.31 | 1.24 |
| #606 (Độc đắc trúng trước khi chia) | 2026-04-27 | 23.1 tỷ | 2.07 tr | 11.2 | — | — |
| #639 (Độc đắc trúng trước khi chia) | 2026-05-14 | 15.2 tỷ | 0.39 tr | 2.1 | — | — |
| #662 | 2026-05-25 | 21.3 tỷ | 2.01 tr | 12.4 | 1.26 | 1.16 |
| #734 | 2026-06-30 | 25.0 tỷ | 2.40 tr | 13.3 | 1.22 | 1.14 |
| #806 | 2026-08-05 | 25.0 tỷ | 2.78 tr | 11.7 | 1.10 | 1.01 |
| #874 | 2026-09-08 | 30.2 tỷ | 3.32 tr | 11.1 | 1.08 | 1.01 |

Kiểm tra phương pháp tái dựng của v3.1 (không có chuỗi Jackpot, nhận diện qua doanh số tăng đột biến): khớp 21/23 sự kiện, bỏ sót [523, 639], sai số Jackpot trung vị 5.6% (lớn nhất 18.7%).
