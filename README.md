# Vietlott Quant Engine

Hệ thống phân tích định lượng cho **toàn bộ sản phẩm** xổ số tự chọn Vietlott — **Mega 6/45**, **Power 6/55**, **Lotto 5/35**, **Keno**, **Bingo18**, **Max 3D / 3D+**, **Max 3D Pro** và lưu trữ **Max 4D**: thu thập dữ liệu bất đồng bộ, bộ kiểm định ngẫu nhiên, mô hình Bayesian/Markov, mạng đồ thị (GCN), lý thuyết trò chơi chống trùng số (Anti-Popularity EV), bao lô tối ưu (covering design) và backtest walk-forward — kèm FastAPI, CLI, Docker.

**Rà soát VLM 03/10/2026:** [Audit Matrix và kết quả kiểm đếm](reports/VLM_AUDIT_2026-10-03.md), [vận hành crawler, cào bù và đối soát vé](docs/VLM_INTEGRITY.md). API `vlm` nằm trong `src/vlm/`; dữ liệu hiện còn thiếu 384 kỳ Keno và 83.654 kỳ Bingo18, nên chưa chứng nhận lịch sử đầy đủ đến hiện tại.

**ML và tự học sau kỳ quay:** [hướng dẫn CLI/API](docs/VLM_ML_FORECAST.md), [kiểm định dữ liệu thật](reports/VLM_ML_2026-10-03.md). Thêm GRU với backpropagation, logistic online, Random Forest và optional XGBoost/LightGBM; crawler tự gọi học sau khi lưu kết quả. `vlm-forecast next --product mega645 --top-n 10` trả xác suất tổ hợp, diagnostics và confidence có kiểm định. Benchmark mới chưa cải thiện log-loss so với fair; giữ xác suất triển khai theo fair cho đến khi có bằng chứng live. Snapshot mới có 538 gap Keno, 83.874 ID Bingo18 còn thiếu từ kỳ đầu và 11 date anomaly Keno, nên các số đếm audit seed ở dòng trên là mốc lịch sử.

> **Cài đặt nhanh** (chi tiết: [docs/INSTALL.md](docs/INSTALL.md))
> - **Windows:** giải nén, bấm đúp `install.cmd` (chưa có Python: `install.cmd -InstallPython`). Sau đó, trong thư mục này: `vietlott forecast next` (Command Prompt) hoặc `.\vietlott.cmd forecast next` (PowerShell).
> - **Linux / macOS:** `bash install.sh`, rồi `./vietlott forecast next`.
> - **Docker:** `docker compose up -d` → API tại http://localhost:8000/docs.
> - **GitHub (repo riêng):** đẩy mã lên, bật Pages (Source: GitHub Actions); workflow tự đồng bộ, học, ghi sổ dự báo và dựng lại trang sau mỗi luồng cập nhật kết quả; luồng đồng bộ tổng thể chạy hai lần mỗi ngày.

**v4.0 — repo riêng, bộ cài hoàn chỉnh** (tách khỏi VLA): gói Python `vietlott_engine` (`pip install -e .`, lệnh `vietlott`; không còn package `src` trùng tên với VLA), bộ cài `install.cmd` / `install.ps1` (Windows PowerShell 5.1 trở lên) và `install.sh` (Linux, macOS), lệnh `vietlott init` (nạp dữ liệu đi kèm, học bộ dự báo, kiểm tra) và `vietlott doctor`, sổ dự báo chỉ nhận dự báo ghi **trước** kỳ quay theo lịch quay chính thức, và GitHub Actions: CI (test trên Linux, Windows), chạy thử chính các bộ cài trên Linux / macOS / Windows, cập nhật tự động + trang tĩnh trên GitHub Pages (JSON cho VLA hoặc trang khác đọc), phát hành bằng tag.


**v2** bổ sung tầng suy luận thống kê (`src/vietlott_engine/inference`): power/MDE và kiểm định tương đương (TOST), mô hình Bayes phân cấp, kiểm soát đa kiểm định Westfall–Young/Holm/BY, e-process kiểm định liên tục (anytime-valid), quét điểm gãy, Diebold–Mariano với HAC, SPA chống data-snooping; backtest suy luận theo cụm; bao lô có chứng nhận tối ưu bằng ILP; EV có khoảng bất định, phân phối tiền thưởng và Kelly.

**v3** thu thập **toàn bộ dữ liệu giải thưởng** công khai (số vé trúng từng hạng giải của từng kỳ, giá trị Jackpot Power), **hiệu chỉnh mô hình hành vi người chơi trên dữ liệu thật** (hồi quy nhị thức âm, sai số sandwich, kiểm tra ngoài mẫu, chuyển giao giữa các game), ước lượng **tỷ lệ trả thưởng, số vé bán từng kỳ và phản ứng doanh số theo Jackpot**, bổ sung **Lotto 5/35** (số đặc biệt 1–12, 2 kỳ/ngày, cơ chế chia giải Độc đắc), **chơi bao** (phân phối chính xác), **tối ưu danh mục để tăng xác suất trúng ít nhất một giải**, quy tắc trần 300 tỷ của Power, và **tái dựng toàn bộ 21 kỳ chia giải Lotto 5/35**. Kết quả: mục 1.4 và `reports/vietlott_v3.md`.

**v3.1 — chơi bao**: danh mục bao chính thức của từng sản phẩm (Mega/Power Bao 5, 7–15, 18; Lotto Bao 4, 6–15, bao số đặc biệt 2–12; Max 3D Pro bao bộ số / bao nhiều bộ số; Max 3D/3D+ đảo số, bao vị trí), bảng giải thưởng kiểu Vietlott khớp các ví dụ đã công bố, sửa lỗi Jackpot khi nhiều lượt của một vé bao cùng trúng (vé nhận nhiều *phần* của một pot), thuế tính trên tổng vé bao, EV khi có người trúng chung, so sánh bao / bao rút gọn / vé lẻ dàn đều / quick pick, và mô-đun xác suất chính xác cho họ Max 3D. Kết quả: mục 1.5 và `reports/bao_vietlott.md`.

**v3.2 — truy cập vietlott.vn & toàn bộ sản phẩm**: crawler vietlott.vn viết lại cho cả 7 sản phẩm (AjaxPro, cookie khởi tạo, trang chi tiết kỳ quay có bảng giải và PDF biên bản), phân biệt *website chặn IP ngoài Việt Nam* với *proxy mạng chặn* và không thử lại vô ích; dữ liệu giải thưởng chuyển sang **bảng giải chính thức trên trang chi tiết từng kỳ** (Mega 1.569, Power 1.405, Lotto 919 kỳ — đủ số người trúng từng hạng và Jackpot), thay dữ liệu bên thứ ba (Power v3.1 sai ở 13 kỳ số người trúng và 23 kỳ Jackpot); thêm Keno (86.272 kỳ), Bingo18 (92.759 kỳ), Max 3D (1.139 kỳ), Max 3D Pro (786 kỳ) với kho dữ liệu + đồng bộ + API, xác suất chính xác và RTP mọi cửa Keno/Bingo18, bộ kiểm định ngẫu nhiên riêng cho từng sản phẩm; phát hiện **cơ chế tích lũy hai pha của Jackpot Lotto 5/35** và dự báo kỳ chia giải tiếp theo (kiểm định lại: sai số trung vị 2,3 kỳ); ước lượng lại doanh số Mega từ chuỗi Jackpot thật (trung vị 672 nghìn vé/kỳ — v3 cao hơn ~43%). Kết quả: mục 1.6, `reports/products.md`, `reports/vietlott_v3.md`.

**v3.3 — nguồn dự phòng khi không vào được vietlott.vn, kết hợp bộ Vietlott Quant Engine 1.3.0**: chuỗi `--source auto` (vietlott.vn → kho cộng đồng [NhanAZ-Data/vietlott-research](https://github.com/NhanAZ-Data/vietlott-research), Keno/Bingo18 cập nhật ~10 phút/lần → bản sao GitHub → snapshot 1.3.0) cho mọi lệnh đồng bộ, ghi rõ nguồn đã dùng; đọc snapshot Parquet của bộ 1.3.0; gộp và đối chiếu các nguồn (khớp 100% trên phần trùng, trừ một Jackpot Lotto ở trang phụ — giữ bản chính thức). Dữ liệu tăng: **Keno 297.367 kỳ từ 08/2019** (v3.2: 86.272), Bingo18 105.563 kỳ, Lotto #920, thêm **Max 4D** (722 kỳ, 2016–2021); 61 kỳ Keno/Bingo18 có trong dữ liệu thuộc danh sách 63 kỳ Vietlott tuyên bố *không xác nhận* và bị loại khỏi phân tích. Bảng giải Keno chuyển sang **bảng của trang chính thức** (bậc 10: 710.000 / 8.000.000 đ; cửa phụ đã xác minh). Lệch chữ số 6 hàng đơn vị lặp lại lần thứ ba trên Max 4D (p = 0,019, giả thuyết đặt trước). Kết quả: mục 1.7.

**v3.4 — vietlott.vn có Cloudflare "xác minh bạn là người"**: engine nhận ra trang thử thách Cloudflare và dừng ngay — không thử lại, không vượt. Thay vào đó, người dùng tự qua bước xác minh trong trình duyệt của mình và lưu trang (.mhtml, .html hoặc HAR); lệnh `products import-pages` đọc các tệp đó, **đối chiếu với dữ liệu đang có** (kèm cận trên tỷ lệ sai 95%), rồi mới nhập — kỳ lệch không bị ghi đè nếu không yêu cầu. `scripts/verification_checklist.py` chọn sẵn 58 trang đáng kiểm nhất (Lotto #920 và #874, ô giải Keno bậc 5 trùng 4 số, mẫu ngẫu nhiên Keno 2019–2022 và Bingo18 nguồn 'unknown', khoảng trống mã kỳ Keno). Kết quả: mục 1.8.

**v3.5 — dự báo tự học cho kỳ tiếp theo, cả 8 sản phẩm**: một hỗn hợp Bayes *fixed-share* gồm các chuyên gia ngẫu nhiên, số nóng / lạnh, gan, lặp kỳ trước, hồi quy logistic học trực tuyến, Dirichlet cho trò chữ số và một rổ giả thuyết lệch thưa. Sau mỗi kỳ, mô hình tự chỉnh trọng số theo mức mỗi chuyên gia đoán đúng, rồi cho xác suất từng số / chữ số kỳ tới, bộ đề xuất, RTP theo mô hình. Nó tự đo mình bằng một **e-value hợp lệ ở mọi thời điểm** và ghi mỗi dự báo vào sổ *trước* kỳ quay để chấm sau. Trên dữ liệu thật, nó chỉ ra đúng một tín hiệu — chữ số 6 hàng đơn vị của Max 3D và Max 3D Pro, trên cùng dữ liệu đã cho phát hiện ở mục 1.6 — và không thấy gì ở 6 sản phẩm còn lại, kể cả Max 4D. Kết quả: mục 1.9, `reports/forecast.md`.

> **Kết luận định lượng (đọc trước).** Trên lịch sử dùng cho chứng nhận v1/v2 (1.371 kỳ Mega 10/2017–09/2026, 1.402 kỳ Power; dữ liệu v3.2 có thêm 197 kỳ Mega đầu tiên từ 2016), không có kiểm định nào bác bỏ giả thuyết *các kỳ quay độc lập và đều*; không mô hình dự đoán nào (tần suất, gan, Bayesian, Markov, GCN) có kỹ năng ngoài mẫu. Hệ thống được thiết kế để **đo** điều đó thay vì giả định ngược lại. Đòn bẩy duy nhất có cơ sở toán học là **giảm số người trúng chung** khi trúng Jackpot (chọn bộ số ít người chơi) — nó tăng kỳ vọng tiền thưởng nhưng **không tăng xác suất trúng**, và kỳ vọng vẫn âm trừ khi Jackpot rất lớn. Bao lô chỉ thay đổi *phân phối* kết quả, không thay đổi kỳ vọng. Với các sản phẩm còn lại (v3.3): Keno (297 nghìn kỳ) và Bingo18 (106 nghìn kỳ) không có kiểm định nào bác bỏ; ngoại lệ duy nhất là **chữ số hàng đơn vị của các trò chữ số** — số 6 ra nhiều hơn ~10%, phát hiện trên Max 3D, xác nhận trên các kỳ Max 3D Pro độc lập (p ≈ 6·10⁻⁶) và lặp lại trên Max 4D (p = 0,019). Lệch có thật về thống kê nhưng chỉ đưa RTP của số có lợi nhất từ 0,55 lên ≈ 0,64 — vẫn âm. Bộ dự báo tự học v3.5, với giả thuyết lệch đặt đều cho mọi chữ số, chỉ ra đúng tín hiệu này trên Max 3D / Pro (cùng dữ liệu, không phải xác nhận độc lập), không thấy nó trên Max 4D, và không thấy gì khác (mục 1.9).
>
> **Tần suất trúng (v3).** Xác suất một vé trúng *bất kỳ* giải nào là hằng số của luật chơi — Mega 2,38% (1/42), Power 1,33% (1/75), Lotto 5/35 9,60% (1/10,4) — không bộ số nào thay đổi được. Điều tối ưu được là xác suất trúng *ít nhất một* giải **với cùng ngân sách**: dàn vé để chúng ít trùng nhau (Mega 28 vé: 55,2% so với 49,0% của 28 vé ngẫu nhiên và chỉ 5,9% nếu dồn vào một vé Bao 8 cùng giá), hoặc với Lotto phủ đủ 12 số đặc biệt (12 vé ⇒ chắc chắn trúng ít nhất giải khuyến khích 10.000 đ). Tổng tiền thưởng kỳ vọng của mọi danh mục cùng số vé là như nhau.
>
> **Khi nào kỳ vọng dương?** Theo mô hình đã hiệu chỉnh, chỉ trong hai tình huống: (1) Jackpot Mega/Power rất lớn và vé thuộc nhóm ít người chọn — với Jackpot Mega 158,8 tỷ của kỳ #01569, RTP mô hình ≈ 155% (6,3 triệu vé theo mô hình doanh số) nhưng tiêu chuẩn Kelly cần vốn ≈ 195 tỷ đ cho *một* vé, P(trúng bất kỳ giải nào) vẫn 2,4%; (2) kỳ chia giải Lotto 5/35 — 17 kỳ đã chia có RTP thực 1,08–2,44 (trước thuế, pot chính thức, doanh số ước lượng) nhưng giảm đều khi người chơi đổ vào (Spearman ρ = −0,90); kỳ gần nhất #874 (09/2026) còn 1,08 trước thuế ≈ 1,01 sau thuế. Keno, Bingo18 và Max 3D trả ≈ 50–58% cho hầu hết các cửa; theo bảng giải chính thức, cửa phụ Keno "Chẵn/Lẻ 11–12" trả 0,60 và "Hòa Chẵn–Lẻ" 0,41.

---

## 1. Kết quả trên dữ liệu thật

Dữ liệu: snapshot tới kỳ Mega #01569 (30/09/2026), Power #01405 (01/10/2026), Lotto 5/35 #00920 (01/10/2026), Max 3D #01139 (30/09/2026), Max 3D Pro #00786 (01/10/2026), Keno và Bingo18 tới 02/10/2026, Max 4D 2016–2021. Báo cáo đầy đủ trong `reports/` (mục 1.0–1.3 là v1/v2 trên Mega/Power; v3 ở mục 1.4; toàn bộ sản phẩm ở mục 1.6).

### 1.0 Chứng nhận công bằng — suy luận v2 (`vietlott inference`)

"Không bác bỏ H₀" chỉ có nghĩa khi kiểm định *đủ mạnh*. v2 trả lời trực tiếp: dữ liệu loại trừ được độ lệch cỡ nào, và độ lệch còn sót lại đáng bao nhiêu tiền.

| Câu hỏi | Mega 6/45 | Power 6/55 |
|---|---|---|
| Số lệch nhất có thật sự lệch? (Westfall–Young, FWER) | số 17, z = −1.73, p_adj = 0.98 | số 4, z = −2.48, p_adj = 0.56 |
| Độ lệch nhỏ nhất phát hiện được (power 80%) | 1 số nóng 37%, hoặc mọi số ±5.6% | 1 số nóng 43%, hoặc mọi số ±5.8% |
| Cần bao nhiêu kỳ để phát hiện 1 số nóng 5%? | ≈ 74.500 kỳ (~480 năm) | ≈ 101.900 kỳ (~650 năm) |
| TOST (giao–hợp, 95%): mọi số nằm trong k/n ± | 22.5% | 31.3% |
| Bayes phân cấp: độ phân tán thật giữa các số (cận trên 95%) | 2.5% của k/n | 4.1% của k/n |
| Bayes factor "công bằng" vs "lệch ≥ 5%" | 10^3.0 : 1 nghiêng về công bằng | 10^1.6 : 1 nghiêng về công bằng |
| RTP vé 6 số "tốt nhất" (pooled) vs vé bất kỳ (Jackpot tối thiểu, không chia) | 27.3% vs 27.0% | 28.9% vs 28.1% |
| RTP ở kịch bản lạc quan nhất còn tương thích với dữ liệu | 33.0% (Bayes) · 80.3% (TOST từng số) | 39.4% (Bayes) · 101% (TOST từng số) |
| E-process: tài sản lớn nhất khi "cá cược" vào lịch sử | 10^0.14 ≈ 1.4× (ngưỡng 20×) | 10^0.27 ≈ 1.9× (ngưỡng 20×) |
| Quét điểm gãy (197/200 cửa sổ, đã hiệu chỉnh quét) | p = 0.46 | p = 0.78 |

Cách đọc: với ~1.400 kỳ, dữ liệu **không đủ** để loại trừ một số đơn lẻ lệch tới ±20–30% (TOST từng số rất bảo thủ). Nhưng dưới giả định các số có thể hoán đổi (Bayes phân cấp), độ phân tán thật gần như chắc chắn dưới 3–4% của k/n; và ngay cả khi chơi đúng 6 số "tốt nhất" theo posterior, RTP chỉ nhích từ 27.0% lên 27.3%. Kết luận thống kê và kết luận tiền bạc trùng nhau: không có lợi thế khai thác được.

### 1.1 Kiểm định tính ngẫu nhiên (FDR 5%, Benjamini–Hochberg)

| Kiểm định | Mega 6/45 p | Power 6/55 p |
|---|---:|---:|
| χ² tần suất (đã hiệu chỉnh hiệp phương sai) | 0.880 | 0.314 |
| Shannon entropy (Monte Carlo 2.000 lần) | 0.876 | 0.304 |
| Số lẻ / số nhỏ (siêu bội chính xác) | 0.79 / 0.61 | 0.10 / 0.11 |
| Phân phối tổng (subset-sum chính xác) | 0.360 | 0.479 |
| Cặp số liên tiếp | 0.092 | 0.034 (q = 0.30) |
| Số lặp lại từ kỳ trước | 0.904 | 0.381 |
| Runs test / Ljung–Box Q(10) trên tổng | 0.55 / 0.26 | 0.85 / 0.47 |
| Phụ thuộc chuyển trạng thái (hoán vị thứ tự kỳ) | 0.17 | 0.94 |
| Cặp đồng xuất hiện có ý nghĩa (trên 990 / 1.485 cặp) | 0 | 0 |

- **Gap/“số gan”**: hazard P(về | đã vắng g kỳ) phẳng ≈ k/n (Mega ≈ 0.133, Power ≈ 0.109) ở mọi g ⇒ số “gan lâu” không có xác suất về cao hơn.
- **Dirichlet–Multinomial**: log Bayes factor (không đều vs đều) = −103.6 (Mega), −113.8 (Power): bằng chứng rất mạnh cho mô hình **đều**. Empirical-Bayes đẩy α₀ → ∞ (tức là “mọi số như nhau”). Mọi hệ số suy giảm λ đều cho điểm dự báo kém hơn baseline k/n.

### 1.2 Backtest walk-forward (5 vé/kỳ, 07/2019 → 09/2026, ~1.100 kỳ mỗi game)

z tính theo **cụm (mỗi kỳ là một cụm)** vì các vé của cùng một kỳ tương quan (hệ số thiết kế deff tới 2.9 với chiến thuật chọn từ một pool hẹp). "Lợi thế <" là cận TOST 95%: lợi thế thật (số trúng/vé) nhỏ hơn mức này so với mức ngẫu nhiên.

| Chiến thuật | Mega: TB (null 0.800) | z | lợi thế < | Power: TB (null 0.655) | z | lợi thế < | Độ phổ biến |
|---|---:|---:|---:|---:|---:|---:|---:|
| random (quick pick) | 0.789 | −1.02 | 3.5% | 0.669 | +1.47 | 4.7% | 1.00 |
| hot_50 | 0.803 | +0.18 | 3.9% | 0.650 | −0.30 | 4.8% | 1.13–1.15 |
| cold_50 | 0.777 | −1.38 | 6.4% | 0.668 | +0.78 | 6.1% | 1.16–1.29 |
| overdue (gan) | 0.804 | +0.25 | 4.0% | 0.669 | +0.88 | 6.4% | 1.02–1.08 |
| Bayesian (λ = 0.99) | 0.804 | +0.36 | 2.6% | 0.648 | −0.68 | 3.6% | 0.98–0.99 |
| Markov | 0.817 | +1.46 | 4.4% | 0.637 | −1.66 | 5.4% | 0.97–1.03 |
| GCN | 0.793 | −0.69 | 3.1% | 0.652 | −0.23 | 2.9% | 0.99–1.02 |
| anti_popularity | 0.805 | +0.29 | 3.7% | 0.645 | −0.70 | 5.0% | **0.42–0.44** |
| wheel 10 số (4 về ⇒ ≥3) | 0.788 | −0.67 | 5.2% | 0.652 | −0.18 | 4.7% | 1.02–1.08 |

- **SPA (Hansen 2005)** — kiểm định "chiến thuật *tốt nhất* trong 9 có hơn ngẫu nhiên không", đã tính tới việc thử nhiều chiến thuật: p = 0.42 (Mega, tốt nhất là markov), p = 0.39 (Power, tốt nhất là… random). White Reality Check: 0.69 / 0.66.
- **E-process theo từng chiến thuật**: hot_50 (Mega) từng đạt 74× và random (Power) 33× ở giữa giai đoạn — vượt ngưỡng 20× của *một* kiểm định — rồi rơi về 0.15× và 1.19×. Sau Holm cho 9 "người cá cược", không chiến thuật nào có ý nghĩa. Đây chính là cái bẫy "đang có phong độ": ngay cả quick pick cũng có chuỗi may mắn.
- ROI: toàn bộ −87% đến −95%, khoảng tin cậy bootstrap chồng lấn hoàn toàn.

GCN walk-forward (Diebold–Mariano, sai số HAC): Mega −0.00002 ± 0.00016 nats/kỳ (t = −0.14), Power −0.0006 ± 0.0005 (t = −1.13); xác suất dự báo được hiệu chuẩn tốt (Spiegelhalter p ≥ 0.93) — nghĩa là mô hình đã học đúng rằng mọi số có xác suất ≈ k/n. E-process dùng chính xác suất GCN để cá cược: tối đa 10^0.04×. Cùng mô hình phát hiện ngay tín hiệu cài sẵn trong dữ liệu mô phỏng (t > 5, `tests/test_ml.py`).

### 1.3 Kỳ vọng tiền thưởng (sau thuế TNCN, 1 vé 10.000đ)

Bảng v1 dùng mô hình độ phổ biến **prior** (mục 4.4); bảng đã hiệu chỉnh trên dữ liệu thật ở mục 1.4.4. *BE* = Jackpot hòa vốn.

| Game, Jackpot, vé bán | Vé 1-2-3-4-5-6 | Vé “ngày sinh” 3-7-8-19-25-31 | Vé ít người chọn |
|---|---|---|---|
| Mega, 30 tỷ, 1,5 triệu vé | RTP 15.8% · BE 1.241 tỷ | RTP 39.5% · BE 101 tỷ | RTP 45.5% · BE 81 tỷ |
| Mega, 100 tỷ, 4 triệu vé | RTP 16.3% | RTP 73.0% | RTP 113% |
| Power, 60 tỷ, 1,5 triệu vé | RTP 17.0% · BE 1.735 tỷ | RTP 34.5% · BE 293 tỷ | RTP 36.5% · BE 267 tỷ |
| Power, 200 tỷ, 5 triệu vé | RTP 16.4% | RTP 61.3% | RTP 76.5% |

**Bất định (v2, `vietlott decide`).** Tham số hành vi người chơi và số vé bán được lấy mẫu từ prior (300 lần), EV tính lại cho từng mẫu:

| Vé | RTP 5% – trung vị – 95% | P(RTP > 1) | Yếu tố chi phối |
|---|---|---:|---|
| Mega 100 tỷ, vé ít người chọn, 2–6 triệu vé | 104% – 114% – 119% | 99.7% | số vé bán (ρ = −0.75), tỷ lệ quick pick |
| Mega 100 tỷ, vé "ngày sinh" | 59% – 80% – 96% | 0.3% | số vé bán, thiên lệch ngày sinh |
| Power 200 tỷ, vé ít người chọn, 3–8 triệu vé | 73% – 76% – 78% | 0% | số vé bán |

**Kelly (tăng trưởng log tối ưu).** Với vé Mega 113% ở trên: độ lệch chuẩn tiền thưởng ≈ 29 triệu đ cho vé 10.000đ, P(trúng bất kỳ giải nào) = 2.4%. Tỷ lệ vốn tối ưu mỗi vé = 1.8·10⁻⁸ ⇒ cần **vốn ≈ 560 tỷ đ** mới hợp lý để mua **một** vé. Khi RTP ≤ 100% (hầu hết các kỳ), mức cược tối ưu bằng 0 — đó là kết quả toán học, không phải lời khuyên đạo đức.

Ngay cả khi RTP > 100%, xác suất trúng Jackpot của một vé vẫn là 1/8.145.060 (Mega) hay 1/28.989.675 (Power). Con số RTP phụ thuộc trực tiếp vào giả định độ phổ biến và số vé bán ra — hãy hiệu chỉnh (mục 4.4) trước khi tin.


### 1.4 v3 — dữ liệu giải thưởng, hành vi người chơi, doanh số, bao, tần suất trúng, chia giải

Tái tạo: `vietlott market` (ghi `data/calibration/`) rồi `python scripts/market_report.py` (ghi `reports/vietlott_v3.{md,json}`).

#### 1.4.1 Dữ liệu giải thưởng (`vietlott prizes --source canonical`)

| Game | Kỳ quay | Kỳ có số vé trúng từng giải | Kỳ có giá trị Jackpot | Nguồn |
|---|---:|---:|---:|---|
| Mega 6/45 | 1.569 (#1 → #1569, 30/09/2026) | 1.569 | 1.569 | trang chi tiết kỳ quay vietlott.vn |
| Power 6/55 | 1.405 (#1 → #1405, 01/10/2026) | 1.405 | 1.405 (JP1 + JP2) | như trên |
| Lotto 5/35 | 920 (#1 → #920, 01/10/2026) | 920 (7 hạng giải) | 920 | như trên (#920 từ kho cộng đồng, nguồn phụ) |

Từ v3.2 mỗi bản ghi là bảng giải trên **trang chi tiết kỳ quay của vietlott.vn** (kèm URL, SHA-256 của trang, link PDF biên bản) — bộ dữ liệu công khai [pqminh-4/vietlott-data](https://github.com/pqminh-4/vietlott-data) thu từ máy đặt tại Việt Nam; crawler của engine sinh đúng định dạng này khi chạy từ Việt Nam (mục 1.6). Giá trị Jackpot trên trang là *mỗi vé trúng* ⇒ pot = giá trị × max(1, số vé trúng). So với dữ liệu bên thứ ba của v3.1 (`data/seed/OFFICIAL_DATA_REPORT.json`): Mega 161/161 kỳ khớp (dữ liệu chính thức thêm 1.408 kỳ có giải thưởng và 197 kỳ đầu vốn thiếu trong mirror); Lotto 917/917 khớp; **Power lệch số người trúng ở 13 kỳ và Jackpot ở 23 kỳ** (lệch tới ×6,2) — đã thay bằng dữ liệu chính thức. Các đối chiếu báo chí của v3 (kỳ chia giải Lotto #38, #54, #146; 19 kỳ trúng Độc đắc Lotto 01–07/2026) vẫn đúng.

#### 1.4.2 Tỷ lệ trả thưởng & số vé bán

| | Power 6/55 | Mega 6/45 | Lotto 5/35 |
|---|---|---|---|
| Phần doanh thu vào Jackpot + giải cố định (s) | **0,418** — 3 hạng giải ước lượng độc lập cho 0,417 / 0,419 / 0,418 | **0,465** — 0,462 / 0,464 / 0,465 | Jackpot **hai pha**: 0,078 (chậm) / 0,375 (nhanh) × doanh thu; cùng giải cố định 0,25 / 0,55 (mục 1.4.7) |
| Vé bán/kỳ: trung vị (p10–p90) | 1,03 tr (0,65–2,90 tr) | 0,67 tr (0,46–1,82 tr); kỳ #01569: 5,25 tr | 170 nghìn (116–267 nghìn); kỳ chia giải 0,7–3,3 tr |
| Phương pháp | đồng nhất thức kế toán ΔJ + F = s·R | như Power, trên chuỗi Jackpot chính thức (v3 dựng lại đường Jackpot từ 161 kỳ ⇒ doanh số cao hơn ~43%) | profile NB từ số người trúng |
| Doanh số theo Jackpot | độ co giãn 0,84 ± 0,02 (gần tuyến tính log) | 0,65 ± 0,03 tại trung vị, **độ cong +0,20 ± 0,05** ("cơn sốt Jackpot") | — |

#### 1.4.3 Hành vi người chơi đã hiệu chỉnh

| | Power 6/55 | Mega 6/45 (chuyển giao từ Power) | Lotto 5/35 |
|---|---|---|---|
| Tỷ lệ vé chọn ngẫu nhiên (quick pick) | 51% ± 14% | (như Power) | 47% ± 7% |
| Số được chọn nhiều nhất | 5, 9, 8, 7, 12, 3, 19, 11 | 5, 9, 8, 7, 12, 3, 19, 11 | 19, 9, 12, 8, 25, 11, 7, 20 |
| Số ít người chọn nhất | 46, 50, 47, 55, 49, 40, 41, 53 | 40, 41, 31, 42, 44, 34, 30, 1 | 1, 35, 4, 2, 29, 30, 31, 21 |
| Kiểm tra ngoài mẫu (kỳ cuối, theo thời gian) | +0,20 log-lik/kỳ, p = 7·10⁻⁵ (281 kỳ) | mô hình riêng: ≈ 0 log-lik/kỳ, p = 0,61 (không đạt); mô hình chuyển giao: kiểm tra mức tương quan 0,30 (riêng: 0,24), hệ số góc 0,74, p = 0,0002 | +3,13 log-lik/kỳ, p = 4·10⁻¹⁵ (184 kỳ) |
| Độ phổ biến vé ngẫu nhiên p01 – p50 – p99 | 0,66 – 0,94 – 1,80 | 0,70 – 0,95 – 1,71 | 0,63 – 0,94 – 1,51 |

Đám đông thật **ôn hòa hơn prior**: vé ít người chọn nhất ≈ 0,6× mức trung bình (prior v1: 0,42×), vé "ngày sinh" ≈ 1,5–2,1×. Với đủ 1.569 kỳ, mô hình đám đông ước lượng riêng cho Mega vẫn **không** vượt đám đông chọn đều ngoài mẫu, còn mô hình chuyển từ Power giải thích mức số giải Ba tốt hơn — quy tắc chọn đặt trước (`transfer_check.selected` trong `market_mega645.json`) giữ mô hình chuyển giao. Không mô hình nào qua kiểm định likelihood trên Mega, nên con số ×r của Mega chỉ là tham khảo. Hệ số mẫu hình tổ hợp (1-2-3-4-5-6, đường thẳng trên phiếu) vẫn là prior vì dữ liệu số người trúng không nhận diện được chúng.

#### 1.4.4 EV với mô hình đã hiệu chỉnh (sau thuế, 1 lượt 10.000 đ)

Số vé bán lấy từ mô hình doanh số (phản ứng theo Jackpot); ×r = độ phổ biến so với vé chọn đều.

| Game · Jackpot · vé bán dự báo | Vé 1-2-3-4-5-6 | Vé "ngày sinh" 3-7-8-19-25-31 | Vé ít người chọn (tối ưu) | Jackpot hòa vốn (có phản ứng doanh số) |
|---|---|---|---|---|
| Mega · 30 tỷ · 0,98 tr | 22,8% (×29,6) | 44,0% (×1,52) | 30-31-40-41-42-44: 45,7% (×0,59) | 86,3 tỷ |
| Mega · 100 tỷ · 3,36 tr | 22,7% | 95,8% | 111,8% | |
| Mega · **158,8 tỷ (kỳ #01569)** · 6,28 tr | 21,4% | 117,1% | **154,8%** | |
| Power · 60 tỷ (+5 tỷ JP2) · 1,46 tr | 24,8% (×40,2) | 38,7% (×2,13) | 40-41-46-49-50-55: 39,7% (×0,56) | 275 tỷ |
| Power · 200 tỷ (+10 tỷ JP2) · 4,07 tr | 27,1% | 78,6% | 85,8% | 253 tỷ |
| Lotto · 6 tỷ · 170 nghìn | 28,1% (×12,3, số ĐB 8) | 30,7% | 1-2-31-32-35 + ĐB 1: 31,1% (×0,51) | 36 tỷ — vượt ngưỡng chia giải 12 tỷ ⇒ **kỳ thường luôn âm** |
| Lotto · 11 tỷ · 170 nghìn | 37,0% | 41,8% | 42,5% | |

Power khó dương hơn Mega: odds 1/29 triệu so với 1/8,1 triệu, và doanh số Power phản ứng mạnh hơn ở Jackpot vừa (độ co giãn 0,84). Mức Kelly cho vé Mega 155%: tỷ lệ vốn tối ưu 5,1·10⁻⁸ ⇒ cần ≈ 195 tỷ đ vốn mới hợp lý mua một vé (`vietlott decide`). v3.2 sửa cách tìm Jackpot hòa vốn: với đường doanh số lồi, EV không đơn điệu theo Jackpot (ngoại suy rất xa làm EV giảm trở lại), nên hàm tìm *điểm cắt đầu tiên* trên lưới log rồi mới tinh chỉnh bằng Brent.

#### 1.4.5 Chơi bao vs vé lẻ (phân phối chính xác, `vietlott bao`)

| Game | Kiểu | Lượt | Chi phí (đ) | P(trúng ≥ 1 giải) | Cùng tiền, vé lẻ độc lập |
|---|---|---:|---:|---:|---:|
| Mega | Bao 5 | 40 | 400.000 | 12,5% | 61,9% |
| Mega | Bao 7 | 7 | 70.000 | 3,9% | 15,5% |
| Mega | Bao 8 | 28 | 280.000 | 5,9% | 49,1% |
| Mega | Bao 10 | 210 | 2.100.000 | 11,3% | 99,4% |
| Power | Bao 5 | 50 | 500.000 | 8,6% | 48,9% |
| Power | Bao 7 | 7 | 70.000 | 2,2% | 9,0% |
| Lotto | Bao 4 + 1 số ĐB | 31 | 310.000 | 16,5% | 95,6% |
| Lotto | Bao 6 + 1 số ĐB | 6 | 60.000 | 10,8% | 45,4% |
| Lotto | 1 bộ 5 số × 6 số ĐB | 6 | 60.000 | **50,7%** | 45,4% |

Tiền thưởng cố định kỳ vọng trên mỗi đồng của bao và vé lẻ **bằng nhau** (tuyến tính của kỳ vọng — đã kiểm bằng test); riêng Jackpot, các lượt của cùng một vé bao trúng chung một pot thì chỉ nhận nhiều *phần* của pot đó (mục 1.5). Bao gom giải thành cụm (Bao 8 Mega trúng 4 số ⇒ 6 giải Nhì + 16 giải Ba cùng lúc) nên xác suất trúng *ít nhất một* giải thấp hơn nhiều. Với Lotto, giải khuyến khích chỉ cần trùng số đặc biệt nên phủ nhiều số đặc biệt khác nhau mới là cách tăng tần suất.

#### 1.4.6 Tăng tần suất trúng: danh mục tối đa P(trúng ≥ 1 giải) (`vietlott coverage`)

| Game | Số vé | Danh mục tối ưu [95%] | Vé ngẫu nhiên | Cận trên B·p | Số giải kỳ vọng (mọi danh mục) |
|---|---:|---:|---:|---:|---:|
| Mega | 10 | 23,0% [22,8; 23,1] | 21,4% | 23,8% | 0,24 |
| Mega | 28 | 55,2% [54,9; 55,4] | 49,0% | 66,7% | 0,67 |
| Power | 10 | 13,0% [12,9; 13,2] | 12,6% | 13,3% | 0,13 |
| Power | 28 | 33,9% [33,7; 34,1] | 31,5% | 37,3% | 0,37 |
| Lotto | 6 | 54,1% [53,9; 54,4] | 45,5% | 57,6% | 0,58 |
| Lotto | 12 | 100% | 70,2% | 100% | 1,15 |

Đánh giá trên 200.000 kỳ mô phỏng **độc lập** với mẫu dùng để tối ưu (không thiên lệch tối ưu hóa); baseline rút một danh mục ngẫu nhiên mới cho mỗi kỳ. Cột cuối cho thấy cái giá: số giải (và tiền thưởng) kỳ vọng không đổi — danh mục chỉ dàn đều các lần trúng ra nhiều kỳ hơn.

#### 1.4.7 Lotto 5/35 — chia giải Độc đắc trên chuỗi Jackpot chính thức (`vietlott rolldown --history`)

Luật: khi Độc đắc vượt 12 tỷ mà chưa có người trúng, kỳ 21:00 ngày hôm sau chia toàn bộ cho giải Nhất (1/3) và giải Nhì–Năm (1/6 mỗi hạng); hạng không có người trúng chuyển phần của mình cho các hạng còn lại.

- **Sự kiện** (đọc trực tiếp từ Jackpot công bố, v3.2): 17 kỳ chia giải đã thực hiện, 6 lần đã công bố chia nhưng có người trúng Độc đắc trước. Phương pháp v3 (không có chuỗi Jackpot, nhận diện qua doanh số tăng ≥ 3×) bắt đúng 21/23 sự kiện, không báo nhầm, sai số pot trung vị 5,7% (lớn nhất 19%) — chỉ bỏ sót #523 và #639, hai kỳ có người trúng Độc đắc ngay kỳ 13:00 nên không có đột biến doanh số.
- **Tích lũy hai pha** (suy ra từ giá trị công bố — Vietlott không công bố quy tắc này): sau mỗi lần Jackpot khởi động lại 6 tỷ, pot chỉ tăng **0,078 × doanh thu** (pha chậm); khi phần chênh lệch so với pha nhanh đã bù đủ ≈ **5,64 tỷ cho mỗi lần khởi động lại** (rất đều qua 28 đoạn), pot chuyển sang **0,375 × doanh thu** (pha nhanh, ở mức Jackpot trung vị 7,9 tỷ, khoảng 7,3–10,1 tỷ). Nghĩa là tiền khởi điểm 6 tỷ của Độc đắc được "hoàn quỹ" từ chính các kỳ đầu của đoạn sau. Tổng tỷ lệ vào quỹ giải (Jackpot + giải cố định) 0,25 ở pha chậm và 0,55 ở pha nhanh. Hệ số c = 0,155 của v3 là trung bình của hai pha.
- **Dự báo kỳ chia giải tiếp theo** (`forecast_next_rolldown`): tính số tiền còn phải hoàn quỹ, số kỳ đến khi chuyển pha và đến ngưỡng 12 tỷ theo doanh số 28 kỳ gần nhất. Sau kỳ #920 (01/10/2026): Jackpot 7,56 tỷ, sắp chuyển pha, ≈ 6,6 kỳ đến ngưỡng ⇒ chia giải khoảng 3–5 ngày tới nếu không ai trúng Độc đắc trước. Kiểm định lại trên 48 thời điểm quá khứ: sai số tuyệt đối trung vị 2,3 kỳ, 62% trong ±3 kỳ, Spearman 0,87.
- **RTP thực của kỳ chia giải** (pot công bố + giải cố định / doanh thu ước lượng của kỳ đó, trước thuế): 2,44 (#38) → 1,08 (#874); nửa đầu trung bình 1,64, nửa sau 1,22; ρ = −0,90 (p = 10⁻⁶). Mọi kỳ chia đều > 1 trước thuế, nhưng kỳ gần nhất chỉ ≈ 1,01 sau thuế với 3,3 triệu vé. Đây là thị trường đang học: càng nhiều người đổ vào kỳ chia giải, lợi thế càng mất. Hòa vốn ở pot 22 tỷ ≈ 2,5 tr vé (`vietlott rolldown --jackpot 22e9 --tickets-sold 2500000`).

### 1.5 Chơi bao — nghiên cứu chuyên sâu (v3.1, `vietlott bao`, `reports/bao_vietlott.md`)

#### 1.5.1 Luật chơi bao của từng sản phẩm

| Sản phẩm | Kiểu bao | Cách sinh lượt | Số lượt (× 10.000 đ) |
|---|---|---|---|
| Mega 6/45 | Bao 5 | 5 số cố định + lần lượt từng số trong 40 số còn lại | 40 |
| | Bao 7 … Bao 15, Bao 18 | mọi bộ 6 số từ v số đã chọn, C(v, 6) | 7 · 28 · 84 · 210 · 462 · 924 · 1.716 · 3.003 · 5.005 · 18.564 |
| Power 6/55 | Bao 5 | 5 số + từng số trong 50 số còn lại | 50 |
| | Bao 7 … 15, Bao 18 | C(v, 6), cùng giá như Mega | như Mega |
| Lotto 5/35 | Bao số đặc biệt | 5 số chính + 2–12 số đặc biệt, mỗi số ĐB một lượt | 2 … 12 |
| | Bao 4 | 4 số chính + từng số trong 31 số còn lại (+ 1 số ĐB) | 31 |
| | Bao 6 … Bao 15 | C(v, 5) bộ 5 số (+ 1 số ĐB) | 6 · 21 · 56 · 126 · 252 · 462 · 792 · 1.287 · 2.002 · 3.003 |
| Max 3D Pro | Bao bộ ba số | mọi hoán vị chữ số của số trước × số sau | 123-456: 36 · 112-456: 18 · 111-456: 6 |
| | Bao nhiều bộ ba số | 3–20 số ⇒ mọi cặp có thứ tự của hai số khác nhau, n(n−1) | 6 … 380 |
| Max 3D / 3D+ | Đảo số, bao vị trí (`*`) | hoán vị chữ số; một/hai vị trí bất kỳ ⇒ 10/100 lượt | theo đại lý (chưa đối chiếu được trang Vietlott) |

Không có Bao 6, 16, 17 cho Mega/Power. Kết hợp bao số chính Lotto với nhiều số đặc biệt không thấy trong tài liệu công khai — hệ thống vẫn tính (mọi tổ hợp × mọi số ĐB) nhưng gắn cờ `documented = false`; `--strict` / `"strict": true` thì từ chối.

#### 1.5.2 Bảng giải thưởng — khớp các ví dụ đã công bố

Mọi bảng được tính chính xác từ luật (không tra bảng), rồi đối chiếu với ví dụ của Vietlott/đại lý (test `tests/test_bao.py`):

| Vé | j số trúng trong bộ | Hệ thống tính | Công bố |
|---|---|---|---|
| Power Bao 5 | 2 · 3 | 200.000 · 3.850.000 | 200.000 · 3.850.000 |
| Power Bao 5 | 4 + số phụ trong 5 số · 5 | Jackpot 2 (×2 phần) + 24.000.000 · Jackpot 1 + Jackpot 2 + 1.920.000.000 | JP2 + 24 triệu · JP1 + JP2 + 1,92 tỷ |
| Power Bao 7 | 3 · 4 · 5 | 200.000 · 1.700.000 · 82.500.000 | 200.000 · 1.700.000 · 82.500.000 |
| Power Bao 7 | 5 + số phụ · 6 | Jackpot 2 + 42.500.000 · Jackpot 1 + 240.000.000 | JP2 + 42,5 triệu · JP1 + 240 triệu |
| Mega Bao 5 | 2 | 120.000 | 120.000 |

`vietlott bao --game power655 --table 8` in bảng của mức bất kỳ (vd. Power Bao 8, 6 số + số phụ trong bộ: Jackpot 1 + Jackpot 2 (×6 phần) + 247.500.000).

#### 1.5.3 Ba điều luật làm thay đổi con số

1. **Nhiều lượt cùng trúng một Jackpot ⇒ nhiều phần của *một* pot.** Power Bao 7 về đủ 6 số và số phụ nằm trong bộ: 6 lượt trúng Jackpot 2. Jackpot chia đều theo lượt trúng nên vé được 6/(6 + K) pot, K = lượt trúng của người khác — không phải 6 × pot. Phiên bản trước cộng 6 × pot; đã sửa (`ticket_payout`), kèm hàm E[n/(n+K)] với K ~ Poisson.
2. **Thuế tính trên vé.** Vé bao là một vé, nên các giải cộng lại trước khi áp thuế 10% phần vượt ngưỡng: Mega Bao 7 về 5 số được 21,5 triệu ⇒ chịu thuế trên phần vượt 20 triệu, trong khi 2 giải Nhất + 5 giải Nhì mua lẻ thì không lần nào vượt ngưỡng. Chênh lệch kỳ vọng nhỏ (dưới 1% tiền thưởng kỳ vọng trong các ví dụ của báo cáo) nhưng luôn bất lợi cho vé bao. `--tax-basis play` để so sánh.
3. **Lotto: số đặc biệt quyết định tần suất.** Giải khuyến khích chỉ cần trùng số đặc biệt, nên *bao số đặc biệt* là kiểu bao duy nhất có P(trúng ≥ 1 giải) **cao hơn** cùng tiền mua vé lẻ (6 số ĐB: 50,7% so với 45,4%). Bao 4: hễ trùng số ĐB (xác suất 1/12) là nhận ít nhất 310.000 đ = tiền vé.

#### 1.5.4 Cùng tiền, khác hình dạng (mô phỏng 100.000 kỳ, giải cố định)

| Cách chơi | Lượt | Chi phí | P(trúng ≥ 1 giải) | P(thưởng ≥ tiền vé) | Thưởng TB | Độ lệch chuẩn |
|---|---:|---:|---:|---:|---:|---:|
| Mega **Bao 8** | 28 | 280.000 | 5,9% | **5,9%** | 39.300 | 788.000 |
| Mega bao rút gọn "3 nếu về 3" từ cùng 8 số | **4** | **40.000** | **5,9%** | 2,9% | 5.700 | 142.000 |
| Mega 28 vé lẻ dàn đều | 28 | 280.000 | **55,8%** | 4,0% | 41.000 | 327.000 |
| Mega 28 vé quick pick | 28 | 280.000 | 49,4% | 3,9% | 39.000 | 298.000 |
| Mega **Bao 10** | 210 | 2.100.000 | 11,2% | 1,6% | 274.000 | 2.930.000 |
| Mega bao rút gọn "3 nếu về 3" từ cùng 10 số | 10 | 100.000 | 11,2% | 1,6% | 12.500 | 157.000 |
| Power **Bao 7** | 7 | 70.000 | 2,2% | 2,2% | 6.300 | 67.000 |
| Power 7 vé lẻ dàn đều | 7 | 70.000 | 9,6% | 0,6% | 9.100 | 312.000 |
| Lotto **Bao 6** (+1 ĐB) | 6 | 60.000 | 10,7% | **10,7%** | 10.100 | 116.000 |
| Lotto 6 vé lẻ dàn đều | 6 | 60.000 | 54,3% | 1,1% | 10.700 | 96.000 |

Cách đọc: cùng số lượt thì tiền thưởng trung bình như nhau. **Bao** dồn giải vào ít kỳ — hiếm khi trúng, nhưng đã trúng (≥ 3 số nằm trong bộ) thì gần như luôn ≥ tiền vé. **Vé lẻ dàn đều** trúng thường xuyên gấp ~10 lần nhưng hiếm khi hoàn vốn. **Bao rút gọn** (covering design, `src/vietlott_engine/wheeling`) giữ nguyên xác suất trúng ≥ 1 giải của bao gốc với 1/7 (Bao 8) đến 1/21 (Bao 10) chi phí — đổi lại mất các giải "cộng dồn" khi về nhiều số. Không cách nào thay đổi P(trúng Jackpot) của một số lượt cho trước: mọi tổ hợp khác nhau đều có xác suất 1/C(n, k).

#### 1.5.5 Bao khi Jackpot lớn

| Vé | Jackpot | Vé bán | RTP sau thuế (không ai trúng chung) | RTP sau thuế (có người trúng chung, độ phổ biến trung bình) |
|---|---:|---:|---:|---:|
| Mega Bao 7 | 158,8 tỷ (kỳ #01569) | 6,3 triệu (mô hình doanh số) | 189% | 136% |
| Mega Bao 8 | 158,8 tỷ | 6,3 triệu | 189% | 136% |
| Power Bao 7 | 60 + 5 tỷ | 1,5 triệu | 41% | 39% |

Bao không làm thay đổi kỳ vọng so với mua lẻ cùng các lượt đó; chọn số ít người chơi (mục 1.4.4) vẫn là đòn bẩy duy nhất cho phần Jackpot (`POST /games/{game}/bao` với `tickets_sold` dùng mô hình đám đông đã hiệu chỉnh).

#### 1.5.6 Họ Max 3D (`vietlott max3d`, `GET /max3d`)

20 số ba chữ số quay độc lập mỗi kỳ (2 + 4 + 6 + 8), giải cộng dồn. Xác suất tính chính xác (liệt kê trạng thái), đối chiếu số liệu công bố:

| Sản phẩm | Giải cao nhất | Xác suất | RTP | P(trúng ≥ 1 giải) | Đối chiếu |
|---|---:|---:|---:|---:|---|
| Max 3D | 1.000.000 | 1/500 | 54,5% | 1,98% | — |
| Max 3D+ | 1 tỷ | 1/500.000 | 54,4% | 3,92% | "xác suất 1/500.000" (Thời báo Tài chính) |
| Max 3D Pro | 2 tỷ | 1/1.000.000 | 54,7% | 3,92% | "trả thưởng lên đến 55%", "tỷ lệ có giải 4%" (Thanh Niên, Tuổi Trẻ) |

| Bao | Lượt | Chi phí | P(trúng ≥ 1 giải) | Cùng tiền, lượt ngẫu nhiên | P(thưởng ≥ tiền vé) |
|---|---:|---:|---:|---:|---:|
| Max 3D Pro bao bộ số 123-456 | 36 | 360.000 | 21,5% | 76,3% | 4,1% |
| Max 3D Pro bao bộ số 112-456 | 18 | 180.000 | 16,5% | 51,4% | 7,3% |
| Max 3D Pro bao 3 số (6 cặp) | 6 | 60.000 | 5,8% | 21,4% | 5,9% |
| Max 3D Pro bao 10 số (90 cặp) | 90 | 900.000 | 18,2% | 97,3% | 3,3% |

Ở Max 3D Pro, P(trúng ≥ 1 giải) chỉ phụ thuộc vào *số lượng số khác nhau* được chọn: 1 − (1 − u/1000)²⁰; bao lặp lại cùng các số nên tần suất thấp hơn nhiều so với cùng tiền mua các cặp khác nhau. Vé hai số giống nhau (333-333) ở Max 3D+ được nhân đôi giải nên RTP giữ ≈ 54,6%; ở Max 3D Pro không thấy quy tắc này — nếu đúng vậy, RTP của vé như thế chỉ ≈ 35%.

### 1.6 Truy cập vietlott.vn & toàn bộ sản phẩm (v3.2, `vietlott products`, `reports/products.md`)

#### 1.6.1 Truy cập vietlott.vn

- **vietlott.vn chỉ phục vụ địa chỉ IP Việt Nam.** Từ IP nước ngoài (máy chủ đám mây, GitHub Actions) trang trả HTTP 403; các dự án thu thập dữ liệu công khai đều chạy crawler trên máy đặt tại Việt Nam vì lý do này. Trong môi trường xây dựng bản v3.2, proxy mạng còn chặn cả `www.vietlott.vn` lẫn `raw.githubusercontent.com` (chỉ `git clone` từ GitHub hoạt động), nên dữ liệu của bản này được dựng từ **bản sao công khai của chính các trang vietlott.vn** (git clone) và crawler được kiểm bằng HTML mẫu + transport giả lập — chưa chạy trực tiếp với website.
- **Cách lấy dữ liệu trực tiếp**:
  1. Chạy engine trên máy ở Việt Nam (PC, VPS đặt tại VN, self-hosted runner):
     ```bash
     vietlott sync --game all --source vietlott                 # Mega, Power, Lotto
     vietlott prizes --game all --source vietlott --last 30     # bảng giải 30 kỳ gần nhất (trang chi tiết)
     vietlott products sync --source vietlott                    # Keno, Bingo18, Max 3D, Max 3D Pro
     ```
  2. Từ nước ngoài: đi qua proxy **do bạn vận hành** tại Việt Nam bằng biến chuẩn `HTTPS_PROXY=http://user:pass@may-o-vn:3128` (httpx tự dùng).
  3. Không cần IP Việt Nam: `--source auto` (mặc định từ v3.3) tự chuyển sang kho cộng đồng (Keno/Bingo18 ~10 phút/lần, sản phẩm khác vài lần/ngày) rồi các bản sao GitHub khi vietlott.vn không trả lời — mục 1.7; hoặc `make official-data` (git clone, dựng lại `data/seed`).
  4. Đọc tay bằng trình duyệt: khi Claude được kết nối với trình duyệt trên một máy ở Việt Nam (Claude in Chrome, hoặc trình duyệt tích hợp của ứng dụng Claude desktop), có thể mở trực tiếp trang kết quả; engine không phụ thuộc vào cách này.
- **Hai lỗi được phân biệt** (không thử lại vô ích): `vietlott.vn rejected the request (HTTP 403)…` = website chặn IP ngoài Việt Nam; `this network's HTTPS proxy does not allow www.vietlott.vn…` = proxy/tường lửa của mạng bạn đang dùng chặn host. Proxy từ chối (403/407) không còn bị retry.
- **Giao thức**: danh sách kết quả qua AjaxPro — `POST /ajaxpro/Vietlott.PlugIn.WebParts.<Part>,Vietlott.PlugIn.WebParts.ashx`, header `X-AjaxPro-Method: ServerSideDrawResult`, body JSON theo từng sản phẩm (`ENDPOINTS` trong `src/vietlott_engine/crawler/sources/vietlott_official.py`), phản hồi `{"value": {"HtmlContent": …}}`; cookie khởi tạo đọc từ `document.cookie` của `/ajaxpro/`. Trang chi tiết kỳ quay `…?id=NNNNN&nocatche=1` cho bảng giải (cột được định vị theo tiêu đề *Giải thưởng / Số lượng giải / Giá trị giải*, nên đổi thứ tự cột không làm hỏng parser), link PDF biên bản trên `media.vietlott.vn` và SHA-256 của trang để truy vết. Nếu website đổi cấu trúc, chỉ cần sửa `ENDPOINTS` hoặc parser tương ứng — test fixture sẽ chỉ ra chỗ hỏng.

#### 1.6.2 Phạm vi dữ liệu (sau khi gộp nguồn ở v3.3)

| Sản phẩm | Cách quay | Lịch | Kỳ trong dữ liệu | Từ – đến | Nguồn |
|---|---|---|---:|---|---|
| Mega 6/45 | 6 số từ 1–45 | 18:00 T4, T6, CN | 1.569 | #1 → #1569 (30/09/2026) | trang chi tiết vietlott.vn: kết quả + bảng giải + Jackpot |
| Power 6/55 | 6 số + số phụ từ 1–55 | 18:00 T3, T5, T7 | 1.405 | #1 → #1405 (01/10/2026) | như trên |
| Lotto 5/35 | 5 số 1–35 + số đặc biệt 1–12 | 13:00 và 21:00 hằng ngày | 920 | #1 → #920 (01/10/2026) | như trên; #920 từ kho cộng đồng (nguồn phụ) |
| Keno | 20 số khác nhau từ 1–80 | ~10 phút/kỳ, 06:00–21:5x | 297.367 | 23/08/2019 → 02/10/2026 | danh sách vietlott.vn (86.272) + kho cộng đồng (211.126, phần lớn từ kho xoso.com.vn) |
| Bingo18 | 3 số độc lập 1–6 | ~6 phút/kỳ, 06:00–21:5x | 105.563 | 03/12/2024 → 02/10/2026 | danh sách vietlott.vn (92.759) + kho cộng đồng (12.834) |
| Max 3D / Max 3D+ | 20 số 000–999 (2 + 4 + 6 + 8) | 18:00 T2, T4, T6 | 1.139 | 22/04/2019 → 30/09/2026 | trang chi tiết vietlott.vn (pqminh-4/vietlott-data) |
| Max 3D Pro | 20 số 000–999 (2 + 4 + 6 + 8) | 18:00 T3, T5, T7 | 786 | 14/09/2021 → 01/10/2026 | như trên |
| Max 4D (đã ngừng) | 6 số 0000–9999 (1 + 2 + 3) | 18:00 T3, T5, T7 | 722 | 19/11/2016 → 31/08/2021 | trang chi tiết vietlott.vn (qua kho cộng đồng) |

Số kỳ Keno/Bingo18 đã trừ 31 và 30 kỳ Vietlott thông báo *không được xác nhận* (ngày 02/04/2026; `data/seed/exclusions.json`, bỏ khỏi mọi phân tích trừ khi `--include-unconfirmed`). Khoảng trống mã kỳ còn lại: Keno 415, Bingo18 564 (v3.2: 100.785 và 12.762); kiểm định giữa hai kỳ liền nhau chỉ dùng cặp có mã liên tiếp. Max 3D/Max 3D Pro: bốn bản sao độc lập (pqminh-4, vietvudanh, googlesky, kho cộng đồng) khớp nhau ở **mọi** kỳ chung. Kho sản phẩm: `data/seed` (đi kèm) + `$VQE_DATA_DIR/products` (đồng bộ), bản ghi mới thắng khi trùng mã kỳ.

#### 1.6.3 Keno & Bingo18 — xác suất chính xác và tỷ lệ trả thưởng (`vietlott products odds`)

- **Bảng giải Keno từ trang chính thức (v3.3)**: trang chi tiết Keno của vietlott.vn (kỳ #0284640, 13/06/2026, lưu trong kho cộng đồng — `data/seed/keno_rules_official.json`), khớp trang sản phẩm mà Vietlott Quant Engine 1.3.0 dùng. Hai chỗ khác bảng v3.2: **bậc 10 trùng 7 = 710.000 đ, trùng 8 = 8.000.000 đ** (bảng VTC Pay 2020 là bảng cũ 600.000 / 7.400.000), và cửa phụ (dưới đây).
- **Keno theo bậc** (siêu bội 80/20): RTP bậc 1–10 = 0,500 / 0,541 / 0,555 / 0,551 / 0,549 / 0,545 / 0,551 / 0,547 / 0,547 / **0,577**; giải cao nhất bậc 10 (2 tỷ) là 1/8,9 triệu. Ô duy nhất chưa thống nhất: bậc 5 trùng 4 số — trang sản phẩm ghi 150.000 đ, trang chi tiết 2026 ghi "0 đ" ở cả 30 kỳ đã thu; engine dùng 150.000 đ (RTP 0,549; nếu là 0 đ thì 0,368).
- **Cửa phụ Keno (đã xác minh)**: Lớn/Nhỏ ≥ 13 số = 26.000, 11–12 số = 10.000 (hoàn vốn) ⇒ RTP 0,555; Hòa Lớn–Nhỏ 26.000 ⇒ 0,528; Chẵn/Lẻ 13–14 số = 40.000, ≥ 15 = 200.000 ⇒ 0,542; **Chẵn/Lẻ 11–12 = 20.000 ⇒ 0,601** (cửa trả cao nhất của Keno); **Hòa Chẵn–Lẻ = 20.000 ⇒ 0,406** (thấp nhất). Bảng 55.000 (xosovip 2025) và 56.000 / 210.000 (VTC Pay 2020) là bản in khác, giữ làm phiên bản so sánh.
- **Bingo18** (216 kết quả đồng khả năng): mọi cửa trả 0,50–0,57 (một số: 0,569; bộ đôi, bộ ba, tổng 3–7 và 14–18: 0,556; Lớn/Nhỏ: 0,562; Hòa: 0,500); luật trả thưởng khớp trang chi tiết Bingo18.
- **Max 3D / 3D+ / 3D Pro**: RTP vé hai số khác nhau 0,545 / 0,544 / 0,547 (mục 1.5.6). Max 4D đã ngừng (08/2021): chỉ có lịch sử kết quả.

#### 1.6.4 Kiểm định ngẫu nhiên các sản phẩm (`vietlott products analyze`)

| Sản phẩm | Kỳ | Kiểm định (so với luật quay chính xác, BH) | q nhỏ nhất | Kết luận |
|---|---:|---|---:|---|
| Keno | 297.367 | tần suất 80 số (χ² hiệu chỉnh không hoàn lại ×79/60), số Lớn và số Chẵn mỗi kỳ (siêu bội), số trùng kỳ trước, tương quan liền kỳ | 0,172 | không bác bỏ |
| — phần chỉ có ở kho xoso.com.vn | 210.227 | như trên (kiểm tra lỗi chép số của nguồn phụ) | 0,163 | không bác bỏ |
| — phần danh sách vietlott.vn | 86.241 | như trên | 0,467 | không bác bỏ |
| Bingo18 | 105.563 | mặt 1–6 (gộp và từng vị trí), phân phối tổng 3–18, bộ ba, độc lập giữa vị trí, phụ thuộc liền kỳ | 0,492 | không bác bỏ |
| Max 3D | 1.139 | chữ số từng hàng, 1.000 số, giải Đặc biệt, trùng trong kỳ (Poisson), lặp từ kỳ trước | **0,000** | **bác bỏ: chữ số hàng đơn vị** |
| Max 3D Pro | 786 | như trên | 0,055 | không bác bỏ ở 5% |
| Max 4D | 722 | chữ số 4 hàng, trùng trong kỳ, lặp từ kỳ trước | 0,052 | không bác bỏ ở 5% |

**Chữ số 6 ở hàng đơn vị — ba trò chơi chữ số.** Ô lệch mạnh nhất trong Max 3D là chữ số 6 hàng đơn vị: 10,89% thay vì 10%. Kiểm định *chỉ ô này* trên các kỳ Max 3D Pro (độc lập, chưa dùng để chọn ô) cho 11,06% (1.739/15.720), p một phía = 6·10⁻⁶; làm ngược lại cũng ra đúng ô đó (p = 5·10⁻⁶). **v3.3 kiểm tra lần thứ ba trên Max 4D** — trò chơi khác, 2016–2021, giả thuyết đặt trước khi xem dữ liệu: 10,96% (475/4.332; 95%: 10,03–11,90%), p một phía = 0,019 — cùng cỡ lệch. Với Max 3D/Max 3D Pro, bốn bản sao độc lập khớp nhau ở mọi kỳ, nên không phải lỗi thu thập; lệch có mặt ở hầu hết các năm của hai trò này (trừ Max 3D 2019 và Max 3D Pro 2024). Nguyên nhân (thiết bị quay, bộ cầu số) không xác định được từ kết quả. Lệch **nhỏ**: số có lợi nhất (ước lượng trong mẫu, còn thiên lên) chỉ tăng xác suất ≈ 18%, đưa RTP từ 0,545 lên ≈ 0,64; vé Max 3D cần tăng ≈ 83% mới hòa vốn. Kết luận về cách chơi không đổi. Chia đôi theo thời gian trong từng sản phẩm thì bằng chứng yếu hơn: ô chọn ở nửa đầu đều nằm ở hàng đơn vị (Max 3D: chữ số 8 thiếu; Max 3D Pro: chữ số 6 dư) và lặp lại ở nửa sau chỉ với p ≈ 0,03 — chưa đạt ngưỡng xác nhận 0,01.

#### 1.6.5 API & CLI cho sản phẩm

```bash
vietlott products list                          # 8 sản phẩm, lịch quay, số kỳ trong kho
vietlott products sync                          # auto: vietlott.vn → kho cộng đồng → bản sao (mục 1.7)
vietlott products analyze --product all         # kiểm định + kiểm tra chéo chữ số Max 3D / Max 4D
vietlott products odds --product keno           # bảng giải, xác suất, RTP mọi cửa
python scripts/products_report.py                        # reports/products.{md,json}
```

`GET /products` (danh mục + dữ liệu đang có), `GET /products/keno/odds`, `GET /products/bingo18/odds`, `GET /products/{keno|bingo18|max3d|max3dpro|max4d}/draws?limit=`, `GET /products/{…}/randomness`, `GET /products/max3d/digit-check`, `POST /products/{…}/sync` `{"source": "auto|vietlott|nhanaz|mirror|canonical|v130", "max_pages": 50}` (báo cáo có `source_used` và `attempts`).

### 1.7 Khi không vào được vietlott.vn: nguồn dự phòng và bộ Vietlott Quant Engine 1.3.0 (v3.3)

**Chuỗi dự phòng.** Mọi lệnh đồng bộ nhận `--source auto` (mặc định cho `products sync` và `prizes`): `sync` và `products sync` thử lần lượt theo `VQE_FALLBACK_ORDER`, `prizes` theo thứ tự vietlott → canonical → nhanaz → v130; dừng ở nguồn đầu tiên trả lời, và ghi lại từng lần thử (`attempts`, `source_used`) — người đọc báo cáo biết dữ liệu đến từ đâu.

| Thứ tự | Nguồn | Khi nào dùng được | Cập nhật |
|---|---|---|---|
| 1 | `vietlott` — vietlott.vn trực tiếp (AjaxPro + trang chi tiết) | máy/proxy ở Việt Nam | tức thời |
| 2 | `nhanaz` — kho cộng đồng [NhanAZ-Data/vietlott-research](https://github.com/NhanAZ-Data/vietlott-research) (MIT): 8 sản phẩm, bảng giải, luật giải, danh sách kỳ không xác nhận; cập nhật bằng GitHub Actions — từ trang vietlott.vn khi lấy được, và từ trang kết quả phụ (xosominhngoc.net.vn, kho xoso.com.vn, onbit) khi không; mỗi dòng giữ `data_source` | mọi nơi có GitHub (raw), hoặc thư mục tải sẵn `VQE_NHANAZ_DIR` / `--path` (git clone, ZIP, hay thư mục cache của 1.3.0) | Keno/Bingo18 ~10 phút/lần ban ngày; sản phẩm khác vài lần/ngày |
| 3 | `github_mirror` / `mirror` / `canonical` — vietvudanh, pqminh-4 | mọi nơi có GitHub | hằng ngày |
| 4 | `v130` — snapshot `data/products/*.parquet` của **Vietlott Quant Engine 1.3.0** (`VQE_V130_DIR` / `--path`) | hoàn toàn offline | theo snapshot |

```bash
vietlott sync --game all --source auto                       # Mega/Power/Lotto (kèm bảng giải khi lấy từ kho)
vietlott prizes --game all                                   # auto: vietlott → canonical → nhanaz → v130
vietlott products sync                                       # Keno, Bingo18, Max 3D, Max 3D Pro
vietlott products sync --source nhanaz --path ~/vietlott-research   # kho đã clone, không cần mạng
make import-v130 V130=~/Vietlott-Quant-Engine-1.3.0                   # nhập snapshot của bộ 1.3.0
make official-data V130=~/Vietlott-Quant-Engine-1.3.0                 # dựng lại data/seed từ mọi nguồn + đối chiếu
```

Đọc Parquet của 1.3.0 bằng gói `duckdb` (đã có trong requirements), `pyarrow`, hoặc DuckDB CLI (`VQE_DUCKDB_CLI`). Kỳ có `status = not_confirmed` trong snapshot thành mục loại trừ.

**Gộp và đối chiếu** (`scripts/build_official_dataset.py`, báo cáo `data/seed/DATA_MERGE_REPORT.json`): bản ghi chính thức đứng trước; lớp sau chỉ thêm mã kỳ còn thiếu; mọi chỗ lệch được liệt kê.

| Sản phẩm | Kết quả đối chiếu trên phần trùng |
|---|---|
| Mega 6/45, Power 6/55 | kho cộng đồng và 1.3.0: 1.569 / 1.405 kỳ trùng — kết quả, số người trúng từng hạng và Jackpot **giống hệt 100%** bản ghi trang chi tiết |
| Lotto 5/35 | 919 kỳ trùng giống hệt, trừ **Jackpot kỳ chia giải #874**: 30,22 tỷ (trang chính thức, khớp với tiền từng hạng đã chia) so với 35,96 tỷ ở trang phụ xosominhngoc → giữ bản chính thức; #920 bổ sung từ kho |
| Keno | 82.546 kỳ trùng với danh sách vietlott.vn: giống hệt; kho thêm 211.126 kỳ (lấp 2019–2022 và các khoảng trống) |
| Bingo18 | 85.980 kỳ trùng: giống hệt; kho thêm 12.834 kỳ |
| Max 3D / Pro / 4D | giống hệt ở mọi kỳ trùng |
| Snapshot 1.3.0 | trùng toàn bộ với kết quả gộp (Keno 297.352, Bingo18 105.548 kỳ…), không thêm kỳ mới — bản v3.3 mới hơn (kho cộng đồng tới 02/10/2026) |

Lưu ý: phần Keno 2019–2022 chỉ có ở kho xoso.com.vn, không có bản chính thức để đối chiếu từng kỳ; bộ kiểm định chạy riêng phần này không thấy dấu hiệu lỗi chép số (bảng 1.6.4). Khi chạy được từ Việt Nam, `products sync --source vietlott --full` sẽ thay bằng dữ liệu chính thức nếu có.

**Phần nào của bộ 1.3.0 đã được kết hợp.** Dữ liệu (8 luồng kết quả, bảng giải, danh sách kỳ không xác nhận) và cách thu thập dự phòng của nó (kho NhanAZ khóa theo hash) — qua nguồn `nhanaz` (bản mới hơn) và `v130` (đúng snapshot của bộ 1.3.0). Bảng giải Keno chính thức của 1.3.0 trùng với bảng v3.3. Các chức năng còn lại của 1.3.0 (định giá/đối soát vé `product-quote`/`product-settle`, danh mục Bingo18 cùng kỳ, phân phối chung Lớn–Chẵn Keno, mô hình ML đã fit cho Mega/Power) không được chép sang; engine này đã có tính toán tương đương cho xác suất/RTP, bao và EV — dùng song song hai bộ không xung đột vì 1.3.0 chỉ được đọc.

### 1.8 vietlott.vn có Cloudflare "xác minh bạn là người": bạn giúp kiểm tra thế nào (v3.4)

**Engine làm gì.** Client HTTP nhận ra trang thử thách Cloudflare: mã 403/429/503 kèm header `cf-mitigated: challenge`, hoặc `server: cloudflare` cùng dấu hiệu trong trang (`cf-chl`, `challenge-platform`, Turnstile, "Just a moment"). Khi gặp, engine dừng ngay và chỉ cách làm bên dưới; `--source auto` chuyển sang nguồn dự phòng (mục 1.7). Engine **không** thử lại, không giải CAPTCHA, không dùng trình duyệt "tàng hình", và không chép cookie `cf_clearance` vào script. Bước xác minh là quyết định của website về truy cập tự động; vượt nó là đi ngược quyết định đó, và dễ khiến IP bị khóa.

**Bạn giúp được gì.** Bạn là người thật, ở Việt Nam: tự qua bước xác minh trong trình duyệt bình thường, rồi lưu lại trang đã tải. Engine chỉ đọc tệp bạn lưu.

1. `make checklist` → `reports/verification_checklist.md` (và `.csv`): 58 trang chia 4 nhóm, kèm đường dẫn và lý do.
2. Mở từng trang, bắt đầu từ nhóm 1. Qua bước xác minh, chờ trang kết quả hiện đủ, rồi lưu bằng một trong hai cách:
   - **Ctrl+S** (⌘S trên Mac) → kiểu "Trang web, một tệp" (.mhtml). Kiểu "Trang web, chỉ HTML" cũng đọc được.
   - **DevTools** (F12) ▸ tab Network, mở *trước* khi tải trang ▸ bật "Preserve log" ▸ chọn bộ lọc **All** (tệp xuất chỉ giữ các dòng đang hiện trong danh sách) ▸ mở chính các trang kết quả ▸ bấm nút mũi tên tải xuống **"Export HAR (sanitized)…"** trên thanh công cụ của tab Network (nếu cửa sổ DevTools hẹp, nút nằm sau dấu »). Đó là tên trên Chrome/Edge/Cốc Cốc bản mới (từ Chrome 130); bản cũ là chuột phải ▸ "Save all as HAR with content"; Firefox là chuột phải ▸ "Save All As HAR". Bản *sanitized* bỏ cookie và header đăng nhập nhưng vẫn giữ nội dung trang — đủ cho engine. Một tệp HAR chứa được nhiều trang, kể cả phản hồi AjaxPro của trang danh sách kết quả.
3. Bỏ các tệp vào một thư mục, rồi chạy `make import-pages PAGES=<thư-mục> DRY=1` để chỉ so sánh. Xem kết quả, rồi chạy lại không có `DRY=1` để nhập.

Hãy mở chậm như người đọc bình thường: vài chục trang là đủ, không cần hàng nghìn. Đừng dùng công cụ tự động hóa trình duyệt để lưu hàng loạt.

**Engine đọc gì từ tệp lưu** (`src/vietlott_engine/crawler/sources/saved_pages.py`):

| Trang | Lấy được | So với |
|---|---|---|
| Chi tiết kỳ Mega / Power / Lotto | bộ số và bảng giải (số người trúng, giá trị từng hạng, Jackpot) | kết quả và bảng giải đang có |
| Chi tiết kỳ Keno | 20 số, các ô giải theo bậc (bậc suy từ dòng "Trùng NN trong 20 số" lớn nhất mỗi bảng), ô cửa phụ | dữ liệu Keno và bảng giải của engine |
| Chi tiết Bingo18, Max 3D / Pro / 4D | kết quả kỳ | dữ liệu sản phẩm |
| Danh sách kết quả (trang, hoặc AjaxPro trong HAR) | nhiều kỳ một lúc | như trên |
| Chính trang thử thách Cloudflare | — | báo "lưu lại sau khi trang kết quả đã tải" |

Sản phẩm được nhận ra từ địa chỉ trang (dòng "saved from url" của Chrome, `Content-Location` của MHTML, canonical / og:url, URL trong HAR), tên phần AjaxPro, hoặc `GameId`.

Lệnh in số kỳ so được, số kỳ giống hệt và **cận trên tỷ lệ sai** (Clopper–Pearson một phía, 95%). Mỗi kỳ lệch được in đủ hai bản. Ví dụ: 30 kỳ Keno 2019–2022 cùng khớp nghĩa là tỷ lệ sai của kho xoso.com.vn ≤ 9,5% với độ tin cậy 95%; 0 kỳ lệch trên 300 kỳ đưa cận xuống ≈ 1%.

Khi nhập, kỳ mới được ghi với nguồn `vietlott.vn/saved`. Kỳ đã có và giống hệt được ghi lại, không đổi giá trị. Kỳ **lệch** với dữ liệu bị giữ lại, không ghi đè, vì parser đọc markup có thể đổi. Bạn xem từng chỗ, rồi thêm `--replace-mismatches` nếu trang đúng.

| Nhóm trong danh sách | Vì sao |
|---|---|
| 1 · ưu tiên | Lotto #920 chỉ có từ trang phụ. Lotto #874 có Jackpot lệch giữa nguồn (30,22 tỷ bản ghi chính thức vs 35,96 tỷ trang phụ). Keno #297782: ô bậc 5 trùng 4 số là 150.000 đ hay 0 đ, và các cửa phụ. 4 kỳ Keno và 1 kỳ Bingo18 gần nhất chỉ có từ trang phụ. |
| 2 · mẫu Keno 2019–2022 | 30 kỳ ngẫu nhiên (hạt giống cố định) của phần chỉ có ở kho xoso.com.vn |
| 3 · mẫu Bingo18 | 10 kỳ ngẫu nhiên kho ghi nguồn 'unknown' |
| 4 · khoảng trống Keno | 10 mã kỳ không có trong dữ liệu: có kỳ quay này không? |

Vì sao việc này có giá trị: trong kho cộng đồng, dòng lấy từ trang vietlott.vn (`official_vietlott`) dừng ở 18–20/08/2026 (Mega #1551, Keno #292654); từ đó kho chỉ dựa vào xosominhngoc. Điều này phù hợp với việc bộ thu của kho cũng bị chặn, nhưng không chứng minh được. Dữ liệu gộp của engine lấp phần lớn khoảng đó bằng bản sao danh sách vietlott.vn (tới 28/09/2026), nên phần **chỉ** có từ trang phụ còn lại là Lotto #920, 190 kỳ Keno (14/09–02/10, chủ yếu sau 28/09) và 195 kỳ Bingo18 (28/09–02/10). Một trang danh sách Keno/Bingo18 lưu bằng HAR chứa nhiều kỳ, nên vài trang là đủ để đối chiếu phần này.

> **Giới hạn.** Parser được viết theo cấu trúc trang mà các crawler công khai đã dùng và được test bằng HTML mẫu. Nó **chưa chạy trên trang thật** lưu sau Cloudflare. Một hai tệp lưu thật là đủ để hiệu chỉnh. HAR xuất kiểu *sanitized* đã bỏ cookie; HAR "with sensitive data" thì không — đừng chia sẻ loại đó.


### 1.9 Dự báo tự học cho kỳ tiếp theo (v3.5, `vietlott forecast`, `reports/forecast.md`)

**Có dự đoán được kết quả với "xác suất cao" không?** Không, nếu máy quay công bằng: mọi bộ số có cùng xác suất, và không thuật toán nào đổi được điều đó. Thứ làm được, và engine làm, là một bộ dự báo *tự học trung thực*. Nó học từ mọi kỳ và cho xác suất từng số kỳ tới. Nó tự đo xem kết quả có lệch khỏi máy quay công bằng theo cách nó đoán không, nói thẳng kết quả, và ghi từng dự báo kèm thời điểm vào sổ trước kỳ quay để chấm sau.

**Chuyên gia.** Mỗi chuyên gia đưa ra một phân phối xác suất *đầy đủ* cho kỳ tới, chỉ từ các kỳ trước (`src/vietlott_engine/forecast/experts.py`):

| Nhóm | Trò chọn số (Mega, Power, Lotto, Keno) | Trò chữ số (Max 3D / Pro / 4D, Bingo18, số đặc biệt Lotto) |
|---|---|---|
| Ngẫu nhiên | máy quay công bằng | máy quay công bằng |
| Nóng / lạnh | tần suất cả lịch sử (2 mức co về đều), gần đây (nửa đời 10 và 100 kỳ); đảo dấu = "số lạnh" | Dirichlet theo vị trí và gộp các vị trí (2 mức), gần đây (20 và 200 kỳ), "chữ số lạnh" |
| Gan / lặp | số lâu chưa về (±), số vừa về kỳ trước (±) | chữ số kỳ trước (±) |
| Học máy | hồi quy logistic học trực tuyến (AdaGrad; 6 đặc trưng: 3 tần suất, độ gan, có mặt ở 1 và 2 kỳ trước) | Dirichlet đã là bộ học tối ưu cho một độ lệch cố định |
| Giả thuyết thưa | từng số lệch ±10% / ±25% (4n giả thuyết) | từng chữ số ở từng vị trí lệch ±10% / ±25% |

Luật xác suất: k số khác nhau theo phân phối Bernoulli có điều kiện P(S) ∝ Π w_i, chuẩn hóa chính xác bằng đa thức đối xứng sơ cấp. Số phụ Power rút từ n − k số còn lại. Ở trò chữ số, các vị trí độc lập. Mọi chuyên gia đều là phân phối hợp lệ trên toàn bộ kết quả có thể (có test vét cạn).

**Tự học.** Hỗn hợp Bayes *fixed-share*: sau mỗi kỳ, trọng số của mỗi chuyên gia nhân với xác suất nó đã cho kết quả thật. Một phần nhỏ a = 1 / (10 × số kỳ mỗi năm) được trộn lại với prior (25% cho "ngẫu nhiên"), nên mô hình vẫn đổi được chuyên gia nếu máy quay thay đổi (khoảng một lần đổi mỗi 10 năm; mỗi lần đổi tốn ≈ log(1/a) bằng chứng). Mức này đặt theo thiết kế. Đã thử thêm 2 năm và không đổi trên cùng dữ liệu: kết luận có / không bằng chứng của mọi sản phẩm giống nhau ở cả ba mức, và hai phát hiện (Max 3D, Max 3D Pro) vẫn vượt ngưỡng chặt hơn 60 = 20 × 3 mức đã thử. Trạng thái lưu ở `data/forecast/<sản phẩm>.json` và học tăng dần: học từng phần, lưu rồi đọc lại cho kết quả giống hệt học một lần (có test).

**Bằng chứng.** Tích các tỷ số "xác suất hỗn hợp / xác suất máy công bằng" qua các kỳ là một martingale không âm dưới giả thuyết máy công bằng. Đó là **e-value hợp lệ ở mọi thời điểm** (bất đẳng thức Ville): xem sau mỗi kỳ bao nhiêu lần cũng được, không cần hiệu chỉnh; vượt 20 (10^1,30) là bằng chứng ở α = 5%. Mô phỏng 200 lịch sử 500 kỳ: máy công bằng 6/45 vượt 20 ở 1,5%, kiểu Max 3D ở 2,0%. Khi chữ số 6 hàng đơn vị ra 11% thay vì 10% (cỡ lệch thấy ở Max 3D), mô hình tìm ra ở 88% số lịch sử dài 1.139 kỳ và 52% số lịch sử dài 786 kỳ (100 lịch sử mỗi loại).

Điều kiện để e-value hợp lệ: điểm bắt đầu học phải định trước khi xem dữ liệu. Vì vậy `forecast fit --last N` (học N kỳ cuối, để chạy nhanh) đánh dấu kết quả là **không dùng làm bằng chứng**: chọn N sau khi xem dữ liệu có thể tạo ra "bằng chứng" giả (kiểm tra lại đã thấy một cửa sổ như thế ở Lotto và ở Max 4D).

**Chấm điểm.**
- *Chấm lùi:* ở mỗi kỳ quá khứ, lựa chọn của mô hình được chấm với kết quả thật. Lựa chọn là k số có trọng số trung bình cao nhất; với trò chữ số là chữ số mạnh nhất ở mỗi vị trí. Kỳ vọng ngẫu nhiên là k²/n số, hoặc m·L/A chữ số mỗi kỳ.
- *Sổ dự báo:* `forecast next` ghi dự báo kèm mã băm và thời điểm vào `ledger.jsonl` trong thư mục dự báo: `data/local/forecast/` trên máy cài bằng bộ cài (đặt trong `.env`), `data/forecast/` khi chạy trên GitHub Actions (sổ công khai được commit vào repo). Hai sổ tách nhau nên chạy trên máy không làm đổi tệp nào của repo; `forecast update` chấm sau kỳ quay; `forecast scoreboard` cho thành tích thật. Chỉ dự báo đầu tiên của mỗi kỳ được tính.
  - Sổ là tệp cục bộ để tự theo dõi, không phải bằng chứng chống sửa.
  - Dự báo nhắm kỳ ngay sau kỳ cuối có trong dữ liệu. Từ v4.0, sổ chỉ nhận dự báo khi kỳ đó **chưa quay** theo lịch chính thức (`forecast/schedule.py`): Mega, Power, Max 3D, Max 3D Pro trước 18:00 ngày quay; Lotto trước 13:00 / 21:00; Keno, Bingo18 chỉ trong khoảng 22:15–05:55 và khi ngày trong dữ liệu đã đủ kỳ; Max 4D đã ngừng nên không ghi. `--force-record` vẫn ghi nhưng không tính vào thành tích. API `?record=true` theo cùng quy tắc.

**Kết quả trên dữ liệu thật** (mọi kỳ có trong `data/seed`, học từ kỳ đầu tiên):

| Sản phẩm | Kỳ đã học | e-value cao nhất / hiện tại | Bằng chứng? | Trọng số "ngẫu nhiên" | Chuyên gia khác nặng nhất | Chấm lùi: trùng × ngẫu nhiên (z) |
|---|---:|---|---|---:|---|---|
| Mega 6/45 | 1.569 | 10^0,20 / 10^−0,46 | không | 50,8% | cold_all_a500 (3,1%) | ×0,969 (−1,26) |
| Power 6/55 | 1.405 | 10^0,33 / 10^−0,29 | không | 35,2% | tilt[6]−0,25 (9,3%) | ×0,982 (−0,61) |
| Lotto 5/35 | 920 | 10^0,12 / 10^−0,46 | không | 37,1% | tilt[34]+0,25 (9,0%) | ×1,014 (0,40) |
| Keno | 297.367 | 10^0,18 / 10^−0,83 | không | 99,2% | logistic (0,1%) | ×1,000 (0,28) |
| Bingo18 | 105.563 | 10^0,12 / 10^−0,65 | không | 98,4% | dirichlet_pooled_a1000 (0,4%) | ×1,003 (0,76) |
| Max 3D / 3D+ | 1.139 | 10^2,57 / 10^1,67 | **có** | 5,7% | tilt[đơn vị=6]+0,10 (73,8%) | ×1,039 (3,39) |
| Max 3D Pro | 786 | 10^2,84 / 10^1,45 | **có** | 12,0% | tilt[đơn vị=6]+0,10 (78,1%) | ×1,026 (1,87) |
| Max 4D (đã ngừng) | 722 | 10^0,72 / 10^−0,15 | không | 32,7% | tilt[trăm=8]−0,10 (16,9%) | ×1,028 (1,25) |

- **Sáu sản phẩm không có tín hiệu.** Keno và Bingo18, nhiều dữ liệu nhất, dồn 99% và 98% trọng số về "ngẫu nhiên". Các hệ số mô hình tự cho, như bộ Mega ×1,02 hay số 34 của Lotto ×1,02, là dao động của dữ liệu và chưa được e-value xác nhận; báo cáo luôn ghi rõ điều này.

  Bảng dưới xem các cách chọn số phổ biến như mô hình xác suất, trên 297 nghìn kỳ Keno. Cột bên phải là log10 tỷ số hợp lý so với máy quay công bằng: mô hình gán xác suất cho kết quả thật kém máy công bằng bao nhiêu bậc 10. Số càng âm, mô hình càng sai.

  | Mô hình (Keno) | log10 tỷ số hợp lý |
  |---|---:|
  | Số nóng, nửa đời 10 kỳ | −33.309,5 |
  | Số lâu chưa về | −22.152,3 |
  | Lặp số kỳ trước | −7.451,6 |
  | Số lạnh cả lịch sử | −160,1 |
  | Số nóng cả lịch sử | −131,1 |
  | Hồi quy logistic học trực tuyến | −51,8 |

  Đây là độ khớp của *mô hình*, không phải tiền thắng thua của *vé*: nếu máy công bằng, một vé chọn theo số nóng vẫn có xác suất trúng như mọi vé khác. Độ lớn chủ yếu phản ánh mô hình nghiêng mạnh đến đâu. Mô hình logistic học được nên tự nghiêng rất ít, vì thế kém ít nhất, nhưng vẫn kém máy công bằng.
- **Max 3D và Max 3D Pro.** Mô hình chỉ ra chữ số 6 hàng đơn vị: giả thuyết "đơn vị = 6, +10%" nhận 74% và 78% trọng số. Nó không được lập trình để tìm chữ số này, vì giả thuyết lệch được đặt đều cho mọi chữ số ở mọi vị trí. Nhưng nó chạy trên cùng dữ liệu đã cho phát hiện ở mục 1.6, nên đây là cùng bằng chứng nhìn theo cách khác, không phải xác nhận độc lập. Trên Max 4D nó không thấy độ lệch này.
  - Max 3D: e-value lần đầu vượt 20 ở kỳ #961 (08/08/2025). Từ đó nó ở trên ngưỡng 81% trong 179 kỳ, và liên tục trên ngưỡng từ #1032 (21/01/2026). Đỉnh 10^2,57 ở #1104 (10/07/2026), nay 10^1,67. Mức giảm từ đỉnh này thường gặp cả khi độ lệch không đổi (17% lịch sử mô phỏng).
  - Max 3D Pro: lần đầu vượt 20 ở #316 (23/09/2023). Từ đó chỉ 47% trong 471 kỳ ở trên ngưỡng; liên tục trên ngưỡng từ #670 (01/01/2026). Đỉnh 10^2,84 ở #359 (02/01/2024), nay 10^1,45. Mức giảm này hiếm nếu độ lệch không đổi (3% lịch sử mô phỏng), nên gợi ý độ lệch yếu hơn sau 2024. Đây chỉ là gợi ý, vì mốc đỉnh được chọn sau khi xem dữ liệu.
  - Giá trị thực tế: chữ số 6 ×1,075, số đề xuất 426 ×1,084, nên RTP Max 3D ≈ 0,59 thay vì 0,545. Phần ×1,009 nhỉnh thêm từ hai chữ số đầu chưa có bằng chứng; chỉ tính đuôi 6 thì RTP ≈ 0,586. Max 3D Pro: cặp hai số đuôi 6 có RTP ≤ 0,63 theo mô hình, so với 0,547. **Vẫn mất trung bình ~41% tiền vé.**

```bash
vietlott forecast next                      # mọi sản phẩm: học kỳ mới, dự báo kỳ tới, ghi vào sổ
vietlott forecast next --product keno       # một sản phẩm (lần đầu Keno học ~300 nghìn kỳ, khoảng 45 giây)
vietlott forecast update                    # sau kỳ quay: học kỳ mới, chấm các dự báo đã ghi
vietlott forecast scoreboard                # thành tích thật của các dự báo đã ghi trước kỳ quay
vietlott forecast fit --product mega645     # học lại từ đầu (sau khi dữ liệu cũ được bổ sung)
make forecast   make forecast-update   make forecast-report
```

API: `GET /forecast/{sản phẩm}?record=true` (học kỳ mới, trả dự báo, ghi sổ), `GET /forecast/{sản phẩm}/evidence` (đường e-value theo kỳ), `GET /forecast/{sản phẩm|all}/scoreboard`, `POST /forecast/{sản phẩm}/fit?last=N`. Bộ lập lịch trong `docker-compose.yml` gọi `/forecast/…?record=true` sau mỗi lần đồng bộ, nên mô hình tự học và sổ dự báo tự đầy mà không cần thao tác.


---

## 2. Kiến trúc

```
vietlott-quant-engine/
├── install.cmd  install.ps1  install.sh   # bộ cài (Windows / Linux, macOS) → .venv + lệnh tắt vietlott(.cmd)
├── src/vietlott_engine/   # gói Python (pip install -e .; lệnh `vietlott`), paths.py tìm thư mục dự án
│   ├── core/            # GameSpec (luật 3 game, cơ cấu giải, thuế, trần 300 tỷ, luật chia giải), Draw/Ticket, DrawHistory, PrizeHistory,
│   │                    # products (8 sản phẩm, ProductHistory cho Keno/Bingo18/Max 3D/Max 4D, danh sách loại trừ)
│   ├── crawler/         # httpx async + retry/backoff/rate-limit, mirror / vietlott.vn (7 sản phẩm, trang chi tiết) / file,
│   │                    # official_data (bản ghi chính thức), product_store (kho + đồng bộ sản phẩm), DuckDB + Parquet;
│   │                    # sources/nhanaz (kho cộng đồng), sources/vqe130 (snapshot 1.3.0), sources/fallback (chuỗi auto),
│   │                    # sources/saved_pages (trang/HAR người dùng lưu từ vietlott.vn → đối chiếu → nhập); http nhận ra Cloudflare
│   │   └── sources/
│   ├── analytics/       # χ² hiệu chỉnh, entropy, subset-sum, runs, Ljung–Box, gap/hazard, ma trận đồng xuất hiện,
│   │                    # products (kiểm định Keno/Bingo18/Max 3D, kiểm tra chéo chữ số)
│   ├── probability/     # Dirichlet–Multinomial + time-decay, Bayes factor, Markov (ma trận chuyển, chuỗi trạng thái)
│   ├── inference/       # v2: power/TOST, Bayes phân cấp, Westfall–Young, e-process, điểm gãy, DM-HAC, SPA
│   ├── game_theory/     # hành vi người chơi (prior + hiệu chỉnh NB), doanh số & tỷ lệ trả thưởng, EV, tối ưu chống trùng số,
│   │                    # decision (bất định + Kelly), bao (danh mục, bảng giải, EV, so sánh), max3d (Max 3D/3D+/Pro),
│   │                    # coverage (P ≥ 1 giải), rolldown (Lotto chia giải, tích lũy hai pha, dự báo), market,
│   │                    # fastgames (Keno, Bingo18: bảng giải, xác suất, RTP)
│   ├── wheeling/        # covering design L(v,k,m,t): greedy + SA + ILP (HiGHS) có chứng nhận cận dưới
│   ├── ml_models/       # feature/graph causal (SnapshotBuilder), GCN NumPy, walk-forward skill, baseline logistic
│   ├── backtest/        # chiến thuật + engine walk-forward chống look-ahead, suy luận theo cụm + SPA
│   ├── forecast/        # v3.5: dự báo tự học mọi sản phẩm — data (chuỗi quan sát), experts, engine (fixed-share, e-value, sổ dự báo)
│   ├── api/             # FastAPI (routers: data, analytics, strategy, inference, products, max3d, catalog = /products, forecast)
│   └── cli.py           # sync | prizes | products | forecast | market | odds | analyze | inference | ev | decide | optimize | bao
│                        # | max3d | coverage | rolldown | wheel | backtest | ml | serve
├── scripts/             # build_official_dataset.py (dựng data/seed từ bản ghi chính thức), build_prize_dataset.py (nguồn v3),
│                        # market_report.py, bao_report.py, products_report.py, verification_checklist.py (trang nên kiểm bằng tay), forecast_report.py,
│                        # build_site.py (trang tĩnh + JSON cho GitHub Pages)
├── .github/workflows/   # ci.yml (test), installer.yml (bộ cài trên 3 hệ điều hành), update.yml (đồng bộ, dự báo, sổ, Pages), release.yml (tag → Release)
├── docs/INSTALL.md      # hướng dẫn cài đặt, đưa lên GitHub, phát hành, sự cố thường gặp
├── tests/               # 178 test (pytest)
├── data/seed/           # snapshot: kết quả + bảng giải 3 game, Keno, Bingo18, Max 3D, Max 3D Pro, Max 4D, exclusions.json,
│                        # keno_rules_official.json, OFFICIAL_DATA_REPORT.json, DATA_MERGE_REPORT.json
├── data/calibration/    # mô hình hành vi + thị trường đã hiệu chỉnh (behaviour_*.json, market_*.json, tickets_sold_*.jsonl)
├── reports/             # báo cáo sinh ra từ dữ liệu thật
├── Dockerfile  docker-compose.yml  pyproject.toml  requirements.txt  Makefile  .env.example  CHANGELOG.md
```

Luồng dữ liệu: `Source → Pydantic validation → integrity check → DuckDB (upsert) → Parquet` → `DrawHistory` (NumPy, read-only) → mọi mô hình. Mỗi module chỉ nhận `DrawHistory`, không biết dữ liệu đến từ đâu.

---

## 3. Cài đặt & chạy

### 3.1 Local

Bộ cài làm mọi bước (môi trường ảo `.venv`, cài gói, `.env`, nạp dữ liệu, học bộ dự báo, kiểm tra) — xem [docs/INSTALL.md](docs/INSTALL.md):

```bash
bash install.sh              # Linux / macOS   (Windows: install.cmd)
bash install.sh --dev        # thêm pytest/ruff/mypy và chạy bộ test
```

Cài tay, nếu muốn:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
vietlott init                # nạp dữ liệu đi kèm, học bộ dự báo, kiểm tra
```

Các lệnh `make` dùng `.venv` nếu có:

```bash
make doctor        # kiểm tra Python, gói, dữ liệu, kho, bộ dự báo, mạng
make seed          # nạp snapshot offline (kết quả + dữ liệu giải thưởng) vào DuckDB + xuất Parquet
make sync          # cập nhật tăng dần từ nguồn (mặc định: GitHub mirror)
make prizes        # bảng giải chính thức (số người trúng từng giải + Jackpot)
make products-sync # Keno, Bingo18, Max 3D, Max 3D Pro (SOURCE=auto|vietlott|nhanaz|mirror|canonical|v130)
make import-v130 V130=~/Vietlott-Quant-Engine-1.3.0   # nhập snapshot của bộ 1.3.0
make market        # hiệu chỉnh hành vi người chơi + mô hình thị trường → data/calibration
make report        # reports/vietlott_v3.{md,json}
make products-report  # reports/products.{md,json}
make checklist     # reports/verification_checklist.{md,csv}: trang vietlott.vn nên kiểm bằng tay (mục 1.8)
make import-pages PAGES=~/Downloads/vietlott DRY=1   # đối chiếu trang bạn đã lưu; bỏ DRY=1 để nhập
make forecast      # dự báo tự học kỳ tới, mọi sản phẩm (mục 1.9); make forecast-update sau kỳ quay
make test          # 178 test
```

### 3.2 Đồng bộ dữ liệu

```bash
vietlott sync --game all                               # GitHub mirror (mặc định)
vietlott sync --game mega645 --source vietlott         # trực tiếp vietlott.vn (AjaxPro) — chỉ từ IP Việt Nam (mục 1.6.1)
vietlott sync --game all --source file --path data/seed # offline
vietlott sync --game power655 --full                   # tải lại toàn bộ
vietlott prizes --game all                             # bảng giải chính thức (mặc định --source canonical)
vietlott prizes --game all --source vietlott --last 30  # trang chi tiết 30 kỳ gần nhất (từ VN)
vietlott products sync                                  # Keno, Bingo18, Max 3D, Max 3D Pro (auto, mục 1.7)
vietlott sync --game all --source auto                  # Mega/Power/Lotto: vietlott.vn → kho cộng đồng → mirror
python scripts/build_official_dataset.py --vendor .vendor --update  # dựng lại data/seed (git clone, không cần raw.githubusercontent.com)
```

Mỗi lần sync: tăng dần theo `draw_id`, validate từng dòng bằng Pydantic (dòng lỗi được báo cáo, không bị bỏ qua âm thầm — ví dụ kỳ Power #00944 trong mirror thiếu số đặc biệt), kiểm tra toàn vẹn (trùng id, ngày đảo thứ tự, khoảng trống id, ngày quay sai lịch), upsert DuckDB và ghi `data/parquet/<game>.parquet`. Mirror Mega 6/45 bắt đầu từ kỳ #00198 (thiếu 197 kỳ đầu) — báo cáo toàn vẹn sẽ hiển thị khoảng trống này.

HTTP client: token bucket (`VQE_RATE_LIMIT_PER_S`), semaphore giới hạn đồng thời, retry exponential backoff + full jitter cho 408/425/429/5xx và lỗi mạng, tôn trọng `Retry-After`; proxy từ chối (403/407) là chính sách mạng nên **không** retry. Trang thử thách Cloudflare ("xác minh bạn là người") cũng **không** retry và không bị vượt: lỗi chỉ sang mục 1.8. Proxy riêng: biến chuẩn `HTTPS_PROXY`.

### 3.3 Phân tích & backtest (CLI)

```bash
vietlott analyze --game all --sims 2000
vietlott inference --game all --sims 2000 --out reports   # chứng nhận công bằng (v2)
vietlott decide --game mega645 --ticket 33 35 37 40 42 45 --jackpot1 100e9 \
       --tickets-sold 4000000 --bankroll 1e9 --sold-range 2000000 6000000   # phân phối, Kelly, bất định
vietlott backtest --game all --tickets 5 --out reports \
       --strategies random hot cold overdue bayes markov gcn anti_popularity wheel
vietlott ml --game all                       # kỹ năng GCN vs logistic, walk-forward
vietlott ev --game power655 --ticket 3 17 22 35 41 44 --jackpot1 120e9 --tickets-sold 3000000
vietlott optimize --game mega645 --n 5 --jackpot1 45e9
vietlott wheel --game mega645 --pool 3 7 11 15 19 23 27 31 35 39 42 44 --t 4 --m 5

# v3 — sản phẩm Vietlott & thị trường
vietlott market                              # s, vé bán, hành vi người chơi, doanh số, chia giải → data/calibration
vietlott odds --game all                     # xác suất chính xác từng hạng giải
vietlott ev --game lotto535 --ticket 3 14 22 29 33 --special 7 --jackpot1 9e9
vietlott ev --game mega645 --ticket 30 31 40 41 44 45 --jackpot1 158.8e9   # vé bán: mô hình doanh số
vietlott bao --game mega645 --numbers 5 12 19 26 33 40 44 45                # Bao 8
vietlott bao --game power655 --numbers 4 15 26 37 48                        # Bao 5
vietlott bao --game lotto535 --numbers 5 12 19 26 33 --specials 1 2 3 4 5 6   # bao số đặc biệt
vietlott bao --game power655 --catalog                                     # mọi kiểu bao + giá
vietlott bao --game power655 --table 7                                     # bảng giải Bao 7
vietlott bao --game mega645 --numbers 3 9 14 22 31 38 41 44 --compare      # bao vs rút gọn vs vé lẻ
vietlott bao --game mega645 --numbers 3 9 14 22 31 38 41 --jackpot1 158.8e9 --tickets-sold 6300000 --after-tax
vietlott max3d --product max3dpro --kind bao_bo_so --numbers 123 456
vietlott max3d --product max3dpro --kind bao_nhieu_bo_so --numbers 123 456 789
python scripts/bao_report.py                                                        # reports/bao_vietlott.md
vietlott coverage --game mega645 --budget 28 [--min-tier second]
vietlott rolldown --history                  # chuỗi Jackpot chính thức: 17 kỳ chia, hai pha, dự báo kỳ tới
vietlott rolldown --jackpot 22e9 --tickets-sold 2500000

# v3.2 — toàn bộ sản phẩm
vietlott products list|sync|analyze|odds [--product keno|bingo18|max3d|max3dpro]
```

EV/optimize/decide tự dùng mô hình đã hiệu chỉnh trong `VQE_CALIBRATION_DIR` (mặc định `data/calibration`) và mô hình doanh số khi không truyền `--tickets-sold`; `--quick-pick` để ghi đè.

**Backtest không look-ahead bias — cơ chế bảo đảm:**

1. Với mỗi kỳ mục tiêu *t*, engine tạo `history.upto(t).detached()`: một **bản sao độc lập, read-only** chỉ chứa các kỳ < *t*. Vì là bản sao, chiến thuật không thể truy cập tương lai kể cả qua `ndarray.base`.
2. Feature và đồ thị của GCN được sinh bởi `SnapshotBuilder` — bộ lọc đệ quy nhân quả; test xác nhận snapshot *t* tính trên toàn bộ lịch sử trùng khớp tuyệt đối với snapshot tính trên tiền tố.
3. Vé trả về được kiểm tra (số lượng, miền giá trị, trùng số); vi phạm ném `LookAheadError`.
4. Suy luận: so với **null siêu bội chính xác** (không phải một lần chạy random) với sai số chuẩn **theo cụm** (kỳ quay là đơn vị độc lập — dưới H₀ trung bình số trúng mỗi kỳ là một martingale difference); BH + Holm qua các chiến thuật; e-process anytime-valid cho từng chiến thuật (Holm qua các chiến thuật); cận tương đương TOST và MDE; **SPA/Reality Check** với stationary bootstrap cho câu hỏi "chiến thuật tốt nhất"; ROI có khoảng tin cậy block-bootstrap.

### 3.4 API (FastAPI)

```bash
vietlott serve --port 8000        # hoặc: uvicorn vietlott_engine.api.main:app
# Swagger UI: http://localhost:8000/docs
```

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health`, `/games` | Trạng thái, luật chơi (3 game), xác suất từng giải |
| GET | `/games/{game}/odds` | Xác suất chính xác từng hạng giải, "1 trên X", số lượt của từng kiểu bao |
| GET | `/games/{game}/prizes?limit=` | Số vé trúng từng giải (+ Jackpot) các kỳ gần nhất |
| POST | `/games/{game}/prizes/sync` | Bảng giải: `{"source": "auto"}` (vietlott → canonical → nhanaz → v130), hoặc một nguồn cụ thể, vd. `{"source": "vietlott", "last": 30}` |
| GET | `/games/{game}/market` | Tỷ lệ trả thưởng, vé bán, mô hình doanh số, hành vi người chơi đã hiệu chỉnh |
| GET | `/games/{game}/bao/catalog` | Mọi kiểu bao Vietlott bán: số chọn, số lượt, giá |
| GET | `/games/{game}/bao/table?level=7&specials=1` | Bảng giải thưởng kiểu Vietlott của một mức bao |
| POST | `/games/{game}/bao` | Phân phối chính xác vé bao `{"numbers": [...], "specials": [...], "tax_basis": "ticket", "strict": false, "tickets_sold": 3600000}` |
| POST | `/games/{game}/bao/compare` | Bao vs bao rút gọn vs vé lẻ dàn đều vs quick pick `{"numbers": [...], "guarantee": 3, "condition": 3}` |
| GET | `/products` | 7 sản phẩm: luật quay, lịch, trang kết quả vietlott.vn, dữ liệu đang có |
| GET | `/products/keno/odds`, `/products/bingo18/odds` | Xác suất chính xác + RTP mọi bậc/cửa (kèm các phiên bản bảng giải khác nhau) |
| GET | `/products/{product}/draws?limit=`, `/products/{product}/randomness` | Kết quả gần nhất; bộ kiểm định Keno / Bingo18 / Max 3D (BH) |
| GET | `/products/max3d/digit-check` | Lệch chữ số mạnh nhất của Max 3D, kiểm định ngoài mẫu trên Max 3D Pro |
| POST | `/products/{product}/sync` | `{"source": "auto"|"vietlott"|"nhanaz"|"mirror"|"canonical"|"v130", "max_pages": 50, "full": false}` → báo cáo có `source_used`, `attempts` |
| GET | `/max3d`, `/max3d/{product}` | Max 3D / 3D+ / 3D Pro: bảng giải, xác suất chính xác, RTP |
| POST | `/max3d/{product}/analyse` | Một lượt hoặc bao `{"kind": "bao_bo_so", "numbers": ["123", "456"]}` |
| POST | `/games/{game}/coverage` | Danh mục B vé tối đa P(trúng ≥ 1 giải) `{"budget": 28, "min_tier": null}` |
| POST | `/games/lotto535/rolldown/ev` | EV kỳ chia giải + doanh số hòa vốn `{"jackpot": 2.2e10, "tickets_sold": 2500000}` |
| GET | `/games/lotto535/rolldown/official` | Kỳ chia giải từ chuỗi Jackpot chính thức, tích lũy hai pha, dự báo kỳ chia tiếp theo + backtest |
| GET | `/games/lotto535/rolldown/history` | Bản tái dựng v3 (đột biến doanh số): pot, giá trị từng giải, RTP |
| GET | `/games/{game}/draws?limit=` | Kết quả gần nhất + khoảng trống dữ liệu |
| POST | `/games/{game}/sync` | Đồng bộ tăng dần `{"source": "vietlott", "full_refresh": false}` |
| GET | `/games/{game}/analytics/randomness` | Bộ kiểm định + FDR |
| GET | `/games/{game}/analytics/gaps` | Gap từng số, kiểm định hình học, bảng hazard |
| GET | `/games/{game}/analytics/cooccurrence` | Cặp số nổi bật + q-value, over-dispersion |
| GET | `/games/{game}/probability/bayesian` | Posterior Dirichlet, khoảng tin cậy, Bayes factor |
| GET | `/games/{game}/probability/bayesian/decay-scan` | Chọn λ theo điểm dự báo walk-forward |
| GET | `/games/{game}/probability/markov` | Ma trận chuyển, kiểm định hoán vị, chuỗi trạng thái |
| GET | `/games/{game}/ml/gcn/skill`, `/ml/gcn/next` | Kỹ năng ngoài mẫu; xác suất kỳ tới (kèm cảnh báo) |
| POST | `/games/{game}/ev` | EV một vé: `{"ticket": [...], "special": 7, "jackpot1": 45e9}` (`special` cho Lotto; bỏ `tickets_sold` ⇒ mô hình doanh số) |
| POST | `/games/{game}/ev/optimize` | Bộ vé ít trùng nhất `{"n_tickets": 5, "max_overlap": 2}` |
| POST | `/games/{game}/ev/decision` | Phân phối tiền thưởng + Kelly `{"ticket": [...], "jackpot1": 1e11, "bankroll": 1e9}` |
| POST | `/games/{game}/ev/uncertainty` | EV dưới dạng phân phối + độ nhạy `{"ticket": [...], "jackpot1": 1e11, "tickets_sold_low": 2e6, "tickets_sold_high": 6e6}` |
| GET | `/games/{game}/inference/report` | Chứng nhận công bằng đầy đủ (v2) |
| GET | `/games/{game}/inference/{per-number,power,hierarchical,sequential,changepoints}` | Từng thành phần suy luận |
| POST | `/games/{game}/wheel` | Bao lô `{"pool": [...], "guarantee": 3, "condition": 4, "exact": true, "time_limit": 10}` |
| POST | `/games/{game}/backtest` | `{"strategies": ["random","bayes"], "tickets_per_draw": 5}` |

Lỗi nghiệp vụ trả JSON `{"error", "detail"}`: 422 (dữ liệu không hợp lệ), 409 (chưa có dữ liệu), 502 (nguồn lỗi), 404 (game không tồn tại). Đồng bộ thất bại vì bị chặn trả báo cáo có trường `error` nói rõ website chặn IP hay proxy mạng chặn.

### 3.5 Docker

```bash
docker compose build
docker compose up -d api                           # API :8000 + tự cập nhật theo lịch từng sản phẩm
docker compose --profile jobs run --rm backtest    # job nghiên cứu → ./reports
```

- `api`: khi volume trống, nạp snapshot có sẵn trong image (kết quả + dữ liệu giải thưởng), mô hình hiệu chỉnh trong `/app/calibration`, rồi sync tăng dần **ở nền** (API phục vụ ngay, không chặn khởi động nếu mạng lỗi).
- Bộ cập nhật chạy trong tiến trình API: kiểm tra mỗi 2 phút trong cửa sổ quay của từng game, retry độc lập và phục hồi sau restart. Xem [`docs/VLM_AUTO_UPDATES.md`](docs/VLM_AUTO_UPDATES.md), `GET /updates/status`. DuckDB do API sở hữu; dùng 1 worker. Workflow `results.yml` cào bù mỗi 10 phút ban ngày và mỗi giờ ban đêm, giữ journal kết quả trong repo khi cache mất; workflow dự báo/trang vẫn chạy hai lần/ngày.
- `backtest` (profile `jobs`): dùng file DuckDB riêng, không tranh khóa với API; đồng bộ bảng giải chính thức và sản phẩm, chạy lại hiệu chỉnh thị trường (`reports/calibration`), báo cáo v3, bao và sản phẩm trước backtest.
- Mở rộng đọc: các replica có thể đọc `data/parquet/*.parquet` (read-only) thay vì file DuckDB.

---

## 4. Phương pháp

### 4.1 Kiểm định ngẫu nhiên (`src/vietlott_engine/analytics`)
- **χ² tần suất hiệu chỉnh.** Trong một kỳ, 6 số được rút *không hoàn lại*, nên số lần xuất hiện của các số tương quan âm: Cov = c·(I − J/n), c = D·k(n−k)/(n(n−1)). Thống kê đúng là Σ(O−E)²·n(n−1)/(D·k(n−k)) ~ χ²(n−1), bằng χ² “sách giáo khoa” × (n−1)/(n−k). Test chứng minh p-value của bản hiệu chỉnh phân phối đều dưới H₀, còn bản ngây thơ thì không; trên dữ liệu thật, bản hiệu chỉnh khớp p-value Monte Carlo (0.880 vs 0.878).
- **Shannon entropy** của phân phối tần suất, p-value Monte Carlo một phía (máy lệch ⇒ entropy thấp), kèm hiệu chỉnh Miller–Madow và KL-divergence tới phân phối đều.
- **Phân phối chính xác**: số lẻ/số nhỏ (siêu bội), tổng (quy hoạch động đếm tập con), cặp liên tiếp (C(k−1,r)·C(n−k+1,k−r)/C(n,k)), số lặp từ kỳ trước (siêu bội). G-test với gộp bin đuôi.
- **Chuỗi thời gian**: runs test Wald–Wolfowitz, Ljung–Box Q(10).
- **Gap**: gap giữa hai lần xuất hiện ~ Geometric(k/n); bảng hazard với khoảng Wilson.
- **Đồng xuất hiện**: O_ij ~ Binomial(D, k(k−1)/(n(n−1))), p-value nhị thức chính xác, q-value BH, kiểm định over-dispersion toàn cục bằng Monte Carlo.

### 4.2 Bayesian & Markov (`src/vietlott_engine/probability`)
- **Dirichlet–Multinomial với time-decay**: α_i = α₀ + Σ_d λ^(D−1−d)·x_{d,i}; marginal Beta cho khoảng tin cậy; log marginal likelihood dạng đóng ⇒ Bayes factor so với mô hình đều; empirical-Bayes α₀; chọn λ bằng log-score dự báo một bước (bộ lọc đệ quy, không rò rỉ).
- **Markov**: ma trận P(j ∈ kỳ t+1 | i ∈ kỳ t) có shrinkage về k/n; ý nghĩa thống kê bằng **kiểm định hoán vị thứ tự các kỳ** (giữ nguyên từng kỳ, chỉ phá cấu trúc thời gian). Chuỗi trạng thái rời rạc (tercile tổng, số lẻ, số lặp): phân phối dừng, G-test bậc 1 vs bậc 0, mutual information (bit).

### 4.3 GCN trên đồ thị đồng xuất hiện (`src/vietlott_engine/ml_models`)
Đồ thị: đồng xuất hiện suy giảm theo thời gian → PPMI → Â = D^−½(A+I)D^−½. Node features: tần suất EWMA (half-life 10/30/100), log-gap, xuất hiện kỳ trước, điểm Markov. Mô hình 2 lớp (Kipf & Welling 2017) viết bằng NumPy với gradient giải tích (đã kiểm bằng sai phân hữu hạn), Adam, early stopping trên khối validation muộn hơn. Đánh giá **rolling-origin**, so log-loss với baseline k/n và với logistic regression không dùng đồ thị. Không cần PyTorch ⇒ image nhỏ.

### 4.4 Lý thuyết trò chơi & EV (`src/vietlott_engine/game_theory`)
- **Mô hình người chơi**: p(c) = q/C(n,k) + (1−q)·p_manual(c), với p_manual(c) ∝ Π w_i · f(c). Trọng số số đơn: thiên lệch ngày sinh (1–31), tháng (1–12), số may/xui (7, 8, 9, 6, 13; đuôi 8/9/6 và 4), sao chép kết quả kỳ trước. Hệ số tổ hợp f(c): cấp số cộng (1-2-3-4-5-6, 5-10-…-30), đường thẳng trên phiếu, chuỗi ≥ 4 số liên tiếp, lặp y nguyên kỳ trước, né cặp liên tiếp lẻ. Hằng số chuẩn hóa tính chính xác bằng đa thức đối xứng sơ cấp e_k(w) + bộ lấy mẫu chính xác cho phần hiệu chỉnh mẫu hình.
- **Các tham số mặc định là prior** (Henze & Riedwyl 1998; Simon 1999; Cook & Clotfelter 1993; Farrell et al. 2000 + văn hóa số của VN), **chưa được hiệu chỉnh trên dữ liệu Vietlott**. `calibrate_number_weights` ước lượng β bằng Poisson-MLE chính xác từ **số vé trúng giải ba theo từng kỳ** (Vietlott công bố trên trang chi tiết kỳ quay) và số vé bán: μ_t = N_t·[q·H(m) + (1−q)·e_m(w_W)·e_{k−m}(w_L)/e_k(w)], gradient giải tích, L-BFGS. Test khôi phục β cài sẵn với tương quan > 0.9. Nguồn mirror hiện chưa có số người trúng nên bước này cần bạn bổ sung dữ liệu.
- **EV**: EV = Σ P(giải)·net(giải cố định) + P(JP)·E_K[net(J/(1+K))], K ~ Poisson(N·p(c)), tính tổng pmf chính xác; Power JP2 dùng λ₂ = N·(p(c) + 5/C(n,k)). Thuế TNCN theo luật 2025 hiệu lực 01/07/2026: 10% trên phần vượt 20 triệu mỗi lần trúng (cấu hình được trong `TaxRule`). Có Jackpot hòa vốn (Brent).
- **Tối ưu**: simulated annealing tối thiểu Σ log p(c) với ràng buộc số trùng giữa các vé ≤ `max_overlap`, cập nhật gia tăng.
- **Quyết định (v2, `decision.py`)**: phân phối đầy đủ của tiền thưởng sau thuế (giải cố định + Jackpot chia theo K ~ Poisson), độ lệch chuẩn, P(trúng); Kelly đa kết cục G(f) = Σ p_o log(1 + f(X_o − 1)) với G′(0) = RTP − 1 ⇒ cược tối ưu = 0 khi RTP ≤ 1; EV dưới bất định: lấy mẫu prior cho quick-pick, thiên lệch ngày sinh/tháng, cường độ mẫu hình, sao chép kỳ trước, số vé bán (log-uniform), rồi báo cáo phân vị, P(RTP > 1) và độ nhạy Spearman.

### 4.5 Bao lô (`src/vietlott_engine/wheeling`)
Bài toán L(v, k, m, t): tập vé k số nhỏ nhất từ pool v số sao cho *nếu ≥ m số trúng nằm trong pool thì có vé trúng ≥ t số*. Greedy set-cover (bitmask + popcount vector hóa) → loại vé thừa → simulated annealing kiểu Nurmela–Östergård để bớt từng vé → kiểm chứng vét cạn mọi m-tập con + cận dưới Schönheim (khi m = t). Đạt đúng tối ưu đã biết C(7,3,2)=7, C(9,3,2)=12, C(10,4,3)=30, C(12,6,3)=15; với tham số lớn là heuristic — so với La Jolla Covering Repository nếu cần tối ưu tuyệt đối. Kết quả kèm mô phỏng Monte Carlo: P(điều kiện xảy ra), P(có giải), chi phí, kỳ vọng giải cố định.

v2: sau heuristic, bài toán được giải lại bằng **ILP set-cover (HiGHS, `scipy.optimize.milp`)** khi kích thước cho phép. ILP hoặc tìm thiết kế ít vé hơn, hoặc chứng minh tối ưu (ví dụ C(9,6,4) = 12), hoặc ít nhất trả về **cận dưới được chứng nhận** = max(Schönheim, cận đếm C(v,m)/Σ_{j≥t}C(k,j)C(v−k,m−j), cận đối ngẫu của ILP). Kết quả ghi rõ `optimality` = `proven_optimal` hoặc `gap` kèm khoảng cách.

### 4.6 Suy luận thống kê (`src/vietlott_engine/inference`, v2)
- **Power & MDE** (`power.py`): dưới đối thuyết p_i = (k/n)(1+ε_i), thống kê χ² hiệu chỉnh ~ χ² phi trung tâm với λ = D·k·(n−1)·Σε²/(n(n−k)) — đã đối chiếu với mô phỏng. Suy ra độ lệch nhỏ nhất phát hiện được và số kỳ cần thiết.
- **Kiểm định tương đương (TOST)**: thay vì "không bác bỏ công bằng", *chứng nhận* mọi số nằm trong k/n·(1 ± δ). Theo nguyên lý giao–hợp (intersection–union), khẳng định cho *tất cả* các số chỉ cần TOST mức α cho từng số, không cần hiệu chỉnh đa kiểm định. Cận được đổi thành tiền: RTP chính xác của vé khi 6 số ở cận trên, dùng mô hình rút thăm conditional-Bernoulli với odds khớp xác suất biên (IPF) và phân phối số trúng chính xác e_m(w_T)·e_{k−m}(w_R)/e_k(w).
- **Bayes phân cấp** (`hierarchical.py`): θ_i ~ Beta(μκ, (1−μ)κ), x_i ~ Binomial(D, θ_i); likelihood biên Beta-Binomial chính xác, prior log-uniform cho độ phân tán tương đối τ/μ; posterior của τ, Bayes factor "lệch vật chất (≥5%)" vs "công bằng", ước lượng co rút (partial pooling) cho từng số — thay thế nguyên tắc cho bảng xếp hạng số nóng/lạnh.
- **Đa kiểm định** (`multiple_testing.py`): Holm, BH, Benjamini–Yekutieli (đúng dưới phụ thuộc bất kỳ), Westfall–Young max-T step-down với phân phối null đồng thời mô phỏng (tính đến tương quan âm giữa các số do Σ = kD).
- **E-process / kiểm định bằng cá cược** (`sequential.py`): trước mỗi kỳ, "người cá cược" đặt tiền theo phân phối q_t trên C(n,k) kết quả, chỉ dựa vào quá khứ; tài sản nhân với q_t(S_t)/p₀(S_t). Dưới H₀ đây là martingale không âm ⇒ bất đẳng thức Ville: P(sup W ≥ 1/α) ≤ α. Có thể theo dõi sau **mỗi** kỳ mới mà không bị phạt nhìn nhiều lần. Trộn nhiều người cá cược (tần suất với nhiều mức nhớ, Markov) vẫn là martingale. Với chiến thuật vé bất kỳ: E_t = 1 + λ(M̄_t − k²/n) hợp lệ vì trung bình số trúng có kỳ vọng có điều kiện đúng bằng k²/n.
- **Điểm gãy** (`changepoint.py`): quét χ² hiệu chỉnh trên ~200 cửa sổ trượt (4 tháng → 2 năm) và CUSUM-bridge từng số; p-value hiệu chỉnh việc quét bằng Monte Carlo trên lịch sử công bằng cùng độ dài (không có look-elsewhere bias). Phát hiện được một giai đoạn lệch cài sẵn 200 kỳ.
- **So sánh dự báo** (`predictive.py`): Diebold–Mariano với phương sai HAC (Newey–West, băng thông Andrews) và hiệu chỉnh Harvey–Leybourne–Newbold; kiểm định hiệu chuẩn Spiegelhalter; bảng reliability.
- **Chống data-snooping** (`bootstrap.py`): stationary bootstrap (Politis–Romano) vector hóa; SPA của Hansen (bản lower/consistent/upper) và Reality Check của White.

### 4.7 Dữ liệu giải thưởng & hiệu chỉnh hành vi người chơi (`src/vietlott_engine/crawler/prize_sources.py`, `src/vietlott_engine/game_theory/calibration.py`, v3)
- **Nguồn (v3.2)**: bản ghi chính thức (`src/vietlott_engine/crawler/official_data.py`, schema canonical 1.0: kết quả, bảng giải, URL, SHA-256 của trang, PDF) ⇒ `Draw` + `PrizeRecord`, pot = giá trị Jackpot × max(1, số vé trúng). Cách v3 (giữ lại cho `--source compal`): bản ghi giải thưởng chỉ được nhận khi bộ số khớp đúng kết quả kỳ đó; số người trúng ưu tiên nguồn khóa theo mã kỳ (Compal123), Jackpot lấy từ bản ghi leoodz có *cùng vector số người trúng* trong cửa sổ ±3 vị trí (sửa lệch một kỳ); giá trị Jackpot mỗi người × số người trúng = pot.
- **Mô hình**: vé của đám đông = q·đều + (1−q)·p_manual, p_manual(c) ∝ Π w_i, log w_i = β_i + γ·x_i(t) (hiệu ứng động: sao chép kỳ trước, tần suất gần đây…), Lotto thêm phân phối số đặc biệt. Số người trúng hạng g ở kỳ t: y_tg ~ NegBin(μ = N_t·P_g(W_t; θ), φ_g). P_g tính chính xác bằng đa thức đối xứng sơ cấp (e_m(w_W)·e_{k−m}(w_L)/e_k(w)) với gradient giải tích.
- **N_t chưa biết** (Mega, Lotto): mỗi kỳ có log-scale ν_t tự do được profile ra (nhận diện từ *tỷ lệ* giữa các hạng giải); N_t biết trước (Power) thì dùng hiệu ứng năm. Phân tán φ_g theo moment, prior ridge, sai số **sandwich** (đã trừ Schur complement của ν), kiểm tra **ngoài mẫu theo thời gian** (20% kỳ cuối), **kiểm tra mức** không cần doanh số (số giải Ba sau khi trừ trung vị trượt vs độ phổ biến dự báo, p hoán vị), và **chuyển giao** Power → Mega khi game đích quá ít dữ liệu.
- **Tỷ lệ trả thưởng** (`sales.py`): dưới quay đều, E[y_m] = N·H_m (H_m siêu bội) cho *mọi* phân phối vé ⇒ kết hợp với đồng nhất thức ΔJ_t + F_t = s·R_t ta được s bằng phương pháp moment trên từng hạng giải; ba hạng cho cùng một s là kiểm chứng độc lập. Từ đó N_t = (ΔJ_t + F_t)/(s·giá vé) cho từng kỳ Power và (v3.2, có chuỗi Jackpot chính thức) Mega. Khi không có chuỗi Jackpot: dựng lại đường Jackpot từ N̂_t và khép theo giá trị công bố. **Chọn mô hình đám đông cho Mega** (quy tắc đặt trước): giữ mô hình riêng trừ khi nó không qua kiểm tra ngoài mẫu (p > 0,05) *và* mô hình chuyển từ Power có tương quan kiểm tra mức cao hơn. Mô hình doanh số log N = a + b·x + c·x² + thứ + năm (x = log Jackpot − trung vị, sai số HAC), ngoại suy theo tiếp tuyến ngoài miền dữ liệu.
- **EV mới**: K ~ Poisson(N·p(c)) với N theo mô hình doanh số (Jackpot hòa vốn *có phản ứng doanh số*), quy tắc trần 300 tỷ Power (phần vượt chuyển sang JP2 nếu JP1 không có người trúng), số đặc biệt Lotto, bất định lấy từ phân phối mẫu của các hệ số đã hiệu chỉnh.

### 4.8 Bao & tần suất trúng (`bao.py`, `coverage.py`, v3)
- **Bao chính xác**: với Bao v, số số trúng nằm trong tập chọn j ~ siêu bội; số lượt có đúng m số trúng = C(j,m)·C(v−j,k−m); số phụ Power (cùng lồng cầu) và số đặc biệt Lotto (lồng riêng) xử lý bằng điều kiện hóa vị trí của nó. Bao k−1 (Bao 5 Mega/Power, Bao 4 Lotto): k − j₀ lượt lên j₀+1 số trúng. Kết quả đối chiếu Monte Carlo 200.000 kỳ trong test.
- **Danh mục tối đa P(≥ 1 giải)**: bài toán phủ cực đại trên mẫu mô phỏng (SAA): tham lam (bảo đảm 1−1/e) từ 4.000 ứng viên + tìm kiếm cục bộ đổi từng số (và số đặc biệt), đánh giá lại trên mẫu độc lập với khoảng Wilson; so với danh mục ngẫu nhiên và cận B·p.

- **v3.1 — luật bao**: danh mục chính thức (`bao_catalog`, `bao_kind`), bảng giải kiểu Vietlott (`bao_prize_table`); trả thưởng của một vé bao = giải cố định + Σ_jackpot pot·n/(n + K) (n lượt của vé trúng cùng một Jackpot, K lượt của người khác); thuế 10% trên phần vượt ngưỡng của *tổng vé* (`tax_basis="ticket"`, hoặc `"play"`); `bao_ev` lấy kỳ vọng theo K ~ Poisson(N·r/C) với r là độ phổ biến trung bình các lượt (mô hình đám đông đã hiệu chỉnh) và λ của Jackpot 2 cộng thêm k − 1 tổ hợp trúng khác; `compare_bao_strategies` mô phỏng cùng một chuỗi kỳ quay cho bao, bao rút gọn (covering design "t nếu về m" từ `src/vietlott_engine/wheeling`), danh mục dàn đều và quick pick.

### 4.9 Họ Max 3D (`max3d.py`, v3.1)
- 20 số 000–999 quay độc lập theo bốn nhóm 2/4/6/8; điều kiện giải: "one" (một số của lượt có trong nhóm), "both" (cả hai số; nếu hai số giống nhau cần ≥ 2 lần xuất hiện), "ordered"/"reversed" (đúng/ngược thứ tự hai số giải đặc biệt), "pair_exact" (hai số giải Nhất bất kể thứ tự). Phân phối chính xác của một lượt bằng liệt kê trạng thái (số lần xuất hiện của mỗi số trong mỗi nhóm, chặn ở 2; nhóm đặc biệt theo thứ tự) — 6.561 trạng thái.
- Bao: sinh lượt (hoán vị chữ số, cặp có thứ tự, ký tự đại diện `*`); EV chính xác theo tuyến tính (phân phối mỗi lượt chỉ phụ thuộc hai số có trùng nhau hay không); P(trúng ≥ 1 giải) chính xác = 1 − (1 − u/1000)²⁰ với u số khác nhau; P(giải cao nhất) chính xác; phần còn lại (P(thưởng ≥ tiền vé), phân vị) bằng Monte Carlo vector hóa.

### 4.10 Lotto 5/35 chia giải (`rolldown.py`, v3)
- **EV kỳ chia giải**: EV = Σ_τ P_τ·E[net(cố định_τ + J·f_τ/(1+K_τ))], K_τ ~ Poisson(N·P_τ), E[1/(1+K)] = (1−e^{−λ})/λ, phần của hạng trống chia đều cho các hạng còn lại, xác suất Độc đắc bị trúng trước khi chia; doanh số hòa vốn bằng Brent.
- **Chuỗi chính thức (v3.2, `rolldown_history_official`)**: ngày d được công bố chia giải khi pot *sau kỳ cuối của ngày d−1* vượt 12 tỷ, không có người trúng và kỳ đó không phải kỳ vừa chia; kỳ cuối ngày d là kỳ chia nếu pot kỳ sau khởi động lại (< 0,6×), là "trúng trước" nếu Độc đắc có người trúng trong ngày d. Tốc độ tích lũy từng kỳ r_t = (J_t − J_{t−1})/R_t (J_{t−1} = 6 tỷ sau khởi động lại) tách hai pha bằng ngưỡng 0,2; phần chênh lệch (r_nhanh − r_chậm)·doanh thu pha chậm chia cho số lần khởi động lại kể từ pha nhanh trước ⇒ ≈ 5,64 tỷ/lần. Dự báo: tiền còn phải hoàn quỹ = số lần khởi động lại chưa bù × 5,64 tỷ − phần đã bù; số kỳ tới pha nhanh và tới 12 tỷ theo doanh số trung vị 28 kỳ; backtest dựng lại từ mọi kỳ thứ 10 chỉ dùng dữ liệu tới kỳ đó (hai hằng số tốc độ lấy từ toàn chuỗi — một chút nhìn trước, đã ghi rõ). Test mô phỏng thị trường hai pha cài sẵn và khôi phục đúng sự kiện, hai tốc độ, mức hoàn quỹ và độ chính xác dự báo.
- **Lịch sử (v3, khi không có chuỗi Jackpot)**: sự kiện từ đột biến doanh số; tốc độ tích lũy c từ *khoảng* thời điểm vượt 12 tỷ của từng đoạn (c ∈ (6 tỷ/R(→d−1), 6 tỷ/R(→d−2)]) cộng cận trên từ các đoạn kết thúc bằng trúng Độc đắc; c_r (ngày chia giải) bằng bình phương tối thiểu trên pot công bố; giá trị từng giải tính từ số người trúng thật. Test mô phỏng thị trường có c, c_r cài sẵn và khôi phục được cả sự kiện lẫn tham số.

### 4.11 Keno, Bingo18 (`fastgames.py`, v3.2)
- **Keno**: P(trùng m trong b số chọn) = C(20,m)·C(60,b−m)/C(80,b); cửa phụ dùng số lượng số Lớn (hoặc Chẵn) trong 20 số rút ~ siêu bội(80, 40, 20). RTP = Σ P·giải / 10.000. Mỗi ô bảng giải mà các nguồn in khác nhau được giữ lại cùng RTP của từng phiên bản (`KENO_TABLE_VARIANTS`, `SideBet.alternatives`).
- **Bingo18**: liệt kê 216 kết quả có thứ tự đồng khả năng — xác suất và RTP chính xác của mọi cửa.

### 4.12 Kiểm định Keno, Bingo18, Max 3D (`src/vietlott_engine/analytics/products.py`, v3.2)
- Mỗi sản phẩm so với luật quay chính xác của nó: Keno — χ² tần suất 80 số nhân hệ số không hoàn lại (n−1)/(n−k) = 79/60, G-test số Lớn / số Chẵn mỗi kỳ theo siêu bội (gộp đuôi tới kỳ vọng ≥ 5), số trùng với kỳ liền trước ~ siêu bội(80, 20, 20), tương quan liền kỳ; Bingo18 — mặt 1–6 gộp và theo vị trí, tổng 3–18, bộ ba ~ nhị thức(6/216), bảng liên hợp giữa vị trí và giữa hai kỳ liền nhau; Max 3D — chữ số từng hàng, 1.000 số, số trùng trong kỳ ~ Poisson(C(20,2)/1000 mỗi kỳ), lặp từ kỳ trước. Chỉ dùng cặp kỳ có mã liên tiếp; p-value hiệu chỉnh Benjamini–Hochberg.
- **Kiểm tra chéo chữ số** (`digit_replication`): chọn ô (hàng, chữ số) lệch mạnh nhất trên mẫu *phát hiện*, rồi kiểm định nhị thức một phía **chỉ ô đó** trên mẫu *xác nhận* độc lập (Max 3D ↔ Max 3D Pro, hoặc nửa đầu ↔ nửa sau) — p-value xác nhận không bị thiên lệch chọn lọc. Hệ số xác suất của từng số = tích tỷ lệ ba chữ số × 1000 (giả định các hàng độc lập, mang tính mô tả).
- Test: kích thước (60 lịch sử Bingo18 công bằng: tỷ lệ bác bỏ ≤ ~13%; ba sản phẩm công bằng không bác bỏ) và power (chữ số 6 cài lệch 25% được phát hiện, q < 10⁻³, và xác nhận chéo p < 10⁻⁴; ô giả không xác nhận).

### 4.13 Nguồn dự phòng và gộp dữ liệu (`crawler/sources/nhanaz.py`, `vqe130.py`, `fallback.py`, v3.3)
- **Kho cộng đồng**: CSV `datasets/draws/<sp>/{all,YYYY-MM}.csv`, `prizes/<sp>/all.csv`, `exclusions.csv`, `prize_rules.csv` → bản ghi canonical (Mega/Power/Lotto, kèm bảng giải) hoặc dòng sản phẩm; hai bố cục bảng giải Lotto (cột tường minh, và cột lệch do tiêu đề trang cũ: `column_3` = số người trúng, `column_4` = giá trị) đều đọc được; kỳ `not_confirmed` tách ra thành mục loại trừ. Đồng bộ tăng dần theo tháng cho Keno/Bingo18 (từ tháng của kỳ cuối đã có).
- **Snapshot 1.3.0**: mỗi Parquet `draw_number, draw_date, status, payload` được đổi sang đúng dạng dòng của kho rồi qua cùng bộ chuyển đổi, nên hai nguồn được kiểm như nhau.
- **Chuỗi dự phòng**: `FallbackDrawSource` / `--source auto` thử từng nguồn, bắt lỗi nguồn (proxy chặn, 403 do IP, 404, thiếu tệp) và ghi `attempts`; nguồn nào trả lời trước được dùng. Bảng giải đi kèm kết quả từ kho được ghi luôn vào kho dữ liệu.
- **Gộp** (`merge_layers`): hợp theo mã kỳ, lớp trên thắng, đếm phần trùng / giống hệt / mâu thuẫn và nguồn gốc từng dòng (`src`); bản ghi chính thức Mega/Power/Lotto chỉ được bổ sung kỳ còn thiếu. **Loại trừ**: `exclusions.json` (thông báo của Vietlott) áp dụng khi đọc, `--include-unconfirmed` để kiểm toán.
- **Kiểm định ô đặt trước** (`digit_cell_test`): một kiểm định nhị thức một phía cho ô (hàng, chữ số) chọn từ dữ liệu khác — dùng cho Max 4D.

### 4.14 Trang lưu từ vietlott.vn (`sources/saved_pages.py`, `scripts/verification_checklist.py`, v3.4)
- Đọc HTML (dòng "saved from url", canonical, og:url), MHTML (giải mã từng phần theo charset khai báo, rồi theo thẻ meta, rồi UTF-8) và HAR (phản hồi AjaxPro `value.HtmlContent` và trang chi tiết). Phân loại theo đường dẫn, tên phần AjaxPro hoặc `GameId` (5 = Max 3D, 6 = Keno, 7 = Max 3D Pro, 8 = Bingo18).
- Kiểm hợp lệ như mọi nguồn: Keno 20 số khác nhau trong 1–80, Bingo18 3 số trong 1–6, Max 3D/4D đúng số chữ số; Mega/Power/Lotto qua cùng bộ nhập bản ghi chính thức.
- Đối chiếu: tỷ lệ sai k/n với cận trên Clopper–Pearson một phía 95% (k = 0 ⇒ 1 − 0,05^(1/n)). Mẫu kiểm được chọn ngẫu nhiên với hạt giống cố định, nên không phụ thuộc vào việc người kiểm chọn trang nào.

### 4.15 Dự báo tự học (`src/vietlott_engine/forecast`, v3.5)
- Mỗi sản phẩm là một hoặc hai *thành phần*: chọn số (tập k trong n, kèm số phụ cùng lồng của Power) hoặc chữ số (m số × L vị trí × A ký hiệu; số đặc biệt Lotto là A = 12, L = 1). Mỗi nhóm chuyên gia xử lý một khối kỳ, với trạng thái mang sang khối sau (tổng tích lũy, EWMA bằng `lfilter` có điều kiện đầu, vị trí lần cuối xuất hiện, tham số logistic và bộ tích gradient của mini-batch dở dang), nên chia khối hay lưu/đọc lại không làm đổi kết quả.
- Hồi quy logistic (chỉ ở trò chọn số): mini-batch căn theo số kỳ toàn cục (20 kỳ; Keno 500 kỳ), AdaGrad trên mất mát Bernoulli từng số, L2 = 10⁻⁴; dự báo cho một kỳ chỉ dùng tham số học từ các mini-batch trước nó.
- Dự báo kỳ tới: bộ đề xuất và sổ dự báo dùng k số có xác suất có mặt chính xác cao nhất; chấm lùi dùng trọng số trung bình (rẻ hơn, gần đúng). Xác suất có mặt từng số π_i = w_i e_{k−1}(w_{−i}) / e_k(w) của từng chuyên gia, rồi trung bình theo trọng số hỗn hợp; phân phối số trùng của một vé = e_r(w_vé) e_{k−r}(w_ngoài) / e_k(w); RTP Keno từng bậc theo bảng giải chính thức; Bingo18 liệt kê 216 kết quả cho mọi cửa; Max 3D: RTP mô hình ≈ RTP × hệ số của số đề xuất.
- Test: mọi chuyên gia là phân phối hợp lệ (vét cạn mọi kết quả, cả số phụ cùng lồng), xác suất có mặt và phân phối số trùng khớp vét cạn, không nhìn trước (đổi kết quả tương lai không đổi điểm quá khứ), học từng khối + lưu/đọc lại = học một lần, kích thước e-value dưới máy công bằng (60 lịch sử chọn số + 30 lịch sử chữ số), power trên lệch cài sẵn (đuôi 6 ở 13%, số 7 có odds ×1,8), kết quả trên dữ liệu thật (Max 3D có tín hiệu, Mega không), sổ dự báo ghi một lần rồi chấm, cửa sổ `--last` không bao giờ được tính là bằng chứng, bộ đề xuất = k số có xác suất có mặt cao nhất, cảnh báo dữ liệu cũ theo lịch quay, CLI và API.

Mọi thủ tục mới đều có test **kích thước** (tỷ lệ bác bỏ sai dưới H₀ ≤ α trên dữ liệu mô phỏng) và **power** (phát hiện được hiệu ứng cài sẵn).

---

## 5. Giới hạn & rủi ro đã biết

- **vietlott.vn**: chỉ phục vụ IP Việt Nam (mục 1.6.1), và có bước xác minh Cloudflare mà engine không vượt (mục 1.8). Parser trang lưu chưa được thử trên trang thật lưu sau Cloudflare. Endpoint AjaxPro, `Key`, `GameId` và cấu trúc HTML do website quyết định và có thể đổi bất cứ lúc nào. Parser đã được test bằng HTML mẫu và transport giả lập, nhưng **chưa chạy trực tiếp với website** (môi trường phát triển không tới được vietlott.vn). Nếu hỏng, cập nhật `ENDPOINTS` / parser trong `src/vietlott_engine/crawler/sources/vietlott_official.py`; nguồn mặc định là bản sao GitHub. Bản sao Keno/Bingo18 có khoảng trống mã kỳ; chạy `products sync --source vietlott --full` từ Việt Nam để lấp.
- **Dự báo tự học (v3.5)**: "xác suất kỳ tới" của mô hình chỉ đáng tin khi e-value đã vượt 20; khi chưa, chúng là mô hình tự đánh giá. Ngay cả khi có bằng chứng (Max 3D, Max 3D Pro), lợi thế nhỏ (×1,07–1,08 cho đuôi 6) và RTP vẫn < 1; ở Max 3D Pro, e-value chỉ ở trên ngưỡng 47% thời gian sau lần vượt đầu. `forecast fit --last N` không cho bằng chứng hợp lệ. Mô hình chỉ học các kỳ có mã lớn hơn kỳ cuối đã học; khi dữ liệu cũ được bổ sung (lấp khoảng trống), chạy `forecast fit` để học lại. Chấm lùi dùng quy tắc chọn số theo trọng số trung bình (gần với xác suất có mặt chính xác, không trùng tuyệt đối).
- **Bảng giải Keno**: lấy theo trang chi tiết chính thức (v3.3); còn một ô chưa thống nhất (bậc 5 trùng 4 số: 150.000 đ theo trang sản phẩm, "0 đ" trên trang chi tiết 2026).
- **Nguồn dự phòng**: kho cộng đồng và snapshot 1.3.0 là bản thu của bên thứ ba (có `data_source` từng dòng, khớp 100% với bản chính thức ở mọi phần trùng đã kiểm), không phải xác nhận của Vietlott. Keno 2019–2022 chỉ có từ kho xoso.com.vn; Lotto #920 từ trang phụ. Khoảng trống mã kỳ Keno/Bingo18 có thể là kỳ chưa phát hành hoặc chưa thu — không tự điền.
- **Giá trị Jackpot lịch sử** không có trong mirror ⇒ backtest tính Jackpot ở mức tối thiểu, không chia giải. Vì cả 9 chiến thuật đều không trúng Jackpot trong mẫu, điều này không ảnh hưởng kết luận.
- **Mô hình độ phổ biến** là prior cho tới khi được hiệu chỉnh (4.4). Hãy xem RTP trong 1.3 như phân tích độ nhạy, không phải dự báo; bảng bất định cho thấy số vé bán ra là yếu tố quyết định.
- **Bayes phân cấp** dùng composite likelihood (bỏ qua tương quan âm yếu giữa các số); cận trên đã được kiểm tra hiệu chuẩn trên lịch sử mô phỏng. **TOST từng số** thì chặt chẽ nhưng bảo thủ.
- **SPA** hơi vượt kích thước danh định khi hiệu suất có tự tương quan (≈7% ở mức 5% trong mô phỏng AR(0.3)) — đã biết trong tài liệu; kết luận trên dữ liệu thật (p ≈ 0.4) không nằm gần ranh giới.
- Cơ cấu giải và luật thuế là tham số trong `src/vietlott_engine/core/games.py` — kiểm tra lại khi Vietlott hoặc luật thay đổi.
- **Dữ liệu giải thưởng**: từ v3.2 là bảng giải trên trang chi tiết vietlott.vn do bên thứ ba thu (có SHA-256 trang, PDF biên bản để kiểm), không phải API chính thức; số vé bán từng kỳ vẫn là ước lượng (Vietlott không công bố theo kỳ). Mô hình đám đông Mega không qua kiểm định likelihood ngoài mẫu (mục 1.4.3). *Ghi chú v3 (đã thay thế):* dữ liệu v3 đến từ các repo cộng đồng (Compal123/vietlot-ai, leoodz/vn-vietlott); đã đối soát với bộ số trúng và với số liệu báo chí, nhưng hãy kiểm tra giấy phép trước khi phân phối lại. Mega chỉ có số người trúng từ 09/2025 (161 kỳ) và không có chuỗi Jackpot ⇒ s và mô hình doanh số Mega dựa trên đường Jackpot dựng lại (khép theo 1 giá trị công bố; mốc "vượt 96 tỷ" ở kỳ #01565 dựng lại thành 110,9 tỷ — cùng chiều nhưng cho thấy sai số vài chục phần trăm giữa đoạn). s Mega (0,497) cao hơn Power (0,408): có thể do khác cơ cấu quỹ hoặc do sai số N̂.
- **Hệ số mẫu hình tổ hợp** (dãy số, đường thẳng trên phiếu) không nhận diện được từ số người trúng ⇒ RTP của vé mẫu hình (×27–36) vẫn dựa trên prior.
- **Luật bao (v3.1)**: trang vietlott.vn chặn truy cập tự động trong môi trường phát triển, nên danh mục và bảng giải bao được đối chiếu qua nhiều nguồn thứ cấp (đại lý được cấp phép, báo chí) và kiểm bằng toán học. Bao Max 3D / Max 3D+ (đảo số, bao vị trí) chỉ có ở hướng dẫn của đại lý; cách tính thuế cho vé bao (theo tổng vé) là cách hiểu luật, cấu hình được. Quy tắc nhân đôi giải cho hai số giống nhau mới thấy ở Max 3D+. Giới hạn 30 tỷ/kỳ của giải Đặc biệt Max 3D Pro không đưa vào EV.
- **Lotto chia giải**: từ v3.2 pot lấy từ chuỗi Jackpot chính thức; cơ chế hai pha là suy luận từ số liệu, không phải quy định công bố; RTP kỳ chia giải dựa trên doanh số ước lượng. *Ghi chú v3 (đã thay thế):* khi chưa có chuỗi Jackpot Lotto thì pot được dựng lại; c_r chỉ dựa trên 3 giá trị công bố, nên báo cáo cả RTP thận trọng. Chỉ 4/21 kỳ chia giải có người trúng Độc đắc, ít hơn 7,9 kỳ mà giả định vé chọn đều dự báo (p ≈ 0,1, chưa có ý nghĩa) — nếu xác nhận, nghĩa là vé trong kỳ chia giải trùng nhau nhiều (bao, nhóm chơi) và pot được chia thường xuyên hơn mô hình EV giả định.

## 6. Kiểm thử

Đợt rà soát VLM: `pip install -e ".[dev,crawler]"` rồi `make test` đạt **262 passed, 1 skipped**; bằng chứng tại [reports/vlm/verification.json](reports/vlm/verification.json). Khi không cài extra `crawler`, test transport curl_cffi được skip; CI cài extra để kiểm tra test này. Các kiểm thử v4.0 và trước đó gồm: v4.0 thêm (`tests/test_packaging.py`, 5 test): tìm thư mục dự án từ thư mục con / nơi khác / `VQE_HOME`, `--version` và trợ giúp có `init` / `doctor`, bộ cài đúng định dạng (`bash -n`, PowerShell có BOM UTF-8 + CRLF, `install.cmd` chỉ ASCII), workflow gọi đúng lệnh, dựng trang tĩnh + JSON; `tests/test_forecast.py` thêm: sổ dự báo chỉ nhận dự báo trước kỳ quay theo lịch (Mega/Power/Max 3D 18:00, Lotto 13:00 và 21:00, Keno/Bingo18 khoảng đêm và ngày đủ kỳ, Max 4D không có kỳ tới), `doctor` qua CLI. v3.5 thêm (`tests/test_forecast.py`, 12 test): xem mục 4.15. v3.4 thêm (`tests/test_saved_pages.py`): nhận ra thử thách Cloudflare (header `cf-mitigated`, hoặc `server: cloudflare` + dấu hiệu trong trang), không retry và báo đúng hướng dẫn, 503 thường vẫn retry; đọc trang lưu đủ định dạng (MHTML quoted-printable, HTML có "saved from url", HAR có AjaxPro, chính trang thử thách, trang lạ); phân loại đường dẫn (Max 3D Pro trước Max 3D, `GameId`); ô giải Keno theo bậc và so với bảng của engine; đối chiếu với dữ liệu (giống hệt / lệch / mới, cận trên tỷ lệ sai); CLI chạy thử không ghi gì, nhập không ghi đè kỳ lệch trừ khi `--replace-mismatches`; danh sách kiểm tái lập được. v3.3 thêm (`tests/test_fallback_sources.py`): đọc kho cộng đồng từ thư mục (đủ định dạng: Keno theo tháng, Mega có bảng giải, Lotto cột lệch, Max 4D, loại trừ, luật Keno) và từ GitHub giả lập (đúng tháng, bỏ qua tháng 404), hai bố cục ô giải, chuỗi dự phòng (vietlott bị proxy chặn → kho; mọi nguồn lỗi → báo đủ), đồng bộ `auto` ghi `attempts`/`source_used`/loại trừ, loại trừ có sẵn được áp dụng, nhập snapshot Parquet 1.3.0 (ghi Parquet thật bằng DuckDB CLI), báo lỗi khi thiếu snapshot, `merge_layers` (ưu tiên, mâu thuẫn, nguồn gốc), kiểm định chữ số Max 4D (kích thước, power, ô đặt trước), dữ liệu Max 4D đi kèm, bảng Keno khớp trang chính thức, CLI đồng bộ từ thư mục kho, API Max 4D. v3.2 thêm (`tests/test_all_products.py`): parser Keno / Bingo18 / Max 3D / bảng giải / trang chi tiết bằng HTML mẫu, ba định dạng bản ghi Max 3D và từ chối dữ liệu sai luật, nhập bản ghi chính thức (pot = giá trị × số vé trúng), dữ liệu đi kèm nhất quán, phân phối siêu bội Keno + RTP mọi bậc trong 0,49–0,56 + mô phỏng 60.000 kỳ, xác suất Bingo18 chính xác, kích thước và power của bộ kiểm định sản phẩm, kiểm tra chéo chữ số (phát hiện lệch cài sẵn, không xác nhận ô giả), thị trường Lotto hai pha mô phỏng (khôi phục sự kiện, hai tốc độ, 5,64 tỷ/lần, dự báo sai số ≤ 3 kỳ), không công bố chia giải ngày sau kỳ vừa chia, proxy 403 không retry và báo đúng nguyên nhân, 403 của website báo chặn IP, cookie khởi tạo + phân trang + `since_id` của crawler sản phẩm, trang chi tiết → bản ghi chính thức, kho sản phẩm + đồng bộ (gộp seed, chèn tăng dần, nguồn sai báo lỗi), đồng bộ bảng giải vào kho, Jackpot hòa vốn tìm điểm cắt đầu tiên khi doanh số lồi, endpoint `/products` và CLI `products`. v3.1 thêm (`tests/test_bao.py`): danh mục bao khớp bảng giá Vietlott (Mega/Power/Lotto), từ chối/gắn cờ kiểu bao không bán, bảng giải Power Bao 5 / Bao 7 và Mega Bao 5 / Bao 7 khớp ví dụ công bố, Bao 4 Lotto hoàn vốn khi trùng số đặc biệt, nhiều lượt cùng trúng một Jackpot chỉ chia một pot, thuế theo vé vs theo lượt, EV có người trúng chung thu về kết quả chính xác khi không có ai khác, mô phỏng khớp phân phối chính xác, bao rút gọn "3 nếu về 3" có đúng P(trúng) của bao gốc; Max 3D Pro RTP ≈ 55% và tỷ lệ có giải ≈ 4%, Max 3D+ giải Nhất 1/500.000, số lượt bao bộ số (36/18/6) và bao nhiều bộ số (6…380), chính xác vs mô phỏng; endpoint bao và Max 3D. v3 thêm: xác suất Lotto 5/35 bằng vét cạn 324.632 tổ hợp, EV Lotto bắt buộc số đặc biệt, quy tắc trần 300 tỷ, đối soát nguồn giải thưởng (sửa lệch một kỳ), khôi phục s và N_t từ thị trường mô phỏng, **khôi phục đám đông cài sẵn** (β, q, N_t) từ số người trúng mô phỏng bằng vé thật, bao chính xác vs Monte Carlo (5 cấu hình, cả số phụ và số đặc biệt), tuyến tính kỳ vọng của bao, danh mục coverage (vượt ngẫu nhiên, đúng cận, 12 số đặc biệt ⇒ 100%), EV chia giải đơn điệu + điểm hòa vốn, phân phối lại hạng trống, khôi phục sự kiện chia giải và c, c_r từ thị trường Lotto mô phỏng, toàn bộ endpoint mới. v2 thêm: kích thước và power của Westfall–Young, power giải tích vs mô phỏng, biên TOST, phân phối số trúng chính xác vs vét cạn, Bayes phân cấp (công bằng vs lệch), chuẩn hóa phân phối cá cược và bất đẳng thức Ville cho e-process (kể cả vé cố định tương quan tối đa), phát hiện giai đoạn lệch, kích thước DM dưới tự tương quan, kích thước/power của SPA, Kelly (điều kiện đạo hàm tại tối ưu), ILP chứng minh C(9,6,4) = 12. Bộ gốc gồm: hiệu chuẩn p-value dưới H₀ (300 lịch sử mô phỏng), phát hiện máy quay lệch và chuỗi có trí nhớ (power), đối chiếu công thức với vét cạn (e_k, subset-sum, bộ lấy mẫu, marginal likelihood), gradient GCN và Poisson-MLE bằng sai phân hữu hạn, khôi phục tham số hiệu chỉnh, covering number tối ưu đã biết, chống look-ahead (chiến thuật “gián điệp” thử đọc `.base` và ghi dữ liệu), retry/backoff/`Retry-After`/rate limit với transport giả lập, round-trip DuckDB → Parquet, và toàn bộ endpoint API.

## 7. Nguồn dữ liệu & giấy phép

Từ v3.3, dữ liệu dự phòng và phần lấp khoảng trống (Keno 2019–2022, Bingo18, Lotto #920, Max 4D, danh sách kỳ không xác nhận, luật giải Keno) lấy từ [NhanAZ-Data/vietlott-research](https://github.com/NhanAZ-Data/vietlott-research) (MIT, xem `data/seed/LICENSE-NhanAZ-vietlott-research`; bản `b88ecd7`, 02/10/2026), đối chiếu với snapshot của Vietlott Quant Engine 1.3.0 (do người dùng cung cấp, MIT). Từ v3.2, kết quả và bảng giải Mega/Power/Lotto, Max 3D và Max 3D Pro trong `data/seed/` lấy từ bản ghi trang chi tiết vietlott.vn của [pqminh-4/vietlott-data](https://github.com/pqminh-4/vietlott-data) (MIT, xem `data/seed/LICENSE-pqminh-4-vietlott-data`; dựng lại bằng `scripts/build_official_dataset.py`, báo cáo `data/seed/OFFICIAL_DATA_REPORT.json`); Keno và Bingo18 từ [vietvudanh/vietlott-data](https://github.com/vietvudanh/vietlott-data) (MIT, xem `data/seed/LICENSE-vietlott-data`); Max 3D / Max 3D Pro đối chiếu thêm với vietvudanh và [googlesky/vietlott-data](https://github.com/googlesky/vietlott-data) (khớp mọi kỳ chung); định dạng request AjaxPro tham khảo crawler của các dự án này. Bảng giải Keno: [VTC Pay (26/02/2020)](https://vtcpay.vn/tin-tuc-473/tin-tu-vtc-pay-28/co-cau-giai-thuong-keno-vietlott-66274), [xosovip (29/08/2025)](https://xosovip.vn/tin-tuc/co-cau-giai-thuong-keno), [onbit](https://onbit.vn/bai-viet/xem/xo-so-keno-vietlott-huong-dan-cach-choi-cung-co-cau-giai-thuong), [xoso.mobi](https://xoso.mobi/tin-tuc/lich-quay-co-cau-giai-thuong-va-the-le-tham-du-xo-so-tu-chon-keno-vietlott-n7431.html), [vesotanphat](https://vesotanphat.com/huong-dan/ket-qua-xo-so-keno-co-cau-giai-thuong-lich-quay-thuong-va-the-le-tham-gia-xs-keno.html); thay đổi cửa phụ Keno 2021: VnExpress, VietnamNet. Bingo18: xskt, onbit. Dữ liệu v3: Số người trúng từng giải: [Compal123/vietlot-ai](https://github.com/Compal123/vietlot-ai); số người trúng và giá trị Jackpot Power 6/55: [leoodz/vn-vietlott](https://github.com/leoodz/vn-vietlott) (dựng lại bằng `scripts/build_prize_dataset.py`, báo cáo đối soát `data/seed/PRIZE_DATA_REPORT.json`). Mốc Jackpot công bố (`data/seed/jackpot_anchors.json`) kèm trích dẫn báo chí (Thanh Niên, Tuổi Trẻ, CafeF, Vietstock).

Luật chơi bao (v3.1): [onbit — Tổng hợp cách chơi bao Mega/Power](https://onbit.vn/bai-viet/xem/tim-hieu-ve-tinh-nang-choi-bao-xo-so-vietlott), [VTC Pay — chơi bao Power 6/55](https://vtcpay.vn/blog/cach-choi-bao-vietlott-6-55-power-chi-tiet-cho-nguoi-moi.html), [dudoanvietlott — giá các mức bao Mega](https://www.dudoanvietlott.net/kinhnghiem/17/Tong-hop-cach-choi-bao-57891011121314151718-Mega-6-45-Vietlott.html), [onbit — mua vé Mega](https://onbit.vn/mua-ve-so/vietlott-mega), [Kenh14](https://kenh14.vn/bi-kip-cho-nguoi-moi-choi-xo-so-quay-nhanh-lotto-5-35-21526070621063036.chn), [VietnamNet](https://vietnamnet.vn/xo-so-moi-cua-vietlott-gay-chu-y-2414106.html) và [xskt](https://xskt.com.vn/tin-tuc/xs-dien-toan/vietlott-gioi-thieu-san-pham-va-cach-choi-xo-so-loto-5-35) (bao Lotto 5/35), [VnExpress](https://vnexpress.net/nguoi-choi-thu-nghiem-xo-so-tu-chon-max-3d-pro-4353946.html), [Thanh Niên](https://thanhnien.vn/xo-so-max-3d-pro-la-gi-ma-co-the-trung-len-toi-hon-36-ti-dong-185260130081610022.htm), [24h](https://www.24h.com.vn/thi-truong-tieu-dung/xo-so-max-3d-pro-la-gi-ma-co-the-trung-len-toi-hon-36-ty-dong-c52a1733404.html), [onbit](https://onbit.vn/bai-viet/xem/max-3d-pro-vietlottcach-choi-va-co-cau-giai-thuong) và [xsmn.mobi](https://xsmn.mobi/co-cau-giai-thuong-max-3d-pro-moi-nhat.html) (Max 3D Pro), [Tuổi Trẻ/PLO](https://tuoitre.vn/plo/co-hoi-trung-tien-ti-voi-xo-so-max-3d-cua-vietlott-post577521.html), [vesoatrungroi](https://vesoatrungroi.com/huong-dan/huong-dan-choi-max-3d+-5), [Thời báo Tài chính](https://thoibaotaichinhvietnam.vn/suc-hap-dan-tu-giai-thuong-tien-ti-cua-xo-so-max-3d-17430.html) và [onbit](https://onbit.vn/bai-viet/xem/cach-choico-cau-giai-thuong-max3d-va-max3dvietlott) (Max 3D / 3D+). Bảng Max 3D+ của xsmn.mobi mâu thuẫn với các nguồn còn lại (và cho RTP > 100%) nên không dùng. Cơ cấu giải thưởng và luật chia giải Lotto 5/35 theo công bố của Vietlott; quy định thuế theo Luật Thuế TNCN 2025.


### Trang chính: kết quả, dự báo, đối chiếu

GitHub Pages có hai trang:

- `index.html`: kết quả của 7 sản phẩm đang hoạt động, toàn bộ nhóm giải Max 3D / Max 3D+ / Pro, cơ cấu giải matrix / Keno / Bingo18, bộ số dự báo kỳ kế tiếp và lịch sử tự đối chiếu. Chọn sản phẩm, chọn kỳ, mở cơ cấu giải, đổi sáng/tối; giao diện cho desktop và mobile.
- `forecast.html`: trang phân tích/tính toán trước đây, giữ e-value, xác suất, lịch sử học và sổ dự báo. Max 4D là dữ liệu lưu trữ ở đây.

Dựng bằng `python scripts/build_site.py --out site` sau khi đồng bộ; builder không gọi mạng hoặc học lại mô hình. `site/data/dashboard.json` là snapshot mới; các JSON `summary`, `scoreboard`, `forecast/*`, `ledger.jsonl` vẫn giữ đường dẫn cũ.

`results.yml` dựng và xuất bản Pages cùng luồng cập nhật kết quả (đến 10 phút/lần trong giờ mở, mỗi giờ ban đêm). `update.yml` đồng bộ tổng thể hai lần/ngày. Trang đang mở kiểm tra JSON mới mỗi 60 giây, giữ bộ lọc/kỳ đã chọn và có nút Làm mới. Cron GitHub có thể trễ; thời gian “Cập nhật” cho biết tuổi snapshot, không cam kết realtime.

Đối chiếu dùng phân phối/bộ số đầu tiên đã lưu trước kỳ quay, đúng sản phẩm + mã kỳ + ngày. Không lấy checkpoint hiện tại để tạo dự báo quá khứ. Dự báo tham khảo Keno/Bingo18 chưa đủ thời điểm để chứng nhận live; không đưa vào thành tích. Jackpot là quỹ của đúng kỳ, số người trúng khuyết hiển thị “—”; không lấy số tiền của kỳ trước. Xem thiết kế và hợp đồng dữ liệu tại [docs/VLM_RESULTS_HOME.md](docs/VLM_RESULTS_HOME.md).
