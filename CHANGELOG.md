# Nhật ký thay đổi

## 4.0.0 — repo riêng, bộ cài hoàn chỉnh

Tách khỏi VLA thành repo độc lập.

- **Gói Python `vietlott_engine`** (bố cục `src/`, `pip install -e .`, lệnh `vietlott`). Thay cho package `src` cũ (trùng tên với VLA). Mã gọi `from src.… import …` cần đổi thành `from vietlott_engine.… import …`; `python -m src.cli …` thành `vietlott …`.
- **Bộ cài:** `install.cmd` / `install.ps1` cho Windows (PowerShell 5.1 trở lên, tự tìm Python qua `py` và thư mục cài mặc định, tùy chọn cài Python bằng winget, cài gói ở chế độ thường để an toàn với tên thư mục có dấu) và `install.sh` cho Linux / macOS (cài editable). Bộ cài tạo `.venv` (tạo lại nếu hỏng), cài gói, tạo `.env`, nạp dữ liệu, học bộ dự báo, tạo lệnh tắt `vietlott(.cmd)`, ghi thư mục dự án vào `.venv/vietlott_home.txt` và kiểm tra. Chạy lại an toàn.
- **`vietlott init`, `vietlott doctor`, `vietlott --version`.** Lệnh tự chuyển về thư mục dự án (`VQE_HOME` để chỉ định), tự đưa đầu ra về UTF-8 trên Windows và báo lỗi một dòng dễ đọc (`VQE_DEBUG=1` để xem chi tiết).
- **Sổ dự báo theo lịch quay chính thức.** Chỉ ghi khi kỳ được dự báo chưa quay. Keno/Bingo18 chỉ ghi trong khoảng 22:15–05:55 và khi ngày dữ liệu đã đủ kỳ. `--force-record` ghi nhưng không tính vào thành tích. API `?record=true` theo cùng quy tắc. Max 4D (đã ngừng) không có kỳ tới.
- **GitHub Actions:** CI (test trên Linux và Windows), `installer.yml` (chạy bộ cài trên Linux / macOS / Windows khi bộ cài đổi và mỗi tuần); `update.yml` chạy 2 lần/ngày: đồng bộ, học kỳ mới, ghi sổ dự báo vào repo (trước bước dựng trang, để trang lỗi cũng không mất sổ), rồi đăng trang GitHub Pages kèm JSON cho VLA / trang khác; `release.yml` phát hành bằng tag; Dependabot.
- **Hai sổ dự báo tách nhau:** máy cài bằng bộ cài ghi vào `data/local/forecast/` (đặt trong `.env`, không theo dõi bằng git); GitHub Actions ghi sổ công khai `data/forecast/ledger.jsonl` và commit vào repo. Chạy trên máy không làm đổi tệp nào của repo, nên `git pull` / GitHub Desktop không xung đột với các commit tự động.
- **`scripts/build_site.py`:** trang tĩnh tiếng Việt (sáng / tối, điện thoại) và `data/*.json`.
- Docker: Python 3.12, cài gói thay vì chép mã; các script báo cáo đọc `VQE_PRODUCT_SEED_DIR`.

## 3.5.0 — dự báo tự học có kiểm chứng

Hỗn hợp Bayes fixed-share của các chuyên gia cho cả 8 sản phẩm, có e-value hợp lệ ở mọi thời điểm và sổ dự báo. Trên dữ liệu thật, chỉ thấy tín hiệu ở chữ số 6 hàng đơn vị của Max 3D / Max 3D Pro. README mục 1.9.

## 3.4.0 — vietlott.vn có Cloudflare

Nhận ra thử thách Cloudflare và dừng, không vượt. Nhập trang hoặc HAR do người dùng tự lưu, có đối chiếu (`products import-pages`). Danh sách trang nên kiểm. README mục 1.8.

## 3.3.0 — nguồn dự phòng

Chuỗi `--source auto` (vietlott.vn → kho cộng đồng → bản sao → snapshot 1.3.0); Keno 297 nghìn kỳ; Max 4D; danh sách kỳ không xác nhận. README mục 1.7.

## 3.0 – 3.2

Dữ liệu giải thưởng chính thức, hiệu chỉnh hành vi người chơi, chơi bao, toàn bộ sản phẩm Vietlott, cơ chế hai pha của Jackpot Lotto 5/35. README mục 1.4–1.6.

## 1.x – 2.x

Kiểm định ngẫu nhiên, Bayes/Markov/GCN, EV chống trùng số, bao lô, backtest walk-forward, suy luận thống kê.
