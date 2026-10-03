# VLM: crawler, kiểm tra kỳ thiếu và đối soát tiền thưởng

Package `vlm` bổ sung giao diện vận hành trên `vietlott_engine`, trong layout `src/` hiện có. Xem [Audit Matrix](../reports/VLM_AUDIT_2026-10-03.md) trước khi sử dụng snapshot cho phân tích.

## Cài đặt

Python 3.11 trở lên:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,crawler]"
```

PostgreSQL cần thêm `python -m pip install -e ".[postgres]"`. Crawler HTTPX hiện có vẫn là mặc định của CLI `vietlott`; chọn transport mới bằng `VQE_HTTP_BACKEND=curl_cffi`. CLI `vlm-audit` sử dụng transport mới trực tiếp.

```bash
export VQE_HTTP_BACKEND=curl_cffi
export VLM_PROXIES='http://proxy-a.example:8080,http://proxy-b.example:8080'
export VLM_FLARESOLVERR_URL='http://127.0.0.1:8191/v1'
```

Proxy và FlareSolverr là tùy chọn, do người vận hành cung cấp. Không cấu hình thì kết nối trực tiếp, rồi chuyển nguồn khi thất bại. Không tự mua hoặc tìm proxy công cộng. Các ví dụ trên là địa chỉ minh họa.

## Transport và nguồn

`ScraperConfig` cho phép đặt timeout, ngân sách mỗi yêu cầu, tốc độ, concurrency, TTL clearance và kích thước tối đa. Mặc định: 20 giây/lần, ngân sách 120 giây, 1 yêu cầu/giây, 4 request đồng thời, tối đa 32 MiB/response. Ngân sách có thể kết thúc sớm chuỗi retry.

- `curl_cffi` chọn Chrome/Firefox và HTTP/2; UA tương ứng với profile TLS, thay đổi cùng profile.
- 403/429/503 và lỗi tạm thời được retry theo 1/2/4/8/16 giây. `Retry-After` được tôn trọng; không thử sớm khi thời gian chờ vượt ngân sách.
- HTML challenge HTTP 200 cũng bị từ chối. 404 và lỗi policy của proxy mạng không được retry vô ích.
- FlareSolverr trả HTML/cookie/UA. Clearance được giữ cùng origin, proxy, profile và UA đến TTL; POST khởi tạo session bằng GET rồi gửi lại request gốc. Dịch vụ có thể không giải được một challenge; lỗi vẫn chuyển sang fallback. Không có chứng nhận vượt mọi Turnstile.
- Giải nén gzip/br chỉ thực hiện một lần khi chuyển response libcurl sang HTTPX.
- Log vận hành chỉ lưu tên loại lỗi; không ghi URL proxy chứa thông tin xác thực hoặc cookie.

Thứ tự mặc định: Vietlott detail/AjaxPro → NhanAZ theo game/tháng → kho canonical `pqminh-4` → mirror `vietvudanh`. Các branch `main` của mirror là nguồn thay đổi theo thời gian; dùng `base_url` trỏ SHA commit để tái lập một lần lấy cụ thể. Cache giới hạn theo URL, kết quả rỗng hoặc sai game/id chuyển tiếp. Max 3D+ dùng cùng kết quả Max 3D.

`TargetedFetcher` giới hạn toàn bộ một adapter ở 40 giây mặc định, bao gồm detail + Ajax; timeout được chuyển sang nguồn kế tiếp. Bốn nguồn vì vậy còn nằm trong ngân sách repair 180 giây cho một vòng ngày gợi ý. Có thể đặt `source_timeout_s` trong Python khi thay đổi số nguồn hoặc ngân sách vận hành.

## Kiểm tra lịch sử

```bash
vlm-audit --game all --as-of 2026-10-03T12:32:56+07:00
vlm-audit --game all --probe-latest
```

Lệnh đọc seed và CSDL repair, nối `prizes_<game>.jsonl` theo game/id/ngày, rồi viết `reports/vlm/missing_draws.json` nguyên tử. Báo cáo gồm leading/internal/trailing gaps, duplicate, dòng không hợp lệ, ngày đảo thứ tự, kỳ loại trừ, nguồn, độ phủ tài chính và lịch quay đã đến hạn. Các mã loại trừ lấy từ `data/seed/exclusions.json`; chúng không vào queue cào bù. Keno/Bingo18 được kiểm tra theo ID đã quan sát; không dùng lịch quay hiện tại để suy ra số kỳ lịch sử.

`boundary_verified=false` nghĩa là chỉ biết ID lớn nhất đã có. Không đồng nghĩa đã cập nhật đến hiện tại. `--probe-latest` xác nhận boundary khi endpoint trả bản ghi hợp lệ; thất bại được giữ trong `latest_probe_errors`. Có thể cung cấp `--latest-ids /path/to/verified_latest.json` với JSON như `{"mega645":1570}`, sau khi xác minh độc lập và ghi nguồn/thời điểm xác minh. Kiểm tra lịch quay vẫn chặn kết luận hoàn tất nếu còn slot đến hạn, kể cả Lotto 13h/21h cùng ngày.

## Cào bù và chạy tiếp

```bash
vlm-audit --game keno --repair --max-repair 20 --concurrency 2
vlm-audit --game bingo18 --repair --max-repair 20 --concurrency 2
vlm-audit --game all --repair --max-repair 100 \
  --database duckdb:data/local/vlm.duckdb
```

Repair chọn đúng kỳ thiếu, ưu tiên internal/trailing trước lịch sử đầu dải lớn. Ngày gợi ý lấy từ hàng xóm đã có; thử cả hai ngày/tháng nếu nguồn đầu thất bại. Không bịa ngày cho lịch sử đầu dải. Từng bản ghi được kiểm định lại trước transaction. `repaired_draws.jsonl` và `repair_status.json` trong `data/local/repair/` ghi checkpoint. Sau repair, báo cáo được tính lại từ seed + CSDL; lần chạy tiếp không lấy lại kỳ đã sửa thành công. Các kỳ thất bại vẫn nằm trong missing list và có thể thử lại khi nguồn hoạt động.

Ngân sách một kỳ repair mặc định 180 giây; một lần chạy lớn có thể kéo dài. Lịch quay đến hạn chưa có latest ID được ghi trong `scheduled_updates_due`, không tự gán ID mới. Cần `--probe-latest` hoặc boundary đã xác minh để đưa các kỳ cuối dải đó vào repair queue.

Snapshot kiểm thử trực tiếp của đợt audit có một kỳ Mega #01570 trong `reports/vlm/repaired_draws.jsonl`, có nguồn nhưng chưa có tài chính. Để tái lập CSDL local từ snapshot đã commit:

```python
import json
from pathlib import Path
from vlm.database.schema import DrawRecord, SQLRepository

Path("data/local").mkdir(parents=True, exist_ok=True)
repo = SQLRepository("sqlite:///data/local/vlm.sqlite")
repo.upsert(DrawRecord.model_validate(json.loads(line))
            for line in Path("reports/vlm/repaired_draws.jsonl").read_text().splitlines())
repo.close()
```

Sau đó chạy lại `vlm-audit`. Seed gốc và kho DuckDB cũ không bị ghi đè; đây là CSDL chuẩn hóa bổ sung. Muốn chuyển toàn bộ seed, sử dụng `audit_rows` và `DrawRecord.from_legacy` theo batch, loại `excluded_ids` trước khi import và kiểm tra báo cáo trước khi dùng dữ liệu.

## Schema và xuất dữ liệu

`DrawRecord` có các trường `draw_id`, `draw_date`, `winning_numbers`, `bonus_number`, `jackpot1_value`, `jackpot2_value`, `jackpot1_winners`, `jackpot2_winners`, `sub_prizes_json`, precision và provenance. Kho có bảng riêng cho cả 9 game, khóa `(game_type, draw_id)`. SQLite/PostgreSQL dùng SQLAlchemy; DuckDB dùng transaction native. Parquet có cột phẳng và cột JSON, xuất qua file tạm rồi thay thế.

- Giờ chuẩn Asia/Ho_Chi_Minh (UTC+7); chuỗi đầy đủ `YYYY-MM-DD HH:mm:ss` hoặc ISO tương đương.
- Archive chỉ có ngày lưu midnight với `time_precision="day"`; không khẳng định kỳ thực tế quay lúc 00:00. Keno/Bingo vẫn cần nguồn có giờ nếu phân tích trong ngày.
- Mega/Power/Keno kiểm tra range và distinct; Power bonus khác sáu số chính; Lotto bonus độc lập 1–12. Max giữ số `000`; Bingo cho phép lặp.
- `NULL` là chưa biết. Không suy ra số vé trúng, doanh thu hoặc quỹ thiếu thành 0.
- Upsert giữ nested metadata trên cập nhật thiếu/empty; số trúng, ngày chính xác và tài chính đã biết mâu thuẫn gây `DataConflict`, rollback toàn bộ batch. Cần đối soát nguồn mâu thuẫn trước khi thay thế.

```python
from vlm.database.schema import SQLRepository, DuckRepository

sql = SQLRepository("sqlite:///data/local/vlm.sqlite")
sql.export_parquet("data/local/vlm.parquet")
duck = DuckRepository("data/local/vlm.duckdb")
duck.import_parquet("data/local/vlm.parquet")
duck.close()
sql.close()

# PostgreSQL: SQLRepository("postgresql+psycopg://user:password@host/db")
```

SQLite/DuckDB/Parquet đã kiểm tra round-trip và rollback. PostgreSQL đã kiểm tra DDL, cần kiểm thử thêm với service thật trước rollout.

## Giá vé và tính thưởng

Tiền trả về là VND trước thuế, dùng `Fraction` khi chia quỹ. `Settlement.gross_payout=None` khi vé trúng Jackpot nhưng thiếu tổng quỹ hoặc tổng lượt trúng. `nominal=True` khi chưa biết toàn bộ lượt trúng để áp trần giải cố định. Số lượt trúng toàn thị trường bao gồm các lượt trúng của vé đang xét.

```python
from vlm.rules.calculator import matrix_cost, settle_matrix

assert matrix_cost("mega645", list(range(1, 19))).cost == 185_640_000
result = settle_matrix("mega645", list(range(1, 8)), [1, 2, 3, 4, 5, 6],
                       jackpot_pots={"jackpot1": 12_000_000_000},
                       total_jackpot_winners={"jackpot1": 2})
assert result.prize_counts == {"jackpot1": 1, "first": 6}
assert result.gross_payout == 6_060_000_000
```

`jackpot*_value` là tổng quỹ trả trong kỳ, bằng giá trị per-winner trên HTML nhân `max(1, winners)` khi có số lượt trúng. Canonical prizes được chuẩn hóa bằng cùng tier mapping của engine. Power `settle_matrix` mặc định nhận quỹ **đã phân phối** (`jackpot_pots_basis="payable"`), tránh chuyển phần vượt 300 tỷ lần hai. Dùng `"pre_transfer"` và đầy đủ hai quỹ/hai số lượt trúng khi tính từ quỹ trước phân phối; `power_jackpot_distribution` cung cấp riêng số chuyển và quỹ nền kỳ tiếp.

- Mega/Power: Bao 5 mở rộng theo số bổ sung từ phần bù; Bao 7–15/18 có `C(n,6)` lượt. Jackpot chia theo lượt trúng, không trả toàn quỹ cho từng tổ hợp.
- Max: `co_ban`, Pro `bao_bo_so`/`bao_nhieu_bo_so`; permutation loại trùng chữ số, roll Pro có thứ tự. Giá gồm `stake_multiple`. Một vé có thể cộng nhiều hạng và nhiều số cùng trúng. Hạng `one` đếm từng số khớp, không gộp cả hai số thành một giải. Các alias tier nội bộ cũ được giữ để tương thích engine.
- Max 3D/3D+ đảo số/bao vị trí và roll ngoài danh mục đã xác minh chỉ chạy khi `allow_unverified=True`, `documented=False`; không coi đó là hình thức bán đã được chứng nhận. Edge case hai số giống nhau và nhiều số quay lặp ở các hạng `both` cần đối soát thêm thể lệ theo thời điểm.
- Keno: đủ bậc 1–10, tám cửa phụ; trần top mỗi bậc 8/9/10 là 10 tỷ/kỳ. Bậc 5 trùng 4: `rules_variant="product"` = 150.000đ, `"detail"` = 0đ, kèm note mâu thuẫn nguồn.
- Bingo18: 38 cửa gồm 6 single, 6 double, 6 triple, any-triple, 16 tổng và small/tie/big. Tổng 3–18 tính với số lặp đúng nguyên trạng.

Lotto/Max4D có trong audit để giữ phạm vi repo. Thuế và hạn mức tài khoản/kênh SMS thuộc engine/channel riêng; `vlm.rules` không chứng nhận các ràng buộc giao dịch đó.

## Kiểm thử

```bash
python -m pytest -ra
ruff check --select F,B023 src scripts tests
```

Tests Bao 5/7/18 so với khai triển vé độc lập, Jackpot per-winner → pool → settlement, payable/pre-transfer, Max cộng thưởng, Keno caps/cửa phụ, Bingo38, strict schema/transaction/Parquet, retry/clearance/gzip, wrong-game fallback, repair qua tháng và chạy tiếp từ checkpoint.
