# VLM Integrity Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bàn giao bốn module được yêu cầu, Audit Matrix và kiểm thử có thể chạy lại.
**Architecture:** API `src/vlm` tích hợp công thức/parser sẵn có, crawler và schema mới. Repair lưu riêng và cập nhật kho dữ liệu sau validation.
**Tech Stack:** Python 3.11+, Pydantic 2, SQLAlchemy 2, DuckDB, curl_cffi tùy chọn, pytest.
**Spec:** `docs/superpowers/specs/2026-10-03-vlm-integrity-design.md`.

## Global Constraints

- Thời gian theo Asia/Ho_Chi_Minh (UTC+7); ngày không có giờ được đánh dấu `day`.
- Không điền số liệu không quan sát được; khóa kết hợp game/id; mâu thuẫn kết quả phải báo lỗi.
- Không bảo đảm lấy đủ lịch sử nếu các nguồn vẫn bị chặn; báo cáo tách rõ phần chưa xác minh.

## Review Focus

- HTML challenge trả 200 phải bị từ chối.
- Kho rỗng và thiếu kỳ đầu phải được báo.
- Kỳ `not_confirmed` không được cào bù như dữ liệu hợp lệ.
- Bao Power có nhiều lượt JP2 chia cùng quỹ, không nhân nguyên quỹ.
- Metadata mới không được xóa trường đã biết; số 000 và repetition trong Bingo/Max được giữ.

### Task 1: Schema và persistence

Files: `src/vlm/database/schema.py`, `tests/test_vlm_schema.py`.
Interfaces: `DrawRecord.from_legacy(game, row)`, `SQLRepository(url).upsert(records)`, `DuckRepository(path)`, `export_parquet(path)`.
- [ ] Viết/run test invalid range, bonus, datetime, duplicate/conflict và round-trip.
- [ ] Tạo models/backend; chạy lại test.

### Task 2: Tính thưởng

Files: `src/vlm/rules/calculator.py`, `tests/test_vlm_calculator.py`.
Interfaces: `matrix_cost`, `settle_matrix`, `power_jackpot_distribution`, `settle_max3d`, `settle_keno`, `settle_bingo18`.
- [ ] Viết/run test Bao 5/7/18, bonus và sharing, Max cumulative, fast-game boundaries.
- [ ] Triển khai bằng công thức/game specs hiện có; kiểm chứng với brute force.

### Task 3: Crawler và repair

Files: `src/vlm/crawler/async_scraper.py`, `src/vlm/crawler/sources.py`, `src/vlm/tools/gap_analyzer.py`, `tests/test_vlm_crawler.py`, `tests/test_vlm_gaps.py`.
Interfaces: `AsyncScraper.request/get/post`, `TargetedFetcher.fetch(game,draw_id,date_hint)`, `analyze_game`, `repair_missing`.
- [ ] Viết/run test retry, challenge 200, solver/fallback, wrong id, leading/trailing/excluded gaps, atomic repair.
- [ ] Triển khai transport và adapter official/GitHub/archive; tích hợp CLI.
- [ ] Audit seed tất cả sản phẩm; repair bounded, lưu kết quả và thất bại.

### Task 4: QA và bàn giao

Files: `reports/VLM_AUDIT_2026-10-03.md`, `docs/VLM_INTEGRITY.md`, dependency/CLI metadata.
- [ ] Chạy pytest toàn repo, Ruff và đóng gói wheel; sửa các lỗi phát hiện trong phạm vi thay đổi.
- [ ] Báo cáo chính xác số kỳ thiếu, coverage chưa xác minh, nguồn/quy tắc còn tranh chấp.
- [ ] Commit nhánh và tạo PR reviewable; không merge main khi chưa được yêu cầu.
