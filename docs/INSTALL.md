# Cài đặt Vietlott Quant Engine

Repo này chạy độc lập, tách khỏi VLA (XSMB). Có bốn cách dùng; chọn cách hợp với máy của bạn.

| Cách | Hợp với | Kết quả |
|---|---|---|
| [Windows](#1-windows) | máy cá nhân Windows 10/11 | lệnh `vietlott` trong thư mục dự án |
| [Linux / macOS](#2-linux--macos) | máy cá nhân, máy chủ | lệnh `./vietlott` |
| [Docker](#3-docker) | máy chủ chạy liên tục | API + lịch đồng bộ 6 giờ/lần |
| [GitHub](#4-đưa-lên-github-repo-riêng) | không cần máy chủ | tự cập nhật 2 lần/ngày, trang GitHub Pages |

**Cần:** Python 3.11 trở lên (khuyên dùng 3.12), khoảng 1 GB đĩa, mạng tới pypi.org cho lần cài đầu. Dữ liệu lịch sử đi kèm repo (`data/seed`), nên phân tích và dự báo chạy được cả khi không có mạng.

## 1. Windows

1. Tải gói `vietlott-quant-engine-<phiên bản>.zip` trong mục **Releases** của repo (hoặc `git clone`), rồi giải nén.
2. Bấm đúp **`install.cmd`**.
   - Chưa có Python: mở *Command Prompt* trong thư mục này (xem bước 3) và chạy `install.cmd -InstallPython` (cài Python 3.12 bằng winget). Hoặc tự cài từ python.org, nhớ chọn *Add python.exe to PATH*.
   - Lần cài đầu mất vài phút: tải gói, nạp dữ liệu, học bộ dự báo (Keno ~300 nghìn kỳ, khoảng 1 phút).
3. Mở cửa sổ lệnh **trong thư mục dự án**: trong File Explorer, bấm vào thanh địa chỉ, gõ `cmd` rồi Enter. Sau đó:

```bat
vietlott forecast next
vietlott doctor
```

Nếu dùng PowerShell (Windows 11 "Open in Terminal" mặc định là PowerShell), thêm `.\` ở đầu: `.\vietlott.cmd forecast next`, `.\install.cmd -InstallPython`. PowerShell không chạy lệnh trong thư mục hiện tại nếu thiếu `.\`.

**Cập nhật mã** (git pull hoặc tải bản mới): chạy lại `install.cmd`. Dữ liệu đã đồng bộ và sổ dự báo được giữ. (Trên Windows, bộ cài cài gói ở chế độ thường để an toàn với tên thư mục có dấu; `-Dev` cài editable.)

Tùy chọn của bộ cài (dùng với `install.cmd` hoặc `install.ps1`):

| Tùy chọn | Tác dụng |
|---|---|
| `-InstallPython` | cài Python 3.12 bằng winget khi máy chưa có |
| `-SkipForecast` | không học bộ dự báo lúc cài (học sau: `vietlott forecast fit`) |
| `-Dev` | cài editable kèm pytest, ruff, mypy và chạy bộ test |
| `-Offline` | không kiểm tra mạng ở bước cuối |
| `-Python C:\đường\dẫn\python.exe` | dùng một bản Python cụ thể |

## 2. Linux / macOS

```bash
git clone https://github.com/<tài-khoản>/vietlott-quant-engine.git
cd vietlott-quant-engine
bash install.sh
./vietlott forecast next
```

Cần Python 3.11+ có kèm venv: Ubuntu 24.04 / Debian 13: `sudo apt install python3 python3-venv`; bản Linux cũ hơn: cài Python 3.12 qua pyenv, conda hoặc kho của bản phân phối; macOS: `brew install python@3.12`. Tùy chọn: `--dev`, `--skip-forecast`, `--offline`, `--python /đường/dẫn/python3.12` (hoặc biến `VQE_PYTHON`), `--venv thư-mục`. Trên Linux / macOS gói được cài editable: `git pull` xong không cần cài lại.

Muốn gõ `vietlott` ở bất cứ đâu: `source .venv/bin/activate`. Lệnh luôn chạy trên thư mục dự án; nếu cần, đặt `VQE_HOME=/đường/dẫn/vietlott-quant-engine`.

## 3. Docker

```bash
docker compose up -d                                  # API + bộ lập lịch
docker compose --profile jobs run --rm backtest       # chạy một lần toàn bộ báo cáo
```

API ở http://localhost:8000/docs. Bộ lập lịch đồng bộ 6 giờ/lần rồi gọi `/forecast/<sản phẩm>?record=true`: mô hình tự học kỳ mới và ghi dự báo vào sổ (chỉ khi kỳ đó chưa quay). Dữ liệu nằm trong volume `vqe-data`.

## 4. Đưa lên GitHub (repo riêng)

1. Tạo repo mới, ví dụ `vietlott-quant-engine`, rồi đẩy mã lên.

   **Bằng GitHub Desktop (Windows):**

   1. *File → New repository…* Name: `vietlott-quant-engine`. Giữ *Local path* (ví dụ `C:\Users\<bạn>\Documents\GitHub`). **Không** chọn *Initialize this repository with a README*; *Git ignore* và *License* để **None** (repo đã có sẵn các tệp này). Bấm *Create repository*.
   2. *Repository → Show in Explorer* để mở thư mục repo vừa tạo. Giải nén gói zip ở chỗ khác, mở thư mục `vietlott-quant-engine-<phiên bản>` bên trong, chọn **tất cả** (Ctrl+A, gồm cả `.github`, `.gitignore`, `.gitattributes`), chép và dán vào thư mục repo. Khi Windows hỏi về `.gitattributes`, chọn **Replace the file in the destination** (GitHub Desktop đã tạo sẵn một bản đơn giản; bản của repo quy định kiểu xuống dòng cho bộ cài).
   3. Quay lại GitHub Desktop, tab *Changes* liệt kê khoảng 170 tệp. Ô *Summary* ghi `Vietlott Quant Engine 4.0.0`, bấm *Commit to main*.
   4. Bấm *Publish repository*. **Bỏ chọn** *Keep this code private* nếu muốn có trang GitHub Pages (xem bên dưới). Bấm *Publish repository*.

   Không cần bước `chmod`: mọi nơi đều gọi `bash install.sh`, nên quyền chạy của tệp không quan trọng.

   **Bằng dòng lệnh git:**

   ```bash
   git init -b main && git add . && git update-index --chmod=+x install.sh
   git commit -m "Vietlott Quant Engine 4.0.0"
   git remote add origin https://github.com/<tài-khoản>/vietlott-quant-engine.git
   git push -u origin main
   ```

   `git update-index --chmod=+x install.sh` giữ quyền chạy của bộ cài khi bạn đẩy từ Windows.

   **Công khai hay riêng tư?** Với tài khoản GitHub miễn phí, GitHub Pages chỉ dùng được cho repo **công khai**; repo riêng tư cần gói trả phí (Pro trở lên). Repo riêng tư vẫn chạy được CI và workflow cập nhật (sổ dự báo vẫn được ghi), chỉ bước đăng trang báo lỗi. Phút chạy Actions của repo riêng tư có giới hạn, và phút macOS / Windows tính nhiều hơn Linux: workflow `installer.yml` (chạy bộ cài trên 3 hệ điều hành) vì vậy chỉ chạy khi bộ cài thay đổi, mỗi tuần một lần, hoặc khi bấm tay.

2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. Workflow tự khai báo quyền ghi nội dung và Pages. Nếu tài khoản/tổ chức đặt giới hạn, vào **Settings → Actions → General → Workflow permissions** và chọn **Read and write permissions**.
4. Vào **Actions → "Cập nhật dữ liệu, dự báo và trang" → Run workflow** để chạy lần đầu. Từ đó nó tự chạy lúc 10:10 và 22:40 giờ Việt Nam.

Workflow làm gì:

- đồng bộ qua chuỗi dự phòng (vietlott.vn → kho cộng đồng → bản sao GitHub). Máy chạy của GitHub ở nước ngoài nên vietlott.vn thường từ chối; đó là bình thường. Vì vậy dự báo có được ghi hay không phụ thuộc vào việc các kho sao chép đã có kỳ mới nhất chưa: thiếu kỳ thì bộ dự báo từ chối ghi (đúng như thiết kế) và thử lại ở lần chạy sau;
- học các kỳ mới, chấm các dự báo đã ghi, ghi dự báo cho kỳ **chưa quay** theo lịch quay chính thức (cron của GitHub có thể trễ vài giờ, kỳ đã quay thì không ghi);
- ghi `data/forecast/ledger.jsonl` vào repo (`[skip ci]`), dựng trang và đăng lên `https://<tài-khoản>.github.io/vietlott-quant-engine/`.

Kho kết quả và trạng thái bộ dự báo được giữ bằng cache của Actions. Mất cache không làm sai gì: lần chạy sau dựng lại từ dữ liệu đi kèm và nguồn, và bộ dự báo học lại cho đúng kết quả cũ.

**Dùng từ VLA hoặc trang khác:** đọc JSON trên trang đã đăng — `data/summary.json` (mỗi sản phẩm một dòng), `data/forecast/<sản phẩm>.json` (đầy đủ, kèm đường e-value), `data/scoreboard.json`, `data/ledger.jsonl`.

**CI** (`.github/workflows/ci.yml`) chạy bộ test trên Python 3.11 và 3.12 (Linux) và 3.12 (Windows) ở mỗi lần đẩy mã. **`installer.yml`** chạy chính các bộ cài trên Linux (`--dev`, gồm cả bộ test), macOS và Windows (`install.cmd`, PowerShell 5.1).

**Dùng hằng ngày với GitHub Desktop:**

- Cài trên máy ngay trong thư mục repo: *Repository → Show in Explorer*, bấm đúp `install.cmd` (mục 1). *Repository → Open in Command Prompt* mở cửa sổ lệnh đúng thư mục để gõ `vietlott forecast next`.
- Workflow cập nhật commit sổ dự báo vào repo 2 lần/ngày, nên bản trên GitHub luôn đi trước máy bạn. Trước khi commit thay đổi của mình, bấm **Fetch origin → Pull origin**.
- Chạy trên máy không làm đổi tệp nào của repo: sổ và bộ dự báo của máy nằm ở `data/local/forecast/` (đặt trong `.env`, git bỏ qua), tách khỏi sổ công khai `data/forecast/ledger.jsonl` do workflow ghi. Nếu tab *Changes* hiện tệp bạn không sửa, đừng commit tệp đó.
- Sau khi kéo về **mã mới** (không chỉ sổ dự báo), chạy lại `install.cmd`.

## 5. Phát hành phiên bản

1. Sửa `__version__` trong `src/vietlott_engine/__init__.py` (pyproject đọc từ đây) và ghi `CHANGELOG.md`.
2. `git tag v4.0.1 && git push origin v4.0.1`.

Workflow `release.yml` kiểm tra phiên bản khớp tag rồi tạo Release kèm gói zip (mã nguồn + dữ liệu đi kèm + bộ cài), wheel và sdist. Người dùng chỉ cần tải **zip** rồi chạy bộ cài. Wheel và sdist dành cho nhà phát triển: chúng không chứa dữ liệu đi kèm (`data/seed`), nên cần đặt `VQE_HOME` tới một bản sao của repo.

## 6. Lệnh thường dùng

| Lệnh | Việc |
|---|---|
| `vietlott init` | nạp dữ liệu đi kèm, học bộ dự báo, kiểm tra (bộ cài tự chạy) |
| `vietlott doctor` | kiểm tra Python, gói, dữ liệu, kho, bộ dự báo, mạng |
| `vietlott forecast next` | học kỳ mới, dự báo kỳ tới, ghi sổ nếu kỳ chưa quay |
| `vietlott forecast update` | sau kỳ quay: học kỳ mới, chấm các dự báo đã ghi |
| `vietlott forecast scoreboard` | thành tích thật của các dự báo ghi trước kỳ quay |
| `vietlott sync --game all --source auto` | đồng bộ Mega, Power, Lotto |
| `vietlott products sync` | đồng bộ Keno, Bingo18, Max 3D, Max 3D Pro |
| `vietlott serve` | API tại http://127.0.0.1:8000/docs |
| `python scripts/build_site.py` | dựng trang tĩnh vào `site/` như trên GitHub Pages |

Danh sách đầy đủ: `vietlott --help`, và README mục 3.

## 7. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| "cần Python 3.11 trở lên" | cài Python 3.12 (`install.cmd -InstallPython` trên Windows). Lối tắt `python` của Microsoft Store khi chưa cài Python không chạy được nên bộ cài bỏ qua; Python cài từ Microsoft Store thì dùng được |
| PowerShell báo không được chạy script | dùng `install.cmd`, hoặc `powershell -ExecutionPolicy Bypass -File .\install.ps1` |
| `pip install` lỗi mạng | mạng công ty: đặt `HTTPS_PROXY`; rồi chạy lại bộ cài (chạy lại an toàn) |
| `doctor` báo vietlott.vn HTTP 403 hoặc Cloudflare | bình thường khi ở ngoài Việt Nam hoặc khi trang yêu cầu xác minh người dùng; `--source auto` dùng nguồn dự phòng. Muốn đối chiếu với trang chính thức: README mục 1.8 |
| "không ghi vào sổ: …" | kỳ được dự báo đã hoặc sắp quay theo lịch, hoặc dữ liệu chưa đủ: đồng bộ (`vietlott sync --game all --source auto`, `vietlott products sync`) rồi chạy lại. Ngay sau khi cài, dữ liệu đi kèm thường đã cũ vài ngày nên lần đầu sẽ gặp thông báo này. Keno/Bingo18 chỉ ghi trong khoảng 22:15–05:55 |
| Lỗi khi chạy `vietlott serve` cùng lúc với lệnh khác | DuckDB chỉ cho một tiến trình ghi một tệp kho: dừng `serve` trước, hoặc gọi API thay cho lệnh |
| Chữ tiếng Việt bị lỗi khi chuyển đầu ra sang tệp | lệnh `vietlott` tự ghi UTF-8; mở tệp bằng trình soạn thảo hỗ trợ UTF-8 |

**Gỡ cài đặt:** xóa thư mục `.venv`, `data/local` (sổ dự báo và bộ dự báo đã học trên máy này; giữ lại nếu muốn giữ sổ), các tệp `vietlott`, `vietlott.cmd`, `.env` và `data/*.duckdb*`. Hoặc xóa cả thư mục dự án.
