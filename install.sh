#!/usr/bin/env bash
# Bộ cài Vietlott Quant Engine cho Linux / macOS.
#
#   bash install.sh                 cài gói, nạp dữ liệu đi kèm, học bộ dự báo, kiểm tra
#   bash install.sh --dev           thêm công cụ phát triển (pytest, ruff, mypy) và chạy bộ test
#   bash install.sh --skip-forecast không học bộ dự báo lúc cài (học sau: ./vietlott forecast fit)
#   bash install.sh --offline       không kiểm tra mạng ở bước cuối
#   bash install.sh --no-deps       không tải gói phụ thuộc (dùng gói đã có trong Python hệ thống)
#   bash install.sh --python /usr/bin/python3.12 --venv .venv
#
# Cài lại an toàn: chạy lại bất cứ lúc nào; dữ liệu đã đồng bộ và sổ dự báo được giữ.
# Biến VQE_PYTHON chọn sẵn bản Python (như --python).
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv"
PYTHON_BIN="${VQE_PYTHON:-}"
DEV=0
SKIP_FORECAST=0
OFFLINE=0
NO_DEPS=0

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mLỗi:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dev) DEV=1 ;;
    --skip-forecast) SKIP_FORECAST=1 ;;
    --offline) OFFLINE=1 ;;
    --no-deps) NO_DEPS=1 ;;
    --python) shift; PYTHON_BIN="${1:?thiếu đường dẫn sau --python}" ;;
    --venv) shift; VENV="${1:?thiếu thư mục sau --venv}" ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "không hiểu tùy chọn '$1' (xem: bash install.sh --help)" ;;
  esac
  shift
done

# ------------------------------------------------------------------ 1. Python ≥ 3.11 có venv
# Debian/Ubuntu tách venv/ensurepip ra gói riêng: bản thiếu chúng bị bỏ qua.
version_ok() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; }
if [ -z "$PYTHON_BIN" ]; then
  for c in python3.14 python3.13 python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && version_ok "$c"; then PYTHON_BIN="$(command -v "$c")"; break; fi
  done
fi
if [ -z "$PYTHON_BIN" ] || ! version_ok "$PYTHON_BIN"; then
  die "cần Python 3.11 trở lên, có kèm venv.
  Ubuntu 24.04 / Debian 13: sudo apt install python3 python3-venv
  Bản Linux cũ hơn: cài Python 3.12 qua pyenv, conda hoặc kho của bản phân phối
  macOS (Homebrew): brew install python@3.12
  rồi chạy lại: bash install.sh   (hoặc: bash install.sh --python /đường/dẫn/python3.12)"
fi
say "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])') ($PYTHON_BIN)"

# ------------------------------------------------------------------ 2. môi trường ảo
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
  warn "Môi trường ảo $VENV hỏng (thường do đổi bản Python); tạo lại"
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  say "Tạo môi trường ảo $VENV"
  if [ "$NO_DEPS" = 1 ]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV" || die "không tạo được môi trường ảo"
  else
    "$PYTHON_BIN" -m venv "$VENV" || die "không tạo được môi trường ảo"
  fi
fi
VPY="$VENV/bin/python"

# ------------------------------------------------------------------ 3. cài gói (editable: git pull xong không cần cài lại)
TARGET="."
[ "$DEV" = 1 ] && TARGET=".[dev]"
if [ "$NO_DEPS" = 1 ]; then
  say "Cài vietlott-quant-engine (không tải phụ thuộc)"
  "$VPY" -m pip install --no-deps --no-build-isolation -e "$TARGET" || die "pip install thất bại (chế độ --no-deps cần setuptools trong Python hệ thống)"
else
  say "Cập nhật pip và cài vietlott-quant-engine cùng các gói phụ thuộc"
  "$VPY" -m pip install --upgrade pip >/dev/null || warn "không cập nhật được pip, tiếp tục với bản hiện có"
  "$VPY" -m pip install -e "$TARGET" || die "pip install thất bại — xem thông báo của pip ở trên (mạng tới pypi.org, proxy: biến HTTPS_PROXY)"
fi
# Cho lệnh vietlott biết thư mục dự án.
pwd > "$VENV/vietlott_home.txt"

# ------------------------------------------------------------------ 4. cấu hình + lệnh tắt
[ -f .env ] || { cp .env.example .env; say "Tạo .env từ .env.example (sửa nếu cần)"; }
case "$VENV" in
  /*) LAUNCH="$VENV/bin/vietlott" ;;
  *)  LAUNCH="\$(cd \"\$(dirname \"\$0\")\" && pwd)/$VENV/bin/vietlott" ;;
esac
cat > vietlott <<EOF
#!/usr/bin/env bash
# Lệnh tắt do install.sh tạo: chạy vietlott trong môi trường ảo của dự án.
exec "$LAUNCH" "\$@"
EOF
chmod +x vietlott

# ------------------------------------------------------------------ 5. dữ liệu, bộ dự báo, kiểm tra
INIT_ARGS=()
[ "$SKIP_FORECAST" = 1 ] && INIT_ARGS+=(--skip-forecast)
[ "$OFFLINE" = 1 ] && INIT_ARGS+=(--offline)
say "Khởi tạo: nạp dữ liệu đi kèm, học bộ dự báo, kiểm tra cài đặt"
VQE_LOG_LEVEL=WARNING "$VENV/bin/vietlott" init ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} || die "bước khởi tạo báo lỗi — chạy ./vietlott doctor để xem chi tiết"

if [ "$DEV" = 1 ]; then
  say "Chạy bộ test"
  "$VPY" -m pytest -q || die "có test thất bại"
fi

cat <<EOF

Xong. Dùng:
  ./vietlott forecast next            dự báo kỳ tới cho cả 8 sản phẩm (ghi vào sổ nếu kỳ chưa quay)
  ./vietlott forecast update          sau kỳ quay: học kỳ mới, chấm các dự báo đã ghi
  ./vietlott products sync            đồng bộ Keno, Bingo18, Max 3D/Pro (tự dùng nguồn dự phòng)
  ./vietlott sync --source auto       đồng bộ Mega, Power, Lotto
  ./vietlott serve                    API tại http://127.0.0.1:8000/docs
  ./vietlott doctor                   kiểm tra lại cài đặt
Hoặc kích hoạt môi trường: source $VENV/bin/activate  rồi gõ  vietlott …
EOF
