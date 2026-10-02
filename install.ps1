<#
.SYNOPSIS
  Bộ cài Vietlott Quant Engine cho Windows (Windows PowerShell 5.1 hoặc PowerShell 7).

.DESCRIPTION
  Tạo môi trường ảo .venv, cài gói và phụ thuộc, nạp dữ liệu đi kèm, học bộ dự báo cho 8 sản phẩm,
  tạo lệnh tắt vietlott.cmd và kiểm tra cài đặt. Chạy lại bất cứ lúc nào (kể cả sau khi cập nhật mã):
  dữ liệu đã đồng bộ và sổ dự báo được giữ.

  Cách nhanh nhất: bấm đúp install.cmd. Hoặc trong PowerShell:
    powershell -ExecutionPolicy Bypass -File .\install.ps1

.PARAMETER Dev            Cài ở chế độ sửa mã (editable) kèm pytest, ruff, mypy, rồi chạy bộ test.
.PARAMETER SkipForecast   Không học bộ dự báo lúc cài (học sau: vietlott forecast fit).
.PARAMETER Offline        Không kiểm tra mạng ở bước cuối.
.PARAMETER NoDeps         Không tải gói phụ thuộc (dùng gói đã có trong Python hệ thống; cần setuptools).
.PARAMETER InstallPython  Nếu chưa có Python 3.11+, cài Python 3.12 bằng winget.
.PARAMETER Python         Đường dẫn python.exe muốn dùng.
.PARAMETER Venv           Thư mục môi trường ảo (mặc định .venv).
#>
[CmdletBinding()]
param(
  [switch]$Dev,
  [switch]$SkipForecast,
  [switch]$Offline,
  [switch]$NoDeps,
  [switch]$InstallPython,
  [string]$Python = "",
  [string]$Venv = ".venv"
)

# "Continue": Windows PowerShell 5.1 would turn a native program's warnings on stderr (pip, py) into
# terminating errors under "Stop"; every native call is checked through its exit code instead.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot -ErrorAction Stop
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$OnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or ($IsWindows -eq $true)
$NoBom = New-Object System.Text.UTF8Encoding($false)

function Say([string]$m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn([string]$m) { Write-Host "!!  $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Host "Lỗi: $m" -ForegroundColor Red; exit 1 }

# Chạy một chương trình ngoài và dừng nếu nó không chạy được hoặc báo lỗi qua mã thoát.
function Run([string]$exe, [string[]]$argv, [string]$what) {
  if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { Fail "$what (không tìm thấy $exe)" }
  $global:LASTEXITCODE = 0
  try { & $exe @argv } catch { Fail "$what ($($_.Exception.Message))" }
  if ($LASTEXITCODE -ne 0) { Fail "$what (mã thoát $LASTEXITCODE)" }
}

# ------------------------------------------------------------------ 1. Python ≥ 3.11 có venv
# Trả về đường dẫn python.exe thật nếu lệnh chạy được, đủ phiên bản và tạo được môi trường ảo.
# (Lối tắt "python" của Microsoft Store khi chưa cài Python không chạy được nên tự bị loại.)
function Test-Python([string]$exe, [string[]]$pre) {
  if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { return $null }
  try {
    $global:LASTEXITCODE = 0
    $v = & $exe @pre -c "import sys, venv, ensurepip; print('%d.%d' % sys.version_info[:2]); print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $v -or $v.Count -lt 2) { return $null }
    $parts = ($v[0]).Split(".")
    if ([int]$parts[0] -gt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11)) { return $v[1] }
  } catch { }
  return $null
}

function Find-Python {
  if ($Python) { return (Test-Python $Python @()) }
  $candidates = @(
    @("py", @("-3.14")), @("py", @("-3.13")), @("py", @("-3.12")), @("py", @("-3.11")), @("py", @("-3")),
    @("python", @()), @("python3", @())
  )
  foreach ($c in $candidates) {
    $found = Test-Python $c[0] $c[1]
    if ($found) { return $found }
  }
  # Bản cài "cho người dùng hiện tại" của python.org / winget, khi PATH chưa được cập nhật.
  if ($env:LOCALAPPDATA) {
    foreach ($ver in @("314", "313", "312", "311")) {
      $p = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$ver\python.exe"
      if (Test-Path -LiteralPath $p) {
        $found = Test-Python $p @()
        if ($found) { return $found }
      }
    }
  }
  return $null
}

$py = Find-Python
if (-not $py -and $InstallPython -and $OnWindows) {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Say "Cài Python 3.12 bằng winget"
    winget install -e --id Python.Python.3.12 --scope user --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { Warn "winget báo mã thoát $LASTEXITCODE; thử tìm Python lần nữa" }
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
    $py = Find-Python
  } else {
    Warn "Không có winget trên máy này."
  }
}
if (-not $py) {
  Fail ("cần Python 3.11 trở lên.`n" +
        "  Cách 1: chạy lại với -InstallPython (dùng winget):  install.cmd -InstallPython`n" +
        "  Cách 2: tải Python 3.12 tại https://www.python.org/downloads/windows/ , khi cài nhớ chọn `"Add python.exe to PATH`"`n" +
        "  rồi chạy lại install.cmd")
}
$pyver = & $py -c "import sys; print(sys.version.split()[0])"
Say "Python: $pyver ($py)"

# ------------------------------------------------------------------ 2. môi trường ảo
if ($OnWindows) { $vpy = Join-Path $Venv "Scripts\python.exe"; $vcli = Join-Path $Venv "Scripts\vietlott.exe" }
else            { $vpy = Join-Path $Venv "bin/python";         $vcli = Join-Path $Venv "bin/vietlott" }
if (Test-Path -LiteralPath $vpy) {
  $global:LASTEXITCODE = 0
  & $vpy -c "import sys" 2>$null
  if ($LASTEXITCODE -ne 0) { Warn "Môi trường ảo $Venv hỏng (thường do đổi bản Python); tạo lại"; Remove-Item -Recurse -Force -LiteralPath $Venv }
}
if (-not (Test-Path -LiteralPath $vpy)) {
  Say "Tạo môi trường ảo $Venv"
  if ($NoDeps) { Run $py @("-m", "venv", "--system-site-packages", $Venv) "không tạo được môi trường ảo" }
  else         { Run $py @("-m", "venv", $Venv) "không tạo được môi trường ảo" }
}

# ------------------------------------------------------------------ 3. cài gói
# Mặc định cài thường (không editable): an toàn cả khi đường dẫn thư mục có chữ tiếng Việt.
# Sau khi cập nhật mã (git pull hoặc tải bản mới) chỉ cần chạy lại bộ cài.
$pipArgs = @("-m", "pip", "install")
if ($Dev) { $pipArgs += @("-e", ".[dev]") } else { $pipArgs += @(".") }
if ($NoDeps) {
  Say "Cài vietlott-quant-engine (không tải phụ thuộc)"
  Run $vpy ($pipArgs + @("--no-deps", "--no-build-isolation")) "pip install thất bại"
} else {
  Say "Cập nhật pip và cài vietlott-quant-engine cùng các gói phụ thuộc"
  $global:LASTEXITCODE = 0
  & $vpy -m pip install --upgrade pip | Out-Null
  if ($LASTEXITCODE -ne 0) { Warn "không cập nhật được pip, tiếp tục với bản hiện có" }
  Run $vpy $pipArgs "pip install thất bại — xem thông báo của pip ở trên (mạng tới pypi.org, proxy: biến HTTPS_PROXY)"
}
# Cho lệnh vietlott biết thư mục dự án (đọc bằng UTF-8, an toàn với tên thư mục có dấu).
[System.IO.File]::WriteAllText((Join-Path (Resolve-Path -LiteralPath $Venv).Path "vietlott_home.txt"), $PSScriptRoot, $NoBom)

# ------------------------------------------------------------------ 4. cấu hình + lệnh tắt
if (-not (Test-Path -LiteralPath ".env")) { Copy-Item ".env.example" ".env" -ErrorAction Stop; Say "Tạo .env từ .env.example (sửa nếu cần)" }
# Lệnh tắt dùng đường dẫn tương đối với thư mục dự án khi có thể (an toàn với tên thư mục có dấu).
$rel = -not [System.IO.Path]::IsPathRooted($Venv)
if ($OnWindows) {
  $exe = if ($rel) { "%~dp0$Venv\Scripts\vietlott.exe" } else { $vcli }
  $lines = @("@echo off", "rem Lenh tat do install.ps1 tao: chay vietlott trong moi truong ao cua du an.", "`"$exe`" %*", "exit /b %ERRORLEVEL%")
  [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot "vietlott.cmd"), (($lines -join "`r`n") + "`r`n"), $NoBom)
} else {
  $exe = if ($rel) { "`$(dirname `"`$0`")/$Venv/bin/vietlott" } else { $vcli }
  $lines = @("#!/usr/bin/env bash", "exec `"$exe`" `"`$@`"")
  [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot "vietlott"), (($lines -join "`n") + "`n"), $NoBom)
  & chmod +x vietlott
}

# ------------------------------------------------------------------ 5. dữ liệu, bộ dự báo, kiểm tra
$initArgs = @("init")
if ($SkipForecast) { $initArgs += "--skip-forecast" }
if ($Offline) { $initArgs += "--offline" }
$env:VQE_LOG_LEVEL = "WARNING"
Say "Khởi tạo: nạp dữ liệu đi kèm, học bộ dự báo, kiểm tra cài đặt"
Run $vcli $initArgs "bước khởi tạo báo lỗi — chạy vietlott doctor để xem chi tiết"

if ($Dev) {
  Say "Chạy bộ test"
  Run $vpy @("-m", "pytest", "-q") "có test thất bại"
}

Write-Host ""
Write-Host "Xong. Trong thư mục này:" -ForegroundColor Green
if ($OnWindows) {
  Write-Host "  Command Prompt:  vietlott forecast next"
  Write-Host "  PowerShell:      .\vietlott.cmd forecast next"
  $l = "vietlott"
} else {
  $l = "./vietlott"
}
Write-Host "  $l forecast next          dự báo kỳ tới cho cả 8 sản phẩm (ghi vào sổ nếu kỳ chưa quay)"
Write-Host "  $l forecast update        sau kỳ quay: học kỳ mới, chấm các dự báo đã ghi"
Write-Host "  $l products sync          đồng bộ Keno, Bingo18, Max 3D/Pro (tự dùng nguồn dự phòng)"
Write-Host "  $l sync --source auto     đồng bộ Mega, Power, Lotto"
Write-Host "  $l serve                  API tại http://127.0.0.1:8000/docs"
Write-Host "  $l doctor                 kiểm tra lại cài đặt"
if ($OnWindows) { $again = "install.cmd" } else { $again = "pwsh ./install.ps1 (hoặc bash install.sh)" }
Write-Host "Cập nhật mã xong (git pull hoặc tải bản mới): chạy lại $again."
