@echo off
rem Bo cai Vietlott Quant Engine cho Windows: bam dup vao tep nay.
rem Tuy chon duoc chuyen tiep cho install.ps1, vi du:  install.cmd -InstallPython -SkipForecast
rem (Trong PowerShell, go .\install.cmd thay vi install.cmd.)
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set RC=%ERRORLEVEL%
echo.
if not "%RC%"=="0" (
  echo Cai dat chua xong - xem thong bao loi o tren.
) else (
  echo Cai dat xong. Trong thu muc nay:
  echo   Command Prompt:  vietlott forecast next
  echo   PowerShell:      .\vietlott.cmd forecast next
)
if not defined CI pause
exit /b %RC%
