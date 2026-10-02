"""v4.0: package layout, project-root detection, installers, static site."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"


def test_project_root_is_found_from_anywhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vietlott_engine.paths import enter_project, project_root

    monkeypatch.delenv("VQE_HOME", raising=False)
    monkeypatch.chdir(ROOT / "tests")
    assert project_root() == ROOT  # a parent folder holds pyproject.toml + data/seed
    monkeypatch.chdir(tmp_path)
    assert project_root() == ROOT  # not under the project: the folder the package lives in
    assert enter_project() == ROOT and Path.cwd() == ROOT
    monkeypatch.setenv("VQE_HOME", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_version_and_help() -> None:
    from vietlott_engine import __version__
    from vietlott_engine.cli import build_parser

    text = build_parser().format_help()
    assert __version__ == "4.0.0" and "init" in text and "doctor" in text and "forecast" in text
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--version"])
    assert e.value.code == 0


def test_installer_files_are_well_formed() -> None:
    sh = ROOT / "install.sh"
    assert sh.read_bytes().startswith(b"#!/usr/bin/env bash\n") and b"\r\n" not in sh.read_bytes()
    if sys.platform != "win32":  # Windows' bash.exe may be the WSL stub
        subprocess.run(["bash", "-n", str(sh)], check=True)
    ps1 = (ROOT / "install.ps1").read_bytes()
    assert ps1.startswith(b"\xef\xbb\xbf") and b"\r\n" in ps1 and b"\n" not in ps1.replace(b"\r\n", b"")  # UTF-8 BOM + CRLF for Windows PowerShell 5.1
    cmd = (ROOT / "install.cmd").read_bytes()
    assert cmd.isascii() and b"install.ps1" in cmd and b"ExecutionPolicy Bypass" in cmd
    attrs = (ROOT / ".gitattributes").read_text()
    assert "*.sh text eol=lf" in attrs and "*.ps1 text eol=crlf" in attrs
    # local runs keep their own ledger, so they never touch the public one that GitHub Actions commits
    assert "\nVQE_FORECAST_DIR=data/local/forecast\n" in (ROOT / ".env.example").read_text(encoding="utf-8")
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/local/" in ignored and "!data/forecast/ledger.jsonl" in ignored


def test_workflows_reference_real_commands() -> None:
    yaml = pytest.importorskip("yaml")
    wf = {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in (ROOT / ".github" / "workflows").glob("*.yml")}
    assert {"ci.yml", "installer.yml", "update.yml", "release.yml"} <= set(wf)
    names = [s.get("name", "") for s in wf["update.yml"]["jobs"]["update"]["steps"]]
    assert names.index("Ghi sổ dự báo vào repo") < names.index("Dựng trang")  # the ledger is committed even if the site fails
    steps = " ".join(str(s.get("run", "")) for job in wf["update.yml"]["jobs"].values() for s in job.get("steps", []))
    for command in ("vietlott init", "vietlott sync", "vietlott products sync", "vietlott forecast next", "scripts/build_site.py"):
        assert command in steps
    assert (ROOT / "scripts" / "build_site.py").exists()


def test_site_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vietlott_engine.forecast.data import load_series
    from vietlott_engine.forecast.engine import Forecaster, record, state_path

    fdir = tmp_path / "forecast"
    for code in ("max3dpro", "lotto535"):
        f = Forecaster(code)
        f.update(load_series(code, seed_dir=SEED))
        f.save(state_path(fdir, f.product))
        if code == "max3dpro":
            record(fdir, f.forecast())
    out = tmp_path / "site"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py"), "--out", str(out), "--dir", str(fdir)], capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])})
    assert r.returncode == 0, r.stderr
    page = (out / "index.html").read_text(encoding="utf-8")
    assert '<html lang="vi">' in page and "Max 3D Pro" in page and "Lotto 5/35" in page and "<script" not in page
    summary = json.loads((out / "data" / "summary.json").read_text(encoding="utf-8"))
    assert [p["product"] for p in summary["products"]] == ["lotto535", "max3dpro"]
    assert summary["products"][1]["evidence_found"] is True
    assert (out / "data" / "ledger.jsonl").exists() and (out / "data" / "forecast" / "max3dpro.json").exists() and (out / ".nojekyll").exists()
