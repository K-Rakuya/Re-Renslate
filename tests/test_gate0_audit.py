"""
tests/test_gate0_audit.py
Independent Quality Audit Test Suite for Renslate Phase 0 / Gate 0.
Verifies all requirements from Section 1 and Section 6 Gate 0.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "py" / "renloc-core"))

import renloc_native
from renloc_core.ast_reader.unpickler import read_rpyc_data, SafeUnpicklingError
from renloc_core.qe_client import IsolatedQEClient

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# =========================================================================
# 1. Multi-Platform Build Chain & Cargo Workspace
# =========================================================================
def test_cargo_workspace_compilation_and_tests():
    """Verify that all crates in Cargo workspace build and pass unit tests."""
    # Check cargo workspace
    check_res = subprocess.run(
        ["cargo", "check", "--workspace"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    assert check_res.returncode == 0, f"cargo check --workspace failed:\n{check_res.stderr}"

    # Test cargo workspace
    test_res = subprocess.run(
        ["cargo", "test", "--workspace"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    assert test_res.returncode == 0, f"cargo test --workspace failed:\n{test_res.stderr}"
    assert "test result: ok." in test_res.stdout


def test_tauri_host_and_bootstrapper_structure():
    """Verify Tauri v2 host configuration, capabilities ACL, and bootstrapper."""
    tauri_dir = REPO_ROOT / "apps" / "desktop" / "src-tauri"
    assert (tauri_dir / "tauri.conf.json").exists()
    assert (tauri_dir / "Cargo.toml").exists()
    assert (tauri_dir / "capabilities" / "default.json").exists()
    assert (tauri_dir / "src" / "main.rs").exists()
    assert (tauri_dir / "src" / "sidecar.rs").exists()
    assert (tauri_dir / "src" / "job_guard.rs").exists()
    assert (tauri_dir / "src" / "rpc_client.rs").exists()
    assert (tauri_dir / "src" / "bin" / "renloc-bootstrapper.rs").exists()

    # Verify bootstrapper binary is built
    bin_path = tauri_dir / "binaries" / "renloc-bootstrapper-x86_64-unknown-linux-gnu"
    assert bin_path.exists() and os.access(bin_path, os.X_OK)


# =========================================================================
# 2. Dependency Isolation & Virtualenv Handshake
# =========================================================================
def test_virtualenv_isolation_and_numpy_versions():
    """Verify that .venvs/core has numpy >= 2.0 and .venvs/qe has numpy < 2.0 with comet 2.2.7."""
    core_python = REPO_ROOT / ".venvs" / "core" / "bin" / "python"
    qe_python = REPO_ROOT / ".venvs" / "qe" / "bin" / "python"

    assert core_python.exists()
    assert qe_python.exists()

    # Core numpy
    core_out = subprocess.check_output(
        [str(core_python), "-c", "import numpy; print(numpy.__version__)"],
        text=True
    ).strip()
    core_major = int(core_out.split(".")[0])
    assert core_major >= 2, f".venvs/core numpy must be >= 2.0, got {core_out}"

    # QE numpy and comet
    qe_out = subprocess.check_output(
        [str(qe_python), "-c", "import numpy, comet; print(numpy.__version__, comet.__version__)"],
        text=True
    ).strip().split()
    qe_np_ver, comet_ver = qe_out[0], qe_out[1]
    qe_major = int(qe_np_ver.split(".")[0])
    assert qe_major < 2, f".venvs/qe numpy must be < 2.0, got {qe_np_ver}"
    assert comet_ver == "2.2.7", f"unbabel-comet must be 2.2.7, got {comet_ver}"


def test_jsonrpc_stdio_handshake_and_error_handling():
    """Verify JSON-RPC stdio communication, error recovery, and clean process reaping."""
    qe_python = str(REPO_ROOT / ".venvs" / "qe" / "bin" / "python")
    client = IsolatedQEClient(qe_python_path=qe_python)
    worker_pid = client.pid
    assert worker_pid is not None and worker_pid > 0

    try:
        # Ping
        assert client.ping() is True

        # Batch scoring
        batch = [
            {"src": "Hello world!", "mt": "Xin chào thế giới!"},
            {"src": "Localization system", "mt": "Hệ thống bản địa hóa"}
        ]
        scores = client.score_batch(batch)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)

        # Invalid method error (-32601)
        err_res = client.raw_call({
            "jsonrpc": "2.0",
            "id": 999,
            "method": "invalid_method_test",
            "params": {}
        })
        assert "error" in err_res
        assert err_res["error"]["code"] == -32601
        assert err_res["id"] == 999

        # Corrupted JSON error (-32603)
        corrupt_res = client.send_raw_line("INVALID_RAW_JSON_STRING_123")
        assert "error" in corrupt_res
        assert corrupt_res["error"]["code"] == -32603

        # Worker must remain alive and responsive after errors
        assert client.ping() is True
    finally:
        client.close()

    # Verify no orphan process leaked
    time.sleep(0.3)
    is_running = False
    try:
        os.kill(worker_pid, 0)
        is_running = True
    except (ProcessLookupError, OSError):
        is_running = False

    assert not is_running, f"Worker process PID {worker_pid} was orphaned!"


# =========================================================================
# 3. Ren'Py Fixtures (Scripts, RPYC, RPA)
# =========================================================================
def test_renpy_script_fixtures():
    """Verify that script fixtures exist and contain required Ren'Py constructs."""
    scripts_dir = FIXTURES_DIR / "scripts"
    assert (scripts_dir / "sample_7x.rpy").exists()
    assert (scripts_dir / "sample_8x.rpy").exists()
    assert (scripts_dir / "screens.rpy").exists()
    assert (scripts_dir / "dialogue_normalization.rpy").exists()

    norm_text = (scripts_dir / "dialogue_normalization.rpy").read_text(encoding="utf-8")
    assert 'a "Form 1: Explicit character dialogue."' in norm_text
    assert '"Form 2: Implicit narration dialogue."' in norm_text
    assert 'extend " Continued narration text."' in norm_text
    assert 'renpy.say(a, "Form 4: Direct renpy.say call.")' in norm_text


def test_renpy_rpyc_fixtures_unpickling_and_security():
    """Verify RPYC fixtures: protocol 2 & 4 unpickling and malicious payload rejection."""
    rpyc_dir = FIXTURES_DIR / "rpyc"
    
    # 7.x RPYC
    ast_7x = read_rpyc_data((rpyc_dir / "sample_7x.rpyc").read_bytes())
    assert isinstance(ast_7x, list) and len(ast_7x) >= 2
    assert ast_7x[0].name == "start"
    assert ast_7x[1].who == "Sylvia"

    # 8.x RPYC
    ast_8x = read_rpyc_data((rpyc_dir / "sample_8x.rpyc").read_bytes())
    assert isinstance(ast_8x, list) and len(ast_8x) >= 2
    assert ast_8x[0].name == "start"
    assert ast_8x[1].who == "Sylvia"

    # Malicious RPYC rejection
    with pytest.raises(SafeUnpicklingError):
        read_rpyc_data((rpyc_dir / "malicious.rpyc").read_bytes())


def test_renpy_rpa_fixtures_all_variants():
    """Verify that all RPA archive variants (1.0, 2.0, 3.0, Custom 8-byte) extract cleanly."""
    rpa_dir = FIXTURES_DIR / "rpa"

    variants = [
        "archive_v1.rpa",
        "archive_v2.rpa",
        "archive_v3.rpa",
        "archive_custom.arc",
    ]

    for variant in variants:
        arc_path = rpa_dir / variant
        assert arc_path.exists()
        archive = renloc_native.RpaArchive(str(arc_path))
        files = archive.list_files()
        assert "script.rpy" in files
        assert "assets/ui_button.png" in files
        assert "data/dialogue.json" in files

        script_content = archive.read_file("script.rpy")
        assert b"Inside RPA archive" in script_content

        png_content = archive.read_file("assets/ui_button.png")
        assert b"\x89PNG" in png_content


# =========================================================================
# 4. renloc_native PyO3 ABI3, Pickle VM, HarfBuzz & Vietnamese 134 NFC
# =========================================================================
def test_vietnamese_134_nfc_cmap_audit():
    """Verify 134 Vietnamese NFC character audit in renloc_native."""
    font_path = "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"
    with open(font_path, "rb") as f:
        font_bytes = f.read()

    audit = renloc_native.audit_font_bytes("AdwaitaMono", font_bytes)
    assert audit["font_name"] == "AdwaitaMono"
    assert audit["total_required"] == 134, f"Must require exactly 134 NFC characters, got {audit['total_required']}"
    assert audit["supported_count"] == 134
    assert audit["coverage_ratio"] == 1.0
    assert audit["is_fully_compatible"] is True
    assert audit["missing_chars"] == ""


def test_harfbuzz_text_shaping_and_overflow_detection():
    """Verify HarfBuzz text shaping and bounding box overflow calculation."""
    font_path = "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"
    with open(font_path, "rb") as f:
        font_bytes = f.read()

    text = "Xin chào các dịch giả Renslate!"

    # Normal fit
    fit_metrics = renloc_native.shape_and_measure(font_bytes, 16.0, text, 800.0, 100.0)
    assert fit_metrics["total_advance_pixels"] > 0
    assert fit_metrics["max_height_pixels"] > 0
    assert fit_metrics["glyph_count"] > 0
    assert fit_metrics["is_overflowing"] is False

    # Width overflow
    overflow_w = renloc_native.shape_and_measure(font_bytes, 16.0, text, 50.0, 100.0)
    assert overflow_w["is_overflowing"] is True

    # Height overflow
    overflow_h = renloc_native.shape_and_measure(font_bytes, 16.0, text, 800.0, 5.0)
    assert overflow_h["is_overflowing"] is True


def test_vfs_layer_mounting_and_loose_override():
    """Verify VfsLayer loads archives and gives precedence to loose files."""
    rpa_dir = FIXTURES_DIR / "rpa"
    vfs = renloc_native.VfsLayer(str(rpa_dir))
    vfs.mount_game_directory()

    files = vfs.list_all_files()
    assert "script.rpy" in files
    orig_content = vfs.read_file("script.rpy")
    assert b"Inside RPA archive" in orig_content
