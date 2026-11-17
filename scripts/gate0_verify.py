#!/usr/bin/env python3
"""
scripts/gate0_verify.py
Verification Gate 0:
  1. Verifies .venvs/core and .venvs/qe Python and NumPy versions (numpy >= 2.0 vs numpy < 2.0).
  2. Spawns isolated QE worker and performs JSON-RPC stdio handshake & ping.
  3. Verifies batch scoring.
  4. Verifies JSON-RPC error handling: method not found (-32601) and corrupted JSON (-32603).
  5. Verifies graceful shutdown and absence of orphan processes.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

# Add core to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "py" / "renloc-core"))

from renloc_core.qe_client import IsolatedQEClient

def check_environments():
    print("[*] Checking Virtualenv Isolation & NumPy versions...")
    core_python = REPO_ROOT / ".venvs" / "core" / "bin" / "python"
    qe_python = REPO_ROOT / ".venvs" / "qe" / "bin" / "python"

    assert core_python.exists(), f"Core Python not found at {core_python}"
    assert qe_python.exists(), f"QE Python not found at {qe_python}"

    # Check Core numpy version >= 2.0.0
    core_np_out = subprocess.check_output(
        [str(core_python), "-c", "import numpy; print(numpy.__version__)"],
        text=True
    ).strip()
    print(f"    - .venvs/core numpy version: {core_np_out}")
    core_major = int(core_np_out.split(".")[0])
    assert core_major >= 2, f"Core NumPy must be >= 2.0.0, got {core_np_out}"

    # Check QE numpy version < 2.0.0
    qe_np_out = subprocess.check_output(
        [str(qe_python), "-c", "import numpy; print(numpy.__version__)"],
        text=True
    ).strip()
    print(f"    - .venvs/qe numpy version:   {qe_np_out}")
    qe_major = int(qe_np_out.split(".")[0])
    assert qe_major < 2, f"QE NumPy must be < 2.0.0, got {qe_np_out}"
    print("[+] Virtualenv isolation and NumPy matrix verified successfully!\n")

def test_jsonrpc_protocol():
    print("[*] Testing JSON-RPC stdio protocol between Core and Worker QE...")
    qe_python = str(REPO_ROOT / ".venvs" / "qe" / "bin" / "python")
    client = IsolatedQEClient(qe_python_path=qe_python)
    worker_pid = client.pid
    print(f"    - Worker QE spawned with PID: {worker_pid}")
    assert worker_pid is not None and worker_pid > 0

    # 1. Ping test
    print("[*] 1. Testing JSON-RPC ping...")
    is_alive = client.ping()
    assert is_alive, "Gate 0 Failure: QE worker failed to respond to ping"
    print("    [+] JSON-RPC Ping OK!")

    # 2. Batch score test
    print("[*] 2. Testing JSON-RPC score_batch...")
    test_batch = [
        {"src": "Hello world!", "mt": "Xin chào thế giới!"},
        {"src": "Good morning.", "mt": "Chào buổi sáng."}
    ]
    scores = client.score_batch(test_batch)
    print(f"    [+] Received scores: {scores}")
    assert len(scores) == 2, f"Gate 0 Failure: expected 2 scores, got {len(scores)}"
    assert all(0.0 <= s <= 1.0 for s in scores), "Gate 0 Failure: scores out of [0, 1] range"

    # 3. Method not found error (-32601)
    print("[*] 3. Testing JSON-RPC method not found (-32601)...")
    err_resp = client.raw_call({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "non_existent_method_xyz",
        "params": {}
    })
    print(f"    [+] Received error response: {err_resp}")
    assert "error" in err_resp, "Expected error in response"
    assert err_resp["error"]["code"] == -32601, f"Expected error code -32601, got {err_resp['error']['code']}"
    assert err_resp["id"] == 101

    # 4. Corrupted / malformed JSON error (-32603)
    print("[*] 4. Testing JSON-RPC corrupted JSON handling (-32603)...")
    corrupt_resp = client.send_raw_line("{{malformed_json_syntax_without_quotes")
    print(f"    [+] Received corrupt JSON response: {corrupt_resp}")
    assert "error" in corrupt_resp, "Expected error for malformed JSON"
    assert corrupt_resp["error"]["code"] == -32603, f"Expected error code -32603, got {corrupt_resp['error']['code']}"

    # Worker should still be alive and responsive after error handling
    assert client.ping(), "Worker died after error handling!"
    print("    [+] Worker survived invalid method and malformed JSON errors!")

    # 5. Graceful shutdown
    print("[*] 5. Testing graceful shutdown...")
    client.close()
    time.sleep(0.3)

    # 6. Orphan process verification
    print("[*] 6. Verifying no orphan process remains...")
    is_running = False
    try:
        os.kill(worker_pid, 0)
        is_running = True
    except (ProcessLookupError, OSError):
        is_running = False

    assert not is_running, f"Gate 0 Failure: Worker process PID {worker_pid} is still running (orphan leaked)!"
    print(f"    [+] Worker PID {worker_pid} exited cleanly without orphan process leak!\n")

def main():
    print("=================================================================")
    print("          RENSLATE GATE 0 AUDIT VERIFICATION SUITE               ")
    print("=================================================================")
    check_environments()
    test_jsonrpc_protocol()
    print("=================================================================")
    print(" [✓] GATE 0 INDEPENDENT QUALITY AUDIT PASSED 100%               ")
    print("=================================================================")

if __name__ == "__main__":
    main()
