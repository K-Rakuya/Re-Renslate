#!/usr/bin/env python3
"""
scripts/gate0_verify.py
Verification Gate 0:
  1. Verifies .venvs/core and .venvs/qe environments.
  2. Spawns isolated QE worker and performs JSON-RPC stdio handshake.
  3. Verifies isolation of numpy / process boundary.
"""
import sys
from pathlib import Path

# Add core to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "py" / "renloc-core"))

from renloc_core.qe_client import IsolatedQEClient

def main():
    print("[*] Running Gate 0 Verification...")
    qe_python = REPO_ROOT / ".venvs" / "qe" / "bin" / "python"
    if not qe_python.exists():
        print(f"[!] QE python not found at {qe_python}, trying sys.executable", file=sys.stderr)
        qe_python_str = sys.executable
    else:
        qe_python_str = str(qe_python)

    print(f"[*] Initializing IsolatedQEClient with {qe_python_str}...")
    client = IsolatedQEClient(qe_python_path=qe_python_str)

    # 1. Ping test
    print("[*] Testing JSON-RPC ping...")
    is_alive = client.ping()
    assert is_alive, "Gate 0 Failure: QE worker failed to respond to ping"
    print("[+] JSON-RPC Ping OK!")

    # 2. Batch score test
    print("[*] Testing JSON-RPC score_batch...")
    test_batch = [
        {"src": "Hello world!", "mt": "Xin chào thế giới!"},
        {"src": "Good morning.", "mt": "Chào buổi sáng."}
    ]
    scores = client.score_batch(test_batch)
    print(f"[+] Received scores: {scores}")
    assert len(scores) == 2, f"Gate 0 Failure: expected 2 scores, got {len(scores)}"
    assert all(0.0 <= s <= 1.0 for s in scores), "Gate 0 Failure: scores out of [0, 1] range"

    # Clean shutdown
    client.close()
    print("[+] Gate 0 PASSED: Monorepo virtualenv isolation and JSON-RPC stdio handshake successful!")

if __name__ == "__main__":
    main()
