"""
renloc_core/qe_client.py
JSON-RPC stdio client communicating with the isolated .venvs/qe process.
Prevents numpy ABI conflicts by communicating exclusively via JSON-RPC over stdio.
"""
import sys
import json
import subprocess
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

class IsolatedQEClient:
    def __init__(self, qe_python_path: Optional[str] = None):
        if qe_python_path is None:
            # Auto-detect .venvs/qe/bin/python
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            candidate = repo_root / ".venvs" / "qe" / "bin" / "python"
            if candidate.exists():
                qe_python_path = str(candidate)
            else:
                qe_python_path = sys.executable

        self.qe_python_path = qe_python_path
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._req_counter = 0
        self._start_worker()

    def _start_worker(self):
        cmd = [self.qe_python_path, "-m", "renloc_qe.worker"]
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        qe_dir = str(repo_root / "py" / "renloc-qe")
        env = dict(subprocess.os.environ)
        # Ensure renloc-qe is in PYTHONPATH for the isolated process
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{qe_dir}:{existing_pythonpath}" if existing_pythonpath else qe_dir

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )

    def ping(self) -> bool:
        res = self._call("ping", {})
        return res == "pong"

    def score_batch(self, items: List[Dict[str, str]]) -> List[float]:
        res = self._call("score_batch", {"items": items})
        if isinstance(res, dict) and "scores" in res:
            return res["scores"]
        return []

    def _call(self, method: str, params: Dict[str, Any]) -> Any:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start_worker()

            self._req_counter += 1
            req_id = self._req_counter
            req = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params
            }
            line = json.dumps(req) + "\n"
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            self._proc.stdin.write(line)
            self._proc.stdin.flush()

            resp_line = self._proc.stdout.readline()
            if not resp_line:
                raise RuntimeError("QE worker terminated unexpectedly or returned EOF.")
            resp = json.loads(resp_line.strip())
            if "error" in resp:
                raise RuntimeError(f"QE worker error: {resp['error']}")
            return resp.get("result")

    def close(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    req = {"jsonrpc": "2.0", "id": 999999, "method": "shutdown", "params": {}}
                    self._proc.stdin.write(json.dumps(req) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=2.0)
                except Exception:
                    self._proc.kill()
            self._proc = None
