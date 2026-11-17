#!/usr/bin/env python3
"""
scripts/setup_envs.py
Sets up isolated Python virtual environments for Renslate:
  - .venvs/core: Python 3.12 with modern numpy (>= 2.0) for AST reader, Static Graph, APSW SQLite writer, and LLM orchestration.
  - .venvs/qe: Python 3.12 with legacy numpy (< 2.0, e.g. 1.26.4) for unbabel-comet == 2.2.7 isolation.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENVS_DIR = REPO_ROOT / ".venvs"
CORE_VENV = VENVS_DIR / "core"
QE_VENV = VENVS_DIR / "qe"

def find_uv() -> str:
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path
    local_uv = Path.home() / ".local" / "bin" / "uv"
    if local_uv.exists():
        return str(local_uv)
    cargo_uv = Path.home() / ".cargo" / "bin" / "uv"
    if cargo_uv.exists():
        return str(cargo_uv)
    raise RuntimeError("uv not found. Please install uv (https://github.com/astral-sh/uv).")

def setup_virtualenvs():
    uv = find_uv()
    print(f"[*] Using uv at: {uv}")
    VENVS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Setup .venvs/core with Python 3.12
    print("[*] Creating .venvs/core with Python 3.12...")
    subprocess.run([uv, "venv", str(CORE_VENV), "--python", "3.12", "--seed"], check=True)
    core_pip = str(CORE_VENV / "bin" / "pip")
    core_python = str(CORE_VENV / "bin" / "python")
    
    # 2. Setup .venvs/qe with Python 3.12
    print("[*] Creating .venvs/qe with Python 3.12...")
    subprocess.run([uv, "venv", str(QE_VENV), "--python", "3.12", "--seed"], check=True)
    qe_pip = str(QE_VENV / "bin" / "pip")
    qe_python = str(QE_VENV / "bin" / "python")

    print("[+] Virtualenvs initialized successfully:")
    print(f"    - Core: {CORE_VENV}")
    print(f"    - QE:   {QE_VENV}")

if __name__ == "__main__":
    setup_virtualenvs()
