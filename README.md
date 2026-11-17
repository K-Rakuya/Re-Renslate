# Renslate (renloc)

Autonomous Local-First Ren'Py Localization System.

## Architecture

1. **Rust Core Crates**:
   - `crates/renloc-rpa`: High-performance binary archive reader (RPA-1.0/2.0/3.0/3.2/4.0/ALT/ZiX) with zero-copy memory mapping (`memmap2`) and safe Rust inert pickle opcode virtual machine.
   - `crates/renloc-shaper`: OpenType text shaping, FriBidi bidirectional analysis, 134-glyph Vietnamese NFC cmap audit, and bounding-box UI overflow detection.
   - `crates/renloc-db`: High-throughput read-only connection pooling with `rusqlite`, WAL mode, FTS5 full-text search, and `sqlite-vec` auto-extension.
   - `crates/renloc-pyext`: PyO3 ABI3 Python extension module exposing native RPA and shaper capabilities to the Python sidecar.

2. **Python Services**:
   - `py/renloc-core`: Main application sidecar running with Python 3.12 (`numpy >= 2.0`). Houses the inert Ren'Py AST unpickler (`NativeRenpyUnpickler`), static ICFG & temporal honorific graph analyzer, single-writer SQLite manager (`apsw`), hybrid LLM constraint compiler, and package synthesizer.
   - `py/renloc-qe`: Isolated worker running in a dedicated virtualenv (`numpy 1.26.x`, `unbabel-comet==2.2.7`) communicating via JSON-RPC stdio.

3. **Desktop Host (Tauri v2 + React)**:
   - `apps/desktop/src-tauri`: Windows Job Object / POSIX death-pipe process supervisor, UDS/Named Pipe MessagePack RPC with 256-bit CSPRNG authentication, and capabilities ACL.
   - `apps/desktop/src`: Virtualized 300,000-row translation grid with keyset pagination.

4. **In-Game Runtime Hook**:
   - `runtime/renpy_hook/renloc_runtime_hook.rpy`: Revertable state containers, idempotent dialogue filter, and 90-frame storm circuit breaker.
