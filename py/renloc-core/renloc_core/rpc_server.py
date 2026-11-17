"""
rpc_server.py
MessagePack RPC Server for Renslate Python Sidecar (renloc-core).
Communicates with the Tauri v2 desktop host (apps/desktop/src-tauri).

Key Features:
  1. Length-prefixed MessagePack protocol (4-byte little-endian length + rmp payload).
  2. 256-bit CSPRNG token authentication via constant-time comparison (secrets.compare_digest).
  3. Unix Domain Socket (UDS) / Windows Named Pipe / Localhost socket transport.
  4. POSIX death-pipe listener for zero-orphan lifecycle enforcement.
  5. High-level endpoints for AST scanning, database querying, font auditing, and package export.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import msgpack

from renloc_core.synthesizer import (
    FontAuditor,
    GuiInspector,
    DialogueShaper,
    IncrementalMatcher,
    PackageSynthesizer,
    TranslationItem,
    UIStringTranslationItem,
)

logger = logging.getLogger("renloc_core.rpc_server")


class RpcServer:
    """
    MessagePack RPC Server serving the Tauri desktop application.
    """

    def __init__(
        self,
        endpoint_path: str | Path,
        auth_token: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.endpoint_path = Path(endpoint_path)
        self.auth_token = auth_token or os.environ.get("RENSLATE_AUTH_TOKEN", "")
        self.loop = loop
        self.server: Optional[asyncio.AbstractServer] = None
        self.is_running: bool = False
        self._handlers: Dict[str, Callable[[Any], Any]] = {}
        self._cached_adapter: Optional[Any] = None
        self._cached_adapter_key: Optional[str] = None
        self._translation_progress: Dict[str, Any] = {
            "is_running": False,
            "total": 0,
            "current": 0,
            "current_source": "",
            "current_target": "",
            "status": "idle",
            "error": None,
        }
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        self.register_handler("ping", self._handle_ping)
        self.register_handler("echo", self._handle_echo)
        self.register_handler("get_status", self._handle_get_status)
        self.register_handler("audit_fonts", self._handle_audit_fonts)
        self.register_handler("measure_dialogue", self._handle_measure_dialogue)
        self.register_handler("scan_project", self._handle_scan_project)
        self.register_handler("audit_project_fonts", self._handle_audit_project_fonts)
        self.register_handler("get_system_specs", self._handle_get_system_specs)
        self.register_handler("assess_model", self._handle_assess_model)
        self.register_handler("test_inference_connection", self._handle_test_inference_connection)
        self.register_handler("get_translation_progress", self._handle_get_translation_progress)
        self.register_handler("unload_model", self._handle_unload_model)
        self.register_handler("start_translation", self._handle_start_translation)
        self.register_handler("export_package", self._handle_export_package)
        self.register_handler("launch_game", self._handle_launch_game)
        self.register_handler("shutdown", self._handle_shutdown)

    def register_handler(self, method: str, handler: Callable[[Any], Any]) -> None:
        self._handlers[method] = handler

    def _handle_ping(self, _payload: Any) -> Dict[str, Any]:
        return {"status": "pong", "version": "0.1.0"}

    def _handle_echo(self, payload: Any) -> Any:
        return payload

    def _handle_get_status(self, _payload: Any) -> Dict[str, Any]:
        return {
            "status": "ready",
            "sidecar_pid": os.getpid(),
            "python_version": sys.version,
            "platform": sys.platform,
        }

    def _handle_audit_fonts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        font_path = payload.get("font_path")
        if not font_path or not Path(font_path).exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        audit = FontAuditor.audit_font_file(font_path)
        return audit.to_dict()

    def _handle_measure_dialogue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        font_path = payload.get("font_path")
        text = payload.get("text", "")
        max_width = payload.get("max_width", 744.0)
        font_size = payload.get("font_size", 22.0)

        if not font_path or not Path(font_path).exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")

        font_bytes = Path(font_path).read_bytes()
        shaper = DialogueShaper(font_bytes=font_bytes, default_font_size=font_size)
        return shaper.measure_text(text=text, font_size=font_size, max_box_width=max_width)

    def _handle_scan_project(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Scans game directory, extracts RPA/RPYC dialogues into SQLite."""
        game_dir_str = payload.get("game_dir")
        if not game_dir_str:
            raise ValueError("Missing 'game_dir' parameter")

        game_dir = Path(game_dir_str).resolve()
        if not game_dir.exists():
            raise FileNotFoundError(f"Game directory does not exist: {game_dir}")

        db_path = Path(payload.get("db_path") or (game_dir / "renslate_project.sqlite")).resolve()

        from renloc_core.db_writer import DbWriter
        from renloc_core.db_writer.schema import init_schema
        from renloc_core.static_graph.dialogue_normalizer import DialogueNormalizer
        from renloc_core.ast_reader.unpickler import read_rpyc_dialogues

        try:
            import renloc_native
            vfs = renloc_native.VfsLayer(str(game_dir))
            vfs.mount_game_directory()
            files = vfs.list_all_files()
        except ImportError:
            files = [str(p.relative_to(game_dir)) for p in game_dir.rglob("*") if p.is_file()]

        normalizer = DialogueNormalizer()
        writer = DbWriter(db_path)
        init_schema(writer.raw_connection)

        # Exclude internal renpy/common engine modules to focus strictly on game scripts
        rpyc_files = [
            f for f in files
            if (f.endswith(".rpyc") or f.endswith(".rpymc"))
            and not f.startswith("renpy/")
            and not f.startswith("common/")
        ]
        extracted_count = 0

        with writer.write_transaction() as cursor:
            for rpyc_rel in rpyc_files:
                try:
                    try:
                        import renloc_native
                        data = vfs.read_file(rpyc_rel)
                    except Exception:
                        data = (game_dir / rpyc_rel).read_bytes()

                    dialogues = read_rpyc_dialogues(data, normalizer=normalizer, default_filename=rpyc_rel)
                    for d in dialogues:
                        extracted_count += 1
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO translation_units
                            (unit_identifier, source_text, speaker, status, row_order, file_path, line_number)
                            VALUES (?, ?, ?, 'PENDING', ?, ?, ?)
                            """,
                            (
                                d.translation_id,
                                d.what,
                                d.who,
                                extracted_count,
                                d.filename or rpyc_rel,
                                d.linenumber or 1,
                            ),
                        )
                except Exception as e:
                    logger.debug("Failed to extract %s: %s", rpyc_rel, e)
                    continue

        return {
            "total_extracted": extracted_count,
            "db_path": str(db_path),
            "files_scanned": len(files),
            "rpyc_count": len(rpyc_files),
        }

    def _handle_audit_project_fonts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Audits all fonts in project directory for 134 Vietnamese NFC glyphs."""
        game_dir_str = payload.get("game_dir")
        if not game_dir_str:
            raise ValueError("Missing 'game_dir' parameter")
        game_dir = Path(game_dir_str).resolve()

        font_files = list(game_dir.rglob("*.ttf")) + list(game_dir.rglob("*.otf"))
        results = []
        is_all_compatible = True

        for ff in font_files:
            try:
                audit = FontAuditor.audit_font_file(ff)
                results.append(audit.to_dict())
                if not audit.is_fully_compatible:
                    is_all_compatible = False
            except Exception as e:
                results.append({"font_name": ff.name, "error": str(e), "is_fully_compatible": False})

        fallback = payload.get("fallback_font") or "game/tl/vietnamese/fonts/fallback.ttf"
        dsl = FontAuditor.generate_font_group_dsl(
            group_name="vietnamese_font_group",
            primary_font_path="game/fonts/original.ttf",
            fallback_font_path=fallback,
        )

        return {
            "fonts": results,
            "is_all_compatible": is_all_compatible,
            "fontgroup_dsl": dsl,
            "total_fonts_checked": len(font_files),
        }

    def _handle_get_system_specs(self, _payload: Any) -> Dict[str, Any]:
        """Returns real-time GPU/RAM specs and detects available local GGUF models."""
        from renloc_core.llm_engine.adapters.llamacpp_adapter import get_system_hardware_info
        specs = get_system_hardware_info()

        candidate_dirs = [
            Path.home() / "models",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.cwd() / "models",
        ]
        if os.environ.get("RENSLATE_MODELS_DIR"):
            candidate_dirs.insert(0, Path(os.environ["RENSLATE_MODELS_DIR"]))
        found_models: List[Dict[str, Any]] = []
        for cdir in candidate_dirs:
            if cdir.exists() and cdir.is_dir():
                for p in cdir.glob("*.gguf"):
                    found_models.append({
                        "path": str(p),
                        "name": p.name,
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    })

        specs["detected_models"] = found_models
        return specs

    def _handle_assess_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Assesses OOM risk and calculates safe layer allocation for a GGUF model."""
        model_path = payload.get("model_path")
        if not model_path:
            raise ValueError("Missing 'model_path' parameter")
        from renloc_core.llm_engine.adapters.llamacpp_adapter import assess_model_memory_risk
        return assess_model_memory_risk(model_path)

    def _handle_test_inference_connection(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Tests connectivity to an external OpenAI-compatible inference server (e.g. llama-server)."""
        server_url = (payload.get("server_url") or "").strip()
        if not server_url:
            return {
                "online": False,
                "message": "Chưa nhập địa chỉ URL máy chủ (server_url).",
                "models": [],
            }

        import json
        import urllib.request
        import urllib.error

        clean = server_url.rstrip("/")
        if clean.endswith("/chat/completions"):
            base_url = clean[:-len("/chat/completions")]
        elif clean.endswith("/v1"):
            base_url = clean
        else:
            base_url = f"{clean}/v1"

        models_url = f"{base_url}/models"
        health_url = f"{clean}/health"

        # Try models endpoint first
        try:
            req = urllib.request.Request(models_url, headers={"User-Agent": "Renslate-Desktop"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = []
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    models = [m.get("id", "") for m in data["data"] if isinstance(m, dict) and m.get("id")]
                return {
                    "online": True,
                    "message": f"Kết nối máy chủ thành công ({models_url})",
                    "models": models,
                }
        except Exception:
            pass

        # Try health endpoint fallback
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "Renslate-Desktop"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                return {
                    "online": True,
                    "message": f"Kết nối máy chủ thành công ({health_url})",
                    "models": [],
                }
        except Exception as e:
            return {
                "online": False,
                "message": f"Không thể kết nối tới {server_url}: {e}",
                "models": [],
            }

    def _handle_get_translation_progress(self, _payload: Any) -> Dict[str, Any]:
        """Returns current translation execution progress."""
        return self._translation_progress

    def _handle_unload_model(self, _payload: Any) -> Dict[str, Any]:
        """Unloads cached GGUF model to completely free VRAM and RAM."""
        if self._cached_adapter is not None:
            self._cached_adapter.unload()
            self._cached_adapter = None
            self._cached_adapter_key = None
        return {"status": "unloaded"}

    async def _handle_start_translation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Translates pending dialogue units using GGUF / local model with OOM guard."""
        db_path_str = payload.get("db_path")
        if not db_path_str:
            raise ValueError("Missing 'db_path' parameter")
        db_path = Path(db_path_str).resolve()

        engine = payload.get("engine", "llama.cpp")
        model_path = payload.get("model_path")
        server_url = payload.get("server_url")
        n_gpu_layers = payload.get("n_gpu_layers", -1)
        n_ctx = int(payload.get("n_ctx", 2048))
        temperature = float(payload.get("temperature", 0.2))
        limit = int(payload.get("limit", 50))

        from renloc_core.db_writer import DbWriter
        writer = DbWriter(db_path)

        from renloc_core.llm_engine.adapters.llamacpp_adapter import (
            LlamaCppAdapter,
            assess_model_memory_risk,
        )

        adapter: Optional[LlamaCppAdapter] = None
        if engine == "llama.cpp":
            if server_url:
                adapter = LlamaCppAdapter(server_url=server_url)
            elif model_path:
                mp = Path(model_path).expanduser().resolve()
                if not mp.exists():
                    raise FileNotFoundError(f"File GGUF model không tồn tại: {model_path}")
                risk = assess_model_memory_risk(mp)
                if not risk["can_fit"] and risk["status"] == "oom_risk":
                    raise MemoryError(risk["message"])

                adapter_key = f"{str(mp)}:{n_gpu_layers}:{n_ctx}"
                if self._cached_adapter is not None and self._cached_adapter_key == adapter_key:
                    adapter = self._cached_adapter
                else:
                    if self._cached_adapter is not None:
                        self._cached_adapter.unload()
                    adapter = LlamaCppAdapter(
                        model_path=str(mp),
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                    )
                    self._cached_adapter = adapter
                    self._cached_adapter_key = adapter_key
            else:
                raise ValueError(
                    "Động cơ llama.cpp yêu cầu cung cấp đường dẫn tệp model GGUF (model_path) "
                    "hoặc địa chỉ máy chủ suy luận (server_url)."
                )
        elif engine == "mock":
            adapter = None
        else:
            raise NotImplementedError(f"Động cơ suy luận '{engine}' chưa được cấu hình backend.")

        translated = 0
        self._translation_progress = {
            "is_running": True,
            "total": 0,
            "current": 0,
            "current_source": "",
            "current_target": "",
            "status": "translating",
            "error": None,
        }

        try:
            with writer.write_transaction() as cursor:
                rows = cursor.execute(
                    """
                    SELECT id, unit_identifier, source_text, speaker
                    FROM translation_units
                    WHERE status = 'PENDING' OR target_text IS NULL
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()

                self._translation_progress["total"] = len(rows)

                for idx, row in enumerate(rows):
                    uid, unit_id, src_text, who = row
                    self._translation_progress["current"] = idx + 1
                    self._translation_progress["current_source"] = src_text

                    if adapter is not None:
                        gen_res = await asyncio.to_thread(
                            adapter.translate_dialogue,
                            source_text=src_text,
                            speaker=who,
                            temperature=temperature,
                        )
                        trans_text = gen_res.text
                    elif engine == "mock":
                        trans_text = f"[Mock Test] {src_text}"
                    else:
                        raise RuntimeError("Không có backend suy luận khả dụng.")

                    self._translation_progress["current_target"] = trans_text
                    cursor.execute(
                        """
                        UPDATE translation_units
                        SET target_text = ?, status = 'COMMITTED', updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (trans_text, uid)
                    )
                    translated += 1
                    await asyncio.sleep(0.001)

            self._translation_progress["status"] = "completed"
            self._translation_progress["is_running"] = False
        except Exception as e:
            self._translation_progress["status"] = "error"
            self._translation_progress["is_running"] = False
            self._translation_progress["error"] = str(e)
            raise e

        return {
            "translated_count": translated,
            "status": "completed",
        }

    def _handle_export_package(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Exports translation package into game/tl/vietnamese/."""
        game_dir_str = payload.get("game_dir")
        if not game_dir_str:
            raise ValueError("Missing 'game_dir' parameter")
        game_dir = Path(game_dir_str).resolve()
        db_path = Path(payload.get("db_path") or (game_dir / "renslate_project.sqlite")).resolve()

        from renloc_core.db_writer import DbWriter
        writer = DbWriter(db_path)
        dialogue_items = []

        with writer.read_transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT unit_identifier, source_text, target_text, speaker, file_path, line_number
                FROM translation_units
                WHERE target_text IS NOT NULL
                ORDER BY row_order ASC
                """
            ).fetchall()

            for r in rows:
                tlid, src, tgt, who, fpath, line = r
                dialogue_items.append(
                    TranslationItem(
                        translation_id=tlid,
                        original_statement_code=f'{who} "{src}"' if who else f'"{src}"',
                        translated_text=tgt,
                        who=who,
                        filename=fpath or "script.rpy",
                        linenumber=line or 1,
                    )
                )

        output_dir = Path(payload.get("output_dir") or (game_dir / "game")).resolve()
        synthesizer = PackageSynthesizer(language=payload.get("language", "vietnamese"))
        created = synthesizer.export_package(output_dir=output_dir, dialogue_items=dialogue_items, string_items=[])

        # Deploy runtime hook
        hook_src = Path(__file__).resolve().parent.parent.parent / "runtime" / "renpy_hook" / "renloc_runtime_hook.rpy"
        if hook_src.exists():
            target_hook = output_dir / "tl" / synthesizer.language / "renloc_runtime_hook.rpy"
            target_hook.write_text(hook_src.read_text(encoding="utf-8"), encoding="utf-8")
            created["renloc_runtime_hook.rpy"] = target_hook

        return {
            "exported_files": list(created.keys()),
            "output_dir": str(output_dir / "tl" / synthesizer.language),
            "total_items": len(dialogue_items),
        }

    def _handle_launch_game(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Detects Ren'Py game launcher executable and starts it."""
        import subprocess
        game_dir_str = payload.get("game_dir")
        if not game_dir_str:
            raise ValueError("Missing 'game_dir' parameter")
        game_dir = Path(game_dir_str).resolve()

        # Look for executable (.sh, .exe, or python script)
        candidates = list(game_dir.glob("*.sh")) + list(game_dir.glob("*.exe")) + list(game_dir.glob("*.py"))
        if not candidates:
            raise FileNotFoundError(f"No executable found in {game_dir}")

        exe = candidates[0]
        proc = subprocess.Popen([str(exe)], cwd=str(game_dir))
        return {
            "launched": True,
            "executable": str(exe.name),
            "pid": proc.pid,
        }

    async def _handle_shutdown(self, _payload: Any) -> Dict[str, Any]:
        logger.info("Shutdown requested via RPC")
        asyncio.create_task(self.stop())
        return {"status": "shutting_down"}

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        while self.is_running:
            try:
                # Read 4-byte little-endian length prefix
                len_bytes = await reader.readexactly(4)
                if not len_bytes:
                    break
                payload_len = struct.unpack("<I", len_bytes)[0]

                # Read MessagePack payload
                raw_payload = await reader.readexactly(payload_len)
                envelope = msgpack.unpackb(raw_payload, raw=False)

                # Verify auth token with constant-time comparison
                token = envelope.get("session_token", "")
                if self.auth_token and not secrets.compare_digest(token, self.auth_token):
                    response = {
                        "ok": False,
                        "result": None,
                        "error": "Unauthorized: Invalid session token",
                    }
                else:
                    method = envelope.get("method", "")
                    payload = envelope.get("payload", {})
                    handler = self._handlers.get(method)

                    if handler is None:
                        response = {
                            "ok": False,
                            "result": None,
                            "error": f"Method not found: {method}",
                        }
                    else:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                result = await handler(payload)
                            else:
                                result = handler(payload)
                            response = {"ok": True, "result": result, "error": None}
                        except Exception as e:
                            logger.exception("Error handling RPC method %s", method)
                            response = {"ok": False, "result": None, "error": str(e)}

                # Encode response with 4-byte length prefix
                resp_bytes = msgpack.packb(response, use_bin_type=True)
                resp_len = struct.pack("<I", len(resp_bytes))
                writer.write(resp_len + resp_bytes)
                await writer.drain()

            except asyncio.IncompleteReadError:
                break
            except Exception as e:
                logger.error("RPC client error: %s", e)
                break

        writer.close()
        await writer.wait_closed()

    async def start(self) -> None:
        """Starts the Unix Domain Socket or Localhost TCP server."""
        self.is_running = True

        if sys.platform != "win32":
            # Unix Domain Socket
            if self.endpoint_path.exists():
                self.endpoint_path.unlink()
            self.endpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.server = await asyncio.start_unix_server(
                self._handle_client, path=str(self.endpoint_path)
            )
            logger.info("RPC Server listening on UDS: %s", self.endpoint_path)
        else:
            # Fallback to localhost port for Windows
            port = int(os.environ.get("RENSLATE_RPC_PORT", 42424))
            self.server = await asyncio.start_server(
                self._handle_client, host="127.0.0.1", port=port
            )
            logger.info("RPC Server listening on 127.0.0.1:%d", port)

    async def stop(self) -> None:
        """Stops the server cleanly."""
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if sys.platform != "win32" and self.endpoint_path.exists():
            try:
                self.endpoint_path.unlink()
            except OSError:
                pass
        logger.info("RPC Server stopped cleanly")


def setup_death_pipe_listener(on_death: Callable[[], None]) -> None:
    """
    Listens on POSIX death-pipe file descriptor (set by parent process).
    When parent exits or drops the pipe, read returns EOF (0 bytes) -> calls on_death.
    """
    pipe_fd_str = os.environ.get("RENSLATE_DEATH_PIPE_FD")
    if not pipe_fd_str:
        return

    try:
        fd = int(pipe_fd_str)

        def death_monitor():
            try:
                # Blocks until parent closes write-end of pipe
                data = os.read(fd, 1)
                if not data:
                    on_death()
            except (OSError, ValueError):
                on_death()

        t = threading.Thread(target=death_monitor, name="DeathPipeMonitor", daemon=True)
        t.start()
    except (ValueError, OSError) as e:
        logger.warning("Failed to initialize death-pipe listener: %s", e)
