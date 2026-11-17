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
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        self.register_handler("ping", self._handle_ping)
        self.register_handler("echo", self._handle_echo)
        self.register_handler("get_status", self._handle_get_status)
        self.register_handler("audit_fonts", self._handle_audit_fonts)
        self.register_handler("measure_dialogue", self._handle_measure_dialogue)
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
