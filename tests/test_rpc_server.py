"""
test_rpc_server.py
Unit tests for renloc_core.rpc_server:
- MessagePack RPC length-prefixed transport
- 256-bit CSPRNG token authentication
- Method dispatching (ping, echo, get_status)
- Death-pipe listener behavior
"""

import asyncio
import os
import struct
import tempfile
from pathlib import Path
import pytest
import msgpack

from renloc_core.rpc_server import RpcServer, setup_death_pipe_listener


def test_rpc_server_roundtrip():
    async def run_test():
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test_rpc.sock"
            token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

            server = RpcServer(endpoint_path=sock_path, auth_token=token)
            await server.start()

            # Connect client to Unix socket
            reader, writer = await asyncio.open_unix_connection(path=str(sock_path))

            # 1. Send valid ping request
            req1 = {
                "session_token": token,
                "method": "ping",
                "payload": {},
            }
            encoded1 = msgpack.packb(req1, use_bin_type=True)
            writer.write(struct.pack("<I", len(encoded1)) + encoded1)
            await writer.drain()

            # Read response
            len_buf = await reader.readexactly(4)
            resp_len = struct.unpack("<I", len_buf)[0]
            resp_bytes = await reader.readexactly(resp_len)
            resp1 = msgpack.unpackb(resp_bytes, raw=False)

            assert resp1["ok"] is True
            assert resp1["result"]["status"] == "pong"
            assert resp1["error"] is None

            # 2. Send invalid token request -> must be rejected
            req2 = {
                "session_token": "wrong_token_value_000000000000000000000000000000000000000000000000",
                "method": "echo",
                "payload": {"hello": "world"},
            }
            encoded2 = msgpack.packb(req2, use_bin_type=True)
            writer.write(struct.pack("<I", len(encoded2)) + encoded2)
            await writer.drain()

            len_buf2 = await reader.readexactly(4)
            resp_len2 = struct.unpack("<I", len_buf2)[0]
            resp_bytes2 = await reader.readexactly(resp_len2)
            resp2 = msgpack.unpackb(resp_bytes2, raw=False)

            assert resp2["ok"] is False
            assert "Unauthorized" in resp2["error"]

            # 3. Send echo request with valid token
            req3 = {
                "session_token": token,
                "method": "echo",
                "payload": {"data": 42},
            }
            encoded3 = msgpack.packb(req3, use_bin_type=True)
            writer.write(struct.pack("<I", len(encoded3)) + encoded3)
            await writer.drain()

            len_buf3 = await reader.readexactly(4)
            resp_len3 = struct.unpack("<I", len_buf3)[0]
            resp_bytes3 = await reader.readexactly(resp_len3)
            resp3 = msgpack.unpackb(resp_bytes3, raw=False)

            assert resp3["ok"] is True
            assert resp3["result"]["data"] == 42

            writer.close()
            await writer.wait_closed()
            await server.stop()

    asyncio.run(run_test())


def test_death_pipe_listener():
    r_fd, w_fd = os.pipe()
    called = []

    os.environ["RENSLATE_DEATH_PIPE_FD"] = str(r_fd)
    setup_death_pipe_listener(lambda: called.append(True))

    # Close write end of pipe (emulating parent process exit)
    os.close(w_fd)
    import time
    time.sleep(0.05)

    assert len(called) > 0, "Death pipe listener should trigger on pipe EOF"
    os.close(r_fd)
    del os.environ["RENSLATE_DEATH_PIPE_FD"]


def test_rpc_system_specs_and_translation():
    async def run_test():
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test_rpc2.sock"
            db_path = Path(tmpdir) / "test_proj.sqlite"
            token = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

            from renloc_core.db_writer import DbWriter, init_schema
            writer = DbWriter(db_path)
            init_schema(writer.raw_connection)
            with writer.write_transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO translation_units (unit_identifier, row_order, file_path, line_number, source_text, speaker, status)
                    VALUES ('say_1', 1, 'script.rpy', 10, 'Hello world', 'Eileen', 'PENDING')
                    """
                )

            server = RpcServer(endpoint_path=sock_path, auth_token=token)
            await server.start()

            reader, r_writer = await asyncio.open_unix_connection(path=str(sock_path))

            async def rpc_call(method: str, payload: dict):
                req = {"session_token": token, "method": method, "payload": payload}
                enc = msgpack.packb(req, use_bin_type=True)
                r_writer.write(struct.pack("<I", len(enc)) + enc)
                await r_writer.drain()
                len_b = await reader.readexactly(4)
                l = struct.unpack("<I", len_b)[0]
                resp_b = await reader.readexactly(l)
                return msgpack.unpackb(resp_b, raw=False)

            # Test get_system_specs
            res_specs = await rpc_call("get_system_specs", {})
            assert res_specs["ok"] is True
            assert "total_ram_mb" in res_specs["result"]

            # Test start_translation (mock mode)
            res_trans = await rpc_call("start_translation", {"db_path": str(db_path), "engine": "mock", "limit": 10})
            assert res_trans["ok"] is True
            assert res_trans["result"]["translated_count"] == 1

            # Test progress
            res_prog = await rpc_call("get_translation_progress", {})
            assert res_prog["ok"] is True
            assert res_prog["result"]["total"] == 1

            # Test test_inference_connection
            res_conn = await rpc_call("test_inference_connection", {"server_url": "http://127.0.0.1:9999/v1"})
            assert res_conn["ok"] is True
            assert res_conn["result"]["online"] is False

            # Test unload_model
            res_unload = await rpc_call("unload_model", {})
            assert res_unload["ok"] is True

            r_writer.close()
            await r_writer.wait_closed()
            await server.stop()

    asyncio.run(run_test())


def test_rpc_scan_project():
    async def run_test():
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test_scan.sock"
            game_dir = Path(tmpdir) / "game_dir"
            game_sub = game_dir / "game"
            game_sub.mkdir(parents=True)

            import shutil
            fixture_rpyc = Path("tests/fixtures/rpyc/sample_8x.rpyc")
            shutil.copy(fixture_rpyc, game_sub / "script.rpyc")

            db_path = Path(tmpdir) / "project.sqlite"
            token = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"

            server = RpcServer(endpoint_path=sock_path, auth_token=token)
            await server.start()

            reader, writer = await asyncio.open_unix_connection(path=str(sock_path))

            req = {
                "session_token": token,
                "method": "scan_project",
                "payload": {"game_dir": str(game_dir), "db_path": str(db_path)},
            }
            enc = msgpack.packb(req, use_bin_type=True)
            writer.write(struct.pack("<I", len(enc)) + enc)
            await writer.drain()

            len_b = await reader.readexactly(4)
            l = struct.unpack("<I", len_b)[0]
            resp_b = await reader.readexactly(l)
            res = msgpack.unpackb(resp_b, raw=False)

            assert res["ok"] is True
            assert res["result"]["total_extracted"] == 2
            assert res["result"]["rpyc_count"] == 1

            writer.close()
            await writer.wait_closed()
            await server.stop()

    asyncio.run(run_test())
