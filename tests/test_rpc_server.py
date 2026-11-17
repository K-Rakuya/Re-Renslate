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
