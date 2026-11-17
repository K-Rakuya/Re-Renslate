"""
tests/test_rpa_section2.py
Tests for multi-part chunked archive stitching and In-Game VFS Fallback
as specified in Section 2 for crates/renloc-rpa and renloc_native.
"""

import os
import sys
import zlib
import pickle
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "py" / "renloc-core"))

import renloc_native


def test_multipart_chunked_archive_stitching_py(tmp_path):
    """Test multi-part chunked archive stitching via renloc_native VfsLayer and RpaArchive."""
    part1_path = tmp_path / "archive_part.rpa.001"
    part2_path = tmp_path / "archive_part.rpa.002"

    # Common content that spans across part 1 and part 2
    part1_chunk = b"CrossPartContent_AAA" * 10
    part2_chunk = b"CrossPartContent_BBB" * 10
    full_content = part1_chunk + part2_chunk

    header_len = 34  # RPA-3.0 header length
    key = 0x5a5a5a5a

    offset_file = header_len
    len_file = len(full_content)

    part1_total_len = header_len + len(part1_chunk)
    offset_index = part1_total_len + len(part2_chunk)

    # Index: file spans boundary of part 1 and part 2
    index = {
        "stitched_asset.dat": [(offset_file ^ key, len_file ^ key, b"")],
        "standalone.txt": [(offset_file ^ key, 10 ^ key, b"")],
    }
    index_compressed = zlib.compress(pickle.dumps(index, protocol=3))

    # Assemble part 1: header + part 1 chunk
    header = f"RPA-3.0 {offset_index:016x} {key:08x}\n".encode("latin1")
    assert len(header) == header_len

    part1_bytes = bytearray()
    part1_bytes.extend(header)
    part1_bytes.extend(part1_chunk)
    part1_path.write_bytes(part1_bytes)

    # Assemble part 2: part 2 chunk + index
    part2_bytes = bytearray()
    part2_bytes.extend(part2_chunk)
    part2_bytes.extend(index_compressed)
    part2_path.write_bytes(part2_bytes)

    # 1. Test PyRpaArchive with auto-discovery on .001
    arc = renloc_native.RpaArchive(str(part1_path))
    assert arc.is_multipart() is True
    assert arc.part_count() == 2
    assert "stitched_asset.dat" in arc.list_files()

    read_data = arc.read_file("stitched_asset.dat")
    assert bytes(read_data) == full_content

    # 2. Test PyVfsLayer mounting directory
    vfs = renloc_native.VfsLayer(str(tmp_path))
    vfs.mount_game_directory()

    files = vfs.list_all_files()
    assert "stitched_asset.dat" in files
    assert bytes(vfs.read_file("stitched_asset.dat")) == full_content

    # 3. Test loose file override
    loose_override = tmp_path / "stitched_asset.dat"
    loose_data = b"LOOSE_OVERRIDE_CONTENT"
    loose_override.write_bytes(loose_data)

    assert bytes(vfs.read_file("stitched_asset.dat")) == loose_data


def test_zix_in_game_fallback_py(tmp_path):
    """Test ZiX-12A/12B detection and In-Game VFS Fallback code generator."""
    zix_path = tmp_path / "game_data.rpa"
    zix_path.write_bytes(b"ZiX-12A 0000000000000000 00000000\n")

    arc = renloc_native.RpaArchive(str(zix_path))
    assert arc.is_fallback_required() is True

    vfs = renloc_native.VfsLayer(str(tmp_path))
    vfs.mount_game_directory()
    assert vfs.requires_in_game_fallback() is True

    # Generate script
    script = vfs.generate_in_game_fallback()
    assert "init -999 python:" in script
    assert "renpy.loader.load" in script
    assert "class RenslateVfsFallback" in script

    # Verify python syntax inside the init block
    # Strip the "init -999 python:" header line to compile Python code
    python_code = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in script.splitlines()
        if not line.startswith("init -999 python:") and not line.startswith("#")
    )
    compiled = compile(python_code, "fallback.py", "exec")
    assert compiled is not None

    # Write script to disk
    out_file = vfs.write_in_game_fallback(str(tmp_path))
    assert os.path.exists(out_file)
    assert Path(out_file).name == "00_renloc_vfs_fallback.rpy"
