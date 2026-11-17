#!/usr/bin/env python3
"""
scripts/test_renloc_native.py
Independent verification of PyO3 ABI3 bindings:
  - renloc_native.audit_font_bytes
  - renloc_native.shape_and_measure
  - renloc_native.RpaArchive
  - renloc_native.VfsLayer
"""
import os
import sys
import tempfile
import zlib
import pickle

import renloc_native

def test_audit_font_bytes():
    print("[*] Testing renloc_native.audit_font_bytes...")
    font_path = "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"
    with open(font_path, "rb") as f:
        font_bytes = f.read()

    audit = renloc_native.audit_font_bytes("AdwaitaMono", font_bytes)
    print("    Result:", audit)
    assert audit["font_name"] == "AdwaitaMono"
    assert audit["total_required"] == 134, f"Expected 134 total_required, got {audit['total_required']}"
    assert isinstance(audit["supported_count"], int)
    assert isinstance(audit["coverage_ratio"], float)
    assert isinstance(audit["is_fully_compatible"], bool)
    assert isinstance(audit["missing_chars"], str)
    print("    [+] audit_font_bytes PASSED!")

def test_shape_and_measure():
    print("[*] Testing renloc_native.shape_and_measure...")
    font_path = "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"
    with open(font_path, "rb") as f:
        font_bytes = f.read()

    # Normal shaping
    shaping = renloc_native.shape_and_measure(font_bytes, 16.0, "Xin chào thế giới!", 500.0, 100.0)
    print("    Normal result:", shaping)
    assert shaping["total_advance_pixels"] > 0
    assert shaping["max_height_pixels"] > 0
    assert shaping["glyph_count"] > 0
    assert shaping["is_overflowing"] is False

    # Overflow detection
    overflow = renloc_native.shape_and_measure(font_bytes, 16.0, "Xin chào thế giới!", 10.0, 5.0)
    print("    Overflow result:", overflow)
    assert overflow["is_overflowing"] is True
    print("    [+] shape_and_measure PASSED!")

def test_rpa_and_vfs():
    print("[*] Testing renloc_native.RpaArchive and renloc_native.VfsLayer across protocols...")
    for proto in [2, 3, 4]:
        with tempfile.TemporaryDirectory() as tmpdir:
            arc_path = os.path.join(tmpdir, f"archive_p{proto}.rpa")
            f1_content = b"Content of file 1 inside archive"
            f2_content = b"Content of file 2 inside archive"

            key = 0x12345678
            f1_off = 34
            f1_len = len(f1_content)
            f2_off = f1_off + f1_len
            f2_len = len(f2_content)
            idx_off = f2_off + f2_len

            prefix = b"" if proto >= 3 else ""
            index_dict = {
                "file1.txt": [(f1_off ^ key, f1_len ^ key, prefix)],
                "sub/file2.txt": [(f2_off ^ key, f2_len ^ key, prefix)]
            }
            raw_pickle = pickle.dumps(index_dict, protocol=proto)
            compressed_index = zlib.compress(raw_pickle)

            header = f"RPA-3.0 {idx_off:016x} {key:08x}\n".encode("latin1")
            assert len(header) == 34

            with open(arc_path, "wb") as f:
                f.write(header)
                f.write(f1_content)
                f.write(f2_content)
                f.write(compressed_index)

            # Test RpaArchive
            archive = renloc_native.RpaArchive(arc_path)
            file_list = sorted(archive.list_files())
            assert file_list == ["file1.txt", "sub/file2.txt"], f"Unexpected files: {file_list}"

            r1 = archive.read_file("file1.txt")
            assert r1 == f1_content, f"File 1 mismatch: {r1}"
            r2 = archive.read_file("sub/file2.txt")
            assert r2 == f2_content, f"File 2 mismatch: {r2}"

            # Test VfsLayer
            vfs = renloc_native.VfsLayer(tmpdir)
            vfs.mount_game_directory()

            all_files = vfs.list_all_files()
            assert "file1.txt" in all_files
            assert vfs.read_file("file1.txt") == f1_content

            # Loose override
            loose_content = b"Overridden loose file 1 content"
            loose_path = os.path.join(tmpdir, "file1.txt")
            with open(loose_path, "wb") as f:
                f.write(loose_content)

            assert vfs.read_file("file1.txt") == loose_content
            print(f"    [+] Protocol {proto} RpaArchive & VfsLayer OK!")

    print("    [+] RpaArchive and VfsLayer tests PASSED!")

if __name__ == "__main__":
    print(f"[*] Python executable: {sys.executable}")
    print(f"[*] Python version: {sys.version}")
    test_audit_font_bytes()
    test_shape_and_measure()
    test_rpa_and_vfs()
    print("[+] ALL renloc_native PyO3 ABI3 tests PASSED without Segfault or ABI mismatch!")
