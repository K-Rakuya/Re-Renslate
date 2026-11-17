"""
test_db_writer.py
Comprehensive unit test suite for renloc_core.db_writer:
- APSW Single-Writer WAL connection and PRAGMAs
- BEGIN IMMEDIATE transaction semantics and nested savepoints
- Schema initialization and trigger index_outbox tracking
- FTS5 full-text search with Vietnamese unicode61 support
- Vector search with sqlite-vec (vec0)
- Outbox atomic batch synchronization
- Drive guard filesystem detection and network drive relocation
- Hot backup (VACUUM INTO) with atomic staging and integrity checks
- Multi-threaded read/write concurrency stress testing
- Interoperability verification with Rust renloc-db query patterns
"""

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
import unittest

import apsw
import pytest

from renloc_core.db_writer import (
    BackupError,
    DbWriter,
    SafePathResult,
    hot_backup,
    init_schema,
    is_network_filesystem,
    resolve_safe_db_path,
    sync_all_pending,
    sync_pending_outbox,
    verify_database_integrity,
)
from renloc_core.db_writer.drive_guard import (
    DRIVE_FIXED,
    DRIVE_REMOTE,
    _check_windows_drive,
)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary database file path."""
    return tmp_path / "test_renslate.sqlite"


class TestConnectionAndPragmas:
    """Tests for APSW Single-Writer connection and WAL pragma configuration."""

    def test_wal_pragmas_applied(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)
        cursor = writer.cursor()

        # Check journal mode is WAL
        jm = list(cursor.execute("PRAGMA journal_mode;"))[0][0].lower()
        assert jm == "wal", f"Expected WAL mode, got {jm}"

        # Check synchronous is NORMAL (1)
        sync = list(cursor.execute("PRAGMA synchronous;"))[0][0]
        assert sync == 1, f"Expected synchronous=NORMAL (1), got {sync}"

        # Check foreign keys ON (1)
        fk = list(cursor.execute("PRAGMA foreign_keys;"))[0][0]
        assert fk == 1, f"Expected foreign_keys=ON (1), got {fk}"

        # Check busy timeout
        bt = list(cursor.execute("PRAGMA busy_timeout;"))[0][0]
        assert bt == 10000, f"Expected busy_timeout=10000, got {bt}"

        # Check temp store is MEMORY (2)
        ts = list(cursor.execute("PRAGMA temp_store;"))[0][0]
        assert ts == 2, f"Expected temp_store=MEMORY (2), got {ts}"

        writer.close()

    def test_write_transaction_immediate_and_rollback(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)

        # Successful transaction
        with writer.write_transaction() as cursor:
            cursor.execute(
                "INSERT INTO translation_units (unit_identifier, source_text, row_order) "
                "VALUES ('u1', 'Test 1', 1);"
            )

        count = list(writer.execute("SELECT count(*) FROM translation_units;"))[0][0]
        assert count == 1

        # Transaction rollback on exception
        with pytest.raises(ValueError, match="Simulated write failure"):
            with writer.write_transaction() as cursor:
                cursor.execute(
                    "INSERT INTO translation_units (unit_identifier, source_text, row_order) "
                    "VALUES ('u2', 'Test 2', 2);"
                )
                raise ValueError("Simulated write failure")

        count_after = list(writer.execute("SELECT count(*) FROM translation_units;"))[0][0]
        assert count_after == 1, "Rolled back transaction must not persist row"

        writer.close()

    def test_nested_write_transaction_savepoints(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)

        with writer.write_transaction() as cursor1:
            cursor1.execute(
                "INSERT INTO translation_units (unit_identifier, source_text, row_order) "
                "VALUES ('outer_1', 'Outer text', 1);"
            )

            # Nested transaction that rolls back via savepoint
            try:
                with writer.write_transaction() as cursor2:
                    cursor2.execute(
                        "INSERT INTO translation_units (unit_identifier, source_text, row_order) "
                        "VALUES ('inner_1', 'Inner text', 2);"
                    )
                    raise RuntimeError("Inner error")
            except RuntimeError:
                pass

            # Outer continues and inserts another row
            cursor1.execute(
                "INSERT INTO translation_units (unit_identifier, source_text, row_order) "
                "VALUES ('outer_2', 'Outer text 2', 3);"
            )

        rows = list(writer.execute("SELECT unit_identifier FROM translation_units ORDER BY row_order ASC;"))
        uids = [r[0] for r in rows]
        assert uids == ["outer_1", "outer_2"], f"Expected outer rows only, got: {uids}"

        writer.close()


class TestSchemaAndTriggers:
    """Tests for database schema initialization and index_outbox triggers."""

    def test_all_tables_exist(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True, load_vec=True)
        cursor = writer.cursor()
        tables = [
            r[0]
            for r in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
            )
        ]

        expected_tables = [
            "translation_units",
            "index_outbox",
            "units_fts",
            "units_vec",
            "glossary",
            "translation_memory",
            "dag_tasks",
        ]
        for tbl in expected_tables:
            assert tbl in tables, f"Expected table '{tbl}' to exist in schema"

        writer.close()

    def test_index_outbox_triggers(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)

        # 1. Insert triggers outbox INSERT
        uid = writer.insert_translation_unit(
            unit_identifier="dialogue_101",
            source_text="Eileen said hello",
            speaker="eileen",
            row_order=1,
            auto_sync=False,
        )
        outbox_rows = list(
            writer.execute(
                "SELECT unit_id, op, status FROM index_outbox WHERE unit_id = ?;",
                (uid,),
            )
        )
        assert len(outbox_rows) == 1
        assert outbox_rows[0] == (uid, "INSERT", "PENDING")

        # 2. Update triggers outbox UPDATE
        writer.update_translation_unit(
            uid,
            target_text="Eileen nói xin chào",
            status="TRANSLATED",
            auto_sync=False,
        )
        outbox_rows = list(
            writer.execute(
                "SELECT unit_id, op, status FROM index_outbox WHERE unit_id = ? ORDER BY id ASC;",
                (uid,),
            )
        )
        assert len(outbox_rows) == 2
        assert outbox_rows[1] == (uid, "UPDATE", "PENDING")

        # 3. Delete triggers outbox DELETE
        writer.delete_translation_unit(uid, auto_sync=False)
        outbox_rows = list(
            writer.execute(
                "SELECT unit_id, op, status FROM index_outbox WHERE unit_id = ? ORDER BY id ASC;",
                (uid,),
            )
        )
        assert len(outbox_rows) == 3
        assert outbox_rows[2] == (uid, "DELETE", "PENDING")

        writer.close()


class TestOutboxSyncAndSearch:
    """Tests for atomic outbox synchronization into FTS5 and sqlite-vec."""

    def test_fts5_vietnamese_diacritics_search(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)

        # Insert Vietnamese texts with tone marks
        writer.insert_translation_unit(
            unit_identifier="vn_1",
            source_text="Ánh trăng vằng vặc soi sáng dòng sông.",
            target_text="The bright moonlight illuminates the river.",
            speaker="narrator",
            row_order=1,
            auto_sync=True,
        )
        writer.insert_translation_unit(
            unit_identifier="vn_2",
            source_text="Gió thu nhè nhẹ thổi qua cánh đồng lúa.",
            target_text="Autumn wind blows gently across the rice field.",
            speaker="narrator",
            row_order=2,
            auto_sync=True,
        )

        # Query FTS5 matching Vietnamese words with diacritics
        res1 = list(
            writer.execute(
                "SELECT u.unit_identifier, u.source_text "
                "FROM units_fts f JOIN translation_units u ON f.rowid = u.id "
                "WHERE units_fts MATCH 'vằng vặc';"
            )
        )
        assert len(res1) == 1
        assert res1[0][0] == "vn_1"

        res2 = list(
            writer.execute(
                "SELECT u.unit_identifier, u.source_text "
                "FROM units_fts f JOIN translation_units u ON f.rowid = u.id "
                "WHERE units_fts MATCH 'cánh đồng';"
            )
        )
        assert len(res2) == 1
        assert res2[0][0] == "vn_2"

        writer.close()

    def test_vec0_vector_sync_and_nearest_search(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True, load_vec=True, vec_dim=4)
        if not writer.has_vec:
            pytest.skip("sqlite-vec extension is not available in environment")

        def mock_embedder(texts):
            # Semantic mock: text containing 'cat' -> [1.0, 0.0, 0.0, 0.0]
            # text containing 'dog' -> [0.0, 1.0, 0.0, 0.0]
            embs = []
            for t in texts:
                if "cat" in t.lower():
                    embs.append([1.0, 0.0, 0.0, 0.0])
                else:
                    embs.append([0.0, 1.0, 0.0, 0.0])
            return embs

        uid1 = writer.insert_translation_unit(
            unit_identifier="vec_1",
            source_text="The cute cat is sleeping",
            row_order=1,
            auto_sync=False,
        )
        uid2 = writer.insert_translation_unit(
            unit_identifier="vec_2",
            source_text="The friendly dog is barking",
            row_order=2,
            auto_sync=False,
        )

        # Sync outbox with embedder
        synced = writer.sync_outbox(embedder=mock_embedder)
        assert synced == 2

        # Query nearest to cat vector [1.0, 0.0, 0.0, 0.0]
        import struct
        cat_vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)

        nearest = list(
            writer.execute(
                "SELECT rowid, distance FROM units_vec WHERE embedding MATCH ? AND k = 1;",
                (cat_vec,),
            )
        )
        assert len(nearest) == 1
        assert nearest[0][0] == uid1
        assert nearest[0][1] == pytest.approx(0.0, abs=1e-4)

        writer.close()

    def test_outbox_batch_deduplication(self, temp_db_path: Path):
        """Test multiple operations on the same unit within a single batch."""
        writer = DbWriter(temp_db_path, auto_init_schema=True)

        # Unit A: Insert, then Update, then Delete -> net is Delete
        uid_a = writer.insert_translation_unit("u_a", "Text A", row_order=1, auto_sync=False)
        writer.update_translation_unit(uid_a, target_text="Updated A", auto_sync=False)
        writer.delete_translation_unit(uid_a, auto_sync=False)

        # Unit B: Insert, then Update -> net is Upsert
        uid_b = writer.insert_translation_unit("u_b", "Text B", row_order=2, auto_sync=False)
        writer.update_translation_unit(uid_b, target_text="Updated B", auto_sync=False)

        # Sync all
        writer.sync_outbox(purge=True)

        # Verify units_fts has only u_b
        fts_rows = list(writer.execute("SELECT rowid, target_text FROM units_fts;"))
        assert len(fts_rows) == 1
        assert fts_rows[0][0] == uid_b
        assert fts_rows[0][1] == "Updated B"

        writer.close()


class TestDriveGuard:
    """Tests for filesystem validation and network drive relocation."""

    def test_in_memory_and_uri_db(self):
        # 1. In-memory database
        mem_writer = DbWriter(":memory:", auto_init_schema=True)
        assert not mem_writer.is_relocated
        assert str(mem_writer.db_path) == ":memory:"
        mem_writer.insert_translation_unit("mem_1", "Hello memory", row_order=1)
        res = list(mem_writer.execute("SELECT count(*) FROM translation_units;"))[0][0]
        assert res == 1
        mem_writer.close()

        # 2. URI database
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            uri_file = f.name
        uri_path = f"file:{uri_file}?mode=rwc"
        uri_writer = DbWriter(uri_path, auto_init_schema=True)
        uri_writer.insert_translation_unit("uri_1", "Hello URI", row_order=1)
        res_uri = list(uri_writer.execute("SELECT count(*) FROM translation_units;"))[0][0]
        assert res_uri == 1
        uri_writer.close()

    def test_linux_mounts_escaping_and_prefix_boundary(self, tmp_path: Path):
        from renloc_core.db_writer.drive_guard import _check_linux_mounts
        
        # Test exact directory boundary: /var/share should NOT match /var/share-other
        fake_mounts = """
        /dev/sda1 / ext4 rw 0 0
        192.168.1.50:/nfs /media/my\\040nfs\\040share nfs rw 0 0
        192.168.1.51:/cifs /mnt/cifs cifs rw 0 0
        """
        with patch("builtins.open", mock_open(read_data=fake_mounts)):
            # Inside NFS share with escaped spaces
            is_net, fstype = _check_linux_mounts(Path("/media/my nfs share/game.sqlite"))
            assert is_net
            assert fstype == "nfs"

            # False prefix boundary: /mnt/cifs-backup should NOT match /mnt/cifs
            is_net_false, fstype_false = _check_linux_mounts(Path("/mnt/cifs-backup/game.sqlite"))
            assert not is_net_false
            assert fstype_false == "ext4"

    def test_simulated_network_mount_relocation(self, tmp_path: Path):
        fake_db = tmp_path / "network_project" / "project.sqlite"
        fake_fallback = tmp_path / "local_fallback"

        # Mock is_network_filesystem to simulate NFS
        with patch("renloc_core.db_writer.drive_guard.is_network_filesystem", return_value=(True, "nfs")):
            result = resolve_safe_db_path(fake_db, fallback_base_dir=fake_fallback)

            assert result.is_relocated
            assert "nfs" in result.fs_type
            assert result.original_path == fake_db.resolve()
            assert fake_fallback in result.safe_path.parents
            assert result.safe_path.name == "project.sqlite"

    def test_windows_unc_and_remote_detection(self):
        # UNC path
        is_net, reason = _check_windows_drive(Path(r"\\192.168.1.100\games\vn\game.sqlite"))
        assert is_net
        assert "UNC" in reason

        # Mock _get_drive_type_win32
        with patch("renloc_core.db_writer.drive_guard._get_drive_type_win32", return_value=DRIVE_REMOTE):
            is_net, reason = _check_windows_drive(Path(r"Z:\game.sqlite"))
            assert is_net
            assert reason == "DRIVE_REMOTE"

        with patch("renloc_core.db_writer.drive_guard._get_drive_type_win32", return_value=DRIVE_FIXED):
            is_net, reason = _check_windows_drive(Path(r"C:\game.sqlite"))
            assert not is_net


class TestHotBackup:
    """Tests for VACUUM INTO hot backup mechanism."""

    def test_hot_backup_creates_valid_compact_copy(self, temp_db_path: Path, tmp_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)

        for i in range(50):
            writer.insert_translation_unit(
                unit_identifier=f"backup_tu_{i}",
                source_text=f"Dialogue line {i}",
                target_text=f"Dong thoai {i}",
                row_order=i,
                auto_sync=True,
            )

        backup_file = tmp_path / "backups" / "snapshot.sqlite"
        result_path = writer.hot_backup(backup_file, overwrite=True)

        assert result_path.exists()
        assert result_path.stat().st_size > 0
        assert verify_database_integrity(result_path)

        # Open backup database and verify row count
        bak_conn = apsw.Connection(str(result_path), flags=apsw.SQLITE_OPEN_READONLY)
        count = list(bak_conn.cursor().execute("SELECT count(*) FROM translation_units;"))[0][0]
        bak_conn.close()
        assert count == 50

        writer.close()

    def test_hot_backup_overwrite_guard(self, temp_db_path: Path, tmp_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)
        backup_file = tmp_path / "existing_backup.sqlite"
        backup_file.write_text("existing content")

        # Fails when overwrite=False
        with pytest.raises(FileExistsError):
            writer.hot_backup(backup_file, overwrite=False)

        # Succeeds when overwrite=True
        result_path = writer.hot_backup(backup_file, overwrite=True)
        assert result_path.exists()
        assert verify_database_integrity(result_path)

        writer.close()


class TestConcurrencyAndRustInterop:
    """Tests for multi-threaded read/write concurrency and Rust renloc-db compatibility."""

    def test_threaded_writers_and_readers_no_deadlock(self, temp_db_path: Path):
        writer = DbWriter(temp_db_path, auto_init_schema=True)
        num_threads = 4
        rows_per_thread = 25
        errors: list[Exception] = []

        def worker_writer(thread_idx: int):
            try:
                for j in range(rows_per_thread):
                    writer.insert_translation_unit(
                        unit_identifier=f"th_{thread_idx}_row_{j}",
                        source_text=f"Source from thread {thread_idx} row {j}",
                        target_text=f"Target {j}",
                        row_order=thread_idx * 1000 + j,
                        auto_sync=True,
                    )
            except Exception as e:
                errors.append(e)

        def worker_reader():
            try:
                for _ in range(30):
                    with writer.read_transaction() as cursor:
                        list(cursor.execute("SELECT count(*) FROM translation_units;"))
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads: list[threading.Thread] = []
        for i in range(num_threads):
            threads.append(threading.Thread(target=worker_writer, args=(i,)))
        for _ in range(2):
            threads.append(threading.Thread(target=worker_reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrency errors occurred: {errors}"

        total_rows = list(writer.execute("SELECT count(*) FROM translation_units;"))[0][0]
        assert total_rows == num_threads * rows_per_thread

        fts_count = list(writer.execute("SELECT count(*) FROM units_fts;"))[0][0]
        assert fts_count == total_rows

        writer.close()

    def test_interop_with_rust_renloc_db_keyset_and_fts(self, temp_db_path: Path):
        """
        Populates database via Python DbWriter and verifies that queries matching
        crates/renloc-db QueryEngine succeed with exact expected structure.
        """
        writer = DbWriter(temp_db_path, auto_init_schema=True)

        for i in range(1, 21):
            writer.insert_translation_unit(
                unit_identifier=f"say_{i:03d}",
                source_text=f"Hello character {i}",
                target_text=f"Xin chao nhan vat {i}",
                speaker="narrator" if i % 2 == 0 else "eileen",
                status="COMMITTED",
                row_order=i,
                auto_sync=True,
            )

        # Keyset query matching QueryEngine::fetch_page_keyset in crates/renloc-db/src/fts_vec.rs
        keyset_sql = """
        SELECT id, unit_identifier, source_text, target_text, speaker, status, row_order
        FROM translation_units
        WHERE row_order > ?
        ORDER BY row_order ASC
        LIMIT ?;
        """
        page1 = list(writer.execute(keyset_sql, (0, 10)))
        assert len(page1) == 10
        assert page1[0][1] == "say_001"
        assert page1[-1][1] == "say_010"

        last_row_order = page1[-1][6]
        page2 = list(writer.execute(keyset_sql, (last_row_order, 10)))
        assert len(page2) == 10
        assert page2[0][1] == "say_011"
        assert page2[-1][1] == "say_020"

        # FTS query matching QueryEngine::search_fts in crates/renloc-db/src/fts_vec.rs
        fts_sql = """
        SELECT u.id, u.unit_identifier, u.source_text, u.target_text, u.speaker, u.status, u.row_order
        FROM units_fts f
        JOIN translation_units u ON f.rowid = u.id
        WHERE units_fts MATCH ?
        ORDER BY rank
        LIMIT ?;
        """
        fts_res = list(writer.execute(fts_sql, ("Xin chao", 5)))
        assert len(fts_res) == 5

        writer.close()
