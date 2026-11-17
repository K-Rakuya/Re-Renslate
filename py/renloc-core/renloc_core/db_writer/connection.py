"""
connection.py
APSW Single-Writer connection manager for Renslate SQLite storage.

Enforces:
1. Standard WAL pragmas:
   PRAGMA journal_mode = WAL;
   PRAGMA synchronous = NORMAL;
   PRAGMA foreign_keys = ON;
   PRAGMA busy_timeout = 10000;
   PRAGMA temp_store = MEMORY;
   PRAGMA mmap_size = 268435456;
2. Strict Single-Writer serialization with Python RLock and BEGIN IMMEDIATE.
3. Automatic savepoint nesting for nested transactions.
4. Drive guard integration against network -shm corruption.
5. Automatic sqlite-vec extension registration.
6. Atomic outbox sync hooks within write cycles.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, Union

import apsw

from .backup import hot_backup
from .drive_guard import SafePathResult, resolve_safe_db_path
from .outbox_sync import EmbedderFn, sync_all_pending, sync_pending_outbox
from .schema import init_schema

logger = logging.getLogger(__name__)


class DbWriter:
    """
    Thread-safe APSW Single-Writer connection wrapper for Renslate.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        auto_init_schema: bool = True,
        load_vec: bool = True,
        vec_dim: int = 384,
        busy_timeout_ms: int = 10000,
        auto_guard: bool = True,
        fallback_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._write_lock = threading.RLock()
        self._tx_depth = 0
        self.vec_dim = vec_dim

        # 1. Drive safety guard
        if auto_guard:
            guard_result = resolve_safe_db_path(db_path, fallback_base_dir=fallback_dir)
            self._path_info: SafePathResult = guard_result
            self.db_path = guard_result.safe_path
        else:
            resolved = Path(db_path).resolve()
            self._path_info = SafePathResult(
                safe_path=resolved,
                original_path=resolved,
                is_relocated=False,
                fs_type="unknown",
                reason="Auto-guard disabled",
            )
            self.db_path = resolved

        # Ensure parent directory exists for on-disk databases
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Open APSW Connection with URI support
        open_flags = (
            apsw.SQLITE_OPEN_READWRITE
            | apsw.SQLITE_OPEN_CREATE
            | apsw.SQLITE_OPEN_URI
        )
        self._conn = apsw.Connection(str(self.db_path), flags=open_flags)

        # 3. Load sqlite-vec extension if requested
        self.has_vec = False
        if load_vec:
            try:
                import sqlite_vec

                self._conn.enable_load_extension(True)
                self._conn.load_extension(sqlite_vec.loadable_path())
                self.has_vec = True
                logger.info("sqlite-vec extension successfully loaded.")
            except Exception as exc:
                logger.warning("sqlite-vec extension could not be loaded: %s. Using table fallback.", exc)
                self.has_vec = False

        # 4. Apply standard WAL pragmas
        self._apply_pragmas(busy_timeout_ms)

        # 5. Initialize database schema
        if auto_init_schema:
            init_schema(self._conn, has_sqlite_vec=self.has_vec, vec_dim=self.vec_dim)

    def _apply_pragmas(self, busy_timeout_ms: int) -> None:
        """Apply required WAL pragmas."""
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")
        cursor.execute("PRAGMA temp_store = MEMORY;")
        cursor.execute("PRAGMA mmap_size = 268435456;")  # 256MB mmap
        cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache

    @property
    def path_info(self) -> SafePathResult:
        """Path safety and relocation details."""
        return self._path_info

    @property
    def is_relocated(self) -> bool:
        """Whether the database was relocated from a network drive to local storage."""
        return self._path_info.is_relocated

    @property
    def raw_connection(self) -> apsw.Connection:
        """Underlying apsw Connection object."""
        return self._conn

    def cursor(self) -> apsw.Cursor:
        """Create a new cursor on the connection."""
        return self._conn.cursor()

    @contextmanager
    def write_transaction(
        self,
        auto_sync: bool = True,
        embedder: Optional[EmbedderFn] = None,
    ) -> Generator[apsw.Cursor, None, None]:
        """
        Context manager for write transactions.
        Serializes writes with Python RLock and starts with BEGIN IMMEDIATE
        to guarantee no deadlock or SQLITE_BUSY upgrade lock conflicts.

        Nested transactions within the same thread use SQLite SAVEPOINTs.
        """
        with self._write_lock:
            cursor = self._conn.cursor()
            if self._tx_depth == 0:
                cursor.execute("BEGIN IMMEDIATE;")
                self._tx_depth += 1
                try:
                    yield cursor
                    self._tx_depth -= 1
                    if auto_sync:
                        sync_pending_outbox(self._conn, embedder=embedder)
                    cursor.execute("COMMIT;")
                except Exception:
                    self._tx_depth = 0
                    cursor.execute("ROLLBACK;")
                    raise
            else:
                # Nested transaction: use SAVEPOINT
                sp_name = f"sp_{self._tx_depth}_{uuid.uuid4().hex[:6]}"
                cursor.execute(f"SAVEPOINT {sp_name};")
                self._tx_depth += 1
                try:
                    yield cursor
                    self._tx_depth -= 1
                    cursor.execute(f"RELEASE SAVEPOINT {sp_name};")
                except Exception:
                    self._tx_depth -= 1
                    cursor.execute(f"ROLLBACK TO SAVEPOINT {sp_name};")
                    raise

    @contextmanager
    def read_transaction(self) -> Generator[apsw.Cursor, None, None]:
        """
        Context manager for read transactions.
        Uses BEGIN DEFERRED for non-blocking WAL snapshot reads.
        """
        cursor = self._conn.cursor()
        cursor.execute("BEGIN DEFERRED;")
        try:
            yield cursor
            cursor.execute("COMMIT;")
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    def execute(self, sql: str, bindings: Optional[Sequence[Any]] = None) -> apsw.Cursor:
        """Execute a single SQL statement."""
        cursor = self._conn.cursor()
        if bindings is not None:
            return cursor.execute(sql, bindings)
        return cursor.execute(sql)

    def executemany(self, sql: str, bindings_list: Sequence[Sequence[Any]]) -> apsw.Cursor:
        """Execute multiple SQL statements with parameter sets."""
        cursor = self._conn.cursor()
        return cursor.executemany(sql, bindings_list)

    def insert_translation_unit(
        self,
        unit_identifier: str,
        source_text: str,
        target_text: Optional[str] = None,
        speaker: Optional[str] = None,
        status: str = "PENDING",
        row_order: int = 0,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        context_flags: Optional[str] = None,
        metadata: Optional[str] = None,
        auto_sync: bool = True,
        embedder: Optional[EmbedderFn] = None,
    ) -> int:
        """
        Insert a translation unit in an immediate write transaction.
        Returns the inserted row id.
        """
        query = """
        INSERT INTO translation_units (
            unit_identifier, source_text, target_text, speaker, status,
            row_order, file_path, line_number, context_flags, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.write_transaction(auto_sync=auto_sync, embedder=embedder) as cursor:
            cursor.execute(
                query,
                (
                    unit_identifier,
                    source_text,
                    target_text,
                    speaker,
                    status,
                    row_order,
                    file_path,
                    line_number,
                    context_flags,
                    metadata,
                ),
            )
            # Retrieve generated primary key
            unit_id = self._conn.last_insert_rowid()
            return unit_id

    def update_translation_unit(
        self,
        unit_id: int,
        target_text: Optional[str] = None,
        status: Optional[str] = None,
        auto_sync: bool = True,
    ) -> None:
        """Update translation unit target text and/or status."""
        updates: list[str] = []
        params: list[Any] = []
        if target_text is not None:
            updates.append("target_text = ?")
            params.append(target_text)
        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if not updates:
            return

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(unit_id)
        query = f"UPDATE translation_units SET {', '.join(updates)} WHERE id = ?;"

        with self.write_transaction(auto_sync=auto_sync) as cursor:
            cursor.execute(query, params)

    def delete_translation_unit(self, unit_id: int, auto_sync: bool = True) -> None:
        """Delete translation unit by ID."""
        with self.write_transaction(auto_sync=auto_sync) as cursor:
            cursor.execute("DELETE FROM translation_units WHERE id = ?;", (unit_id,))

    def sync_outbox(
        self,
        batch_size: int = 1000,
        embedder: Optional[EmbedderFn] = None,
        purge: bool = False,
    ) -> int:
        """Manually trigger atomic synchronization of pending outbox entries."""
        with self._write_lock:
            return sync_all_pending(
                self._conn,
                batch_size=batch_size,
                embedder=embedder,
                purge=purge,
            )

    def hot_backup(self, dest_path: Union[str, Path], overwrite: bool = True) -> Path:
        """
        Perform hot backup using VACUUM INTO without interrupting readers.
        """
        with self._write_lock:
            return hot_backup(self._conn, dest_path=dest_path, overwrite=overwrite)

    def getautocommit(self) -> bool:
        """Returns True if the connection is currently in autocommit mode."""
        return self._conn.getautocommit()

    def checkpoint(self, mode: int = apsw.SQLITE_CHECKPOINT_PASSIVE) -> Tuple[int, int]:
        """
        Execute WAL checkpoint:
        apsw.SQLITE_CHECKPOINT_PASSIVE, SQLITE_CHECKPOINT_FULL, SQLITE_CHECKPOINT_RESTART, or SQLITE_CHECKPOINT_TRUNCATE.
        Returns (wal_pages, checkpointed_pages).
        """
        with self._write_lock:
            return self._conn.wal_checkpoint(mode=mode)

    def close(self) -> None:
        """Close connection cleanly."""
        with self._write_lock:
            self._conn.close()
