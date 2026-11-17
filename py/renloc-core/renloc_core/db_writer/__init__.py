"""
db_writer package
SQLite Single-Writer storage engine for Renslate.

Includes:
- DbWriter: APSW Single-Writer connection manager with WAL pragmas & BEGIN IMMEDIATE
- drive_guard: Safe path validation avoiding network drive -shm corruption
- schema: DDL initialization for translation units, FTS5, units_vec, glossary, TM, DAG
- outbox_sync: Atomic synchronization from index_outbox to FTS5 & vec0
- backup: Hot backup via VACUUM INTO
"""

from .backup import BackupError, hot_backup, verify_database_integrity
from .connection import DbWriter
from .drive_guard import (
    SafePathResult,
    is_network_filesystem,
    resolve_safe_db_path,
)
from .outbox_sync import sync_all_pending, sync_pending_outbox
from .schema import init_schema

__all__ = [
    "DbWriter",
    "init_schema",
    "sync_pending_outbox",
    "sync_all_pending",
    "is_network_filesystem",
    "resolve_safe_db_path",
    "SafePathResult",
    "hot_backup",
    "verify_database_integrity",
    "BackupError",
]
