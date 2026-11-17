"""
backup.py
Hot backup mechanism using SQLite VACUUM INTO via apsw Single-Writer.

Produces a consistent, compacted snapshot of the database without interrupting
concurrent reader threads or rusqlite reader connections.
Features atomic staging (writing to temporary sibling file first), integrity verification,
and atomic replacement.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import apsw

logger = logging.getLogger(__name__)


class BackupError(Exception):
    """Raised when hot backup fails or integrity check fails."""
    pass


def verify_database_integrity(db_path: Path | str, quick: bool = True) -> bool:
    """
    Verify SQLite database integrity using PRAGMA quick_check or integrity_check.
    Returns True if database is healthy.
    """
    path_str = str(Path(db_path).resolve())
    if not os.path.exists(path_str):
        return False

    try:
        conn = apsw.Connection(path_str, flags=apsw.SQLITE_OPEN_READONLY)
        cursor = conn.cursor()
        pragma_cmd = "PRAGMA quick_check;" if quick else "PRAGMA integrity_check;"
        result = list(cursor.execute(pragma_cmd))
        conn.close()

        if len(result) == 1 and result[0][0] == "ok":
            return True
        logger.error("Integrity check failed on '%s': %s", path_str, result)
        return False
    except Exception as exc:
        logger.error("Error verifying integrity of '%s': %s", path_str, exc)
        return False


def hot_backup(
    conn: Any,
    dest_path: Path | str,
    overwrite: bool = True,
    verify_integrity: bool = True,
) -> Path:
    """
    Executes a hot backup using `VACUUM INTO` on the active apsw connection.

    Parameters:
        conn: Active apsw.Connection (Single-Writer).
        dest_path: Destination path for the backup database file.
        overwrite: If True, replaces existing backup file atomically. If False, fails if file exists.
        verify_integrity: If True, runs PRAGMA quick_check on the backup before finalizing.

    Returns:
        Resolved Path of the created backup file.
    """
    destination = Path(dest_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Backup target already exists and overwrite=False: '{destination}'")

    # SQLite VACUUM INTO fails if target file already exists.
    # To maintain atomicity, vacuum into a unique temporary file in the same directory,
    # then verify and rename.
    tmp_filename = f"{destination.name}.tmp_{uuid.uuid4().hex[:10]}"
    tmp_path = destination.parent / tmp_filename

    if tmp_path.exists():
        tmp_path.unlink()

    cursor = conn.cursor()
    try:
        logger.info("Starting hot backup (VACUUM INTO) to temporary file '%s'...", tmp_path)
        cursor.execute("VACUUM INTO ?;", (str(tmp_path),))

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise BackupError(f"VACUUM INTO completed but backup file is missing or empty: {tmp_path}")

        # Integrity verification
        if verify_integrity:
            if not verify_database_integrity(tmp_path, quick=True):
                raise BackupError(f"Integrity check failed on newly created backup file: {tmp_path}")

        # Atomic replacement
        tmp_path.replace(destination)
        logger.info("Hot backup completed successfully: '%s' (%d bytes).", destination, destination.stat().st_size)
        return destination

    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        logger.error("Hot backup failed: %s", exc)
        raise BackupError(f"Hot backup to '{destination}' failed: {exc}") from exc
