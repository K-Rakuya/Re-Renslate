"""
schema.py
Database schema definition and initialization for Renslate.

Sets up:
1. translation_units (core normalized text units)
2. index_outbox & triggers (AFTER INSERT/UPDATE/DELETE tracking, eliminating Session Extension)
3. units_fts (FTS5 virtual table for full-text search with Vietnamese unicode61 support)
4. units_vec (vec0 virtual table for semantic search via sqlite-vec)
5. glossary (terminology database)
6. translation_memory (translation memory pairs)
7. dag_tasks (idempotent asynchronous task scheduler queue)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_TRANSLATION_UNITS = """
CREATE TABLE IF NOT EXISTS translation_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_identifier TEXT NOT NULL UNIQUE,
    source_text TEXT NOT NULL,
    target_text TEXT,
    speaker TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    row_order INTEGER NOT NULL,
    file_path TEXT,
    line_number INTEGER,
    context_flags TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tu_row_order ON translation_units(row_order);
CREATE INDEX IF NOT EXISTS idx_tu_status ON translation_units(status);
CREATE INDEX IF NOT EXISTS idx_tu_unit_id ON translation_units(unit_identifier);
CREATE INDEX IF NOT EXISTS idx_tu_file_line ON translation_units(file_path, line_number);
"""

SCHEMA_INDEX_OUTBOX = """
CREATE TABLE IF NOT EXISTS index_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id INTEGER NOT NULL,
    op TEXT NOT NULL CHECK(op IN ('INSERT', 'UPDATE', 'DELETE')),
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'PROCESSED', 'FAILED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending ON index_outbox(id) WHERE status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_outbox_unit ON index_outbox(unit_id);

CREATE TRIGGER IF NOT EXISTS trg_tu_ai AFTER INSERT ON translation_units
BEGIN
    INSERT INTO index_outbox (unit_id, op) VALUES (NEW.id, 'INSERT');
END;

CREATE TRIGGER IF NOT EXISTS trg_tu_au AFTER UPDATE ON translation_units
BEGIN
    INSERT INTO index_outbox (unit_id, op) VALUES (NEW.id, 'UPDATE');
END;

CREATE TRIGGER IF NOT EXISTS trg_tu_ad AFTER DELETE ON translation_units
BEGIN
    INSERT INTO index_outbox (unit_id, op) VALUES (OLD.id, 'DELETE');
END;
"""

SCHEMA_UNITS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(
    source_text,
    target_text,
    speaker,
    tokenize='unicode61 remove_diacritics 0'
);
"""

SCHEMA_GLOSSARY = """
CREATE TABLE IF NOT EXISTS glossary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_term TEXT NOT NULL,
    tgt_term TEXT NOT NULL,
    pos TEXT,
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    forbidden INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(src_term, tgt_term)
);

CREATE INDEX IF NOT EXISTS idx_glossary_src ON glossary(src_term);
"""

SCHEMA_TRANSLATION_MEMORY = """
CREATE TABLE IF NOT EXISTS translation_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_text TEXT NOT NULL,
    target_text TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    similarity_score REAL DEFAULT 1.0,
    quality_score REAL,
    source_lang TEXT DEFAULT 'en',
    target_lang TEXT DEFAULT 'vi',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tm_hash ON translation_memory(source_hash);
"""

SCHEMA_DAG_TASKS = """
CREATE TABLE IF NOT EXISTS dag_tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'RETRY')),
    priority INTEGER NOT NULL DEFAULT 0,
    unit_id INTEGER REFERENCES translation_units(id) ON DELETE CASCADE,
    dependencies TEXT,
    payload TEXT,
    result TEXT,
    error TEXT,
    retries INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    worker_id TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dag_status_prio ON dag_tasks(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_dag_unit_id ON dag_tasks(unit_id);
"""


def get_vec_schema(vec_dim: int = 384, has_sqlite_vec: bool = True) -> str:
    """Generate DDL for units_vec virtual table (vec0) or fallback table."""
    if has_sqlite_vec:
        return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS units_vec USING vec0(
    embedding float[{vec_dim}]
);
"""
    else:
        return """
CREATE TABLE IF NOT EXISTS units_vec (
    rowid INTEGER PRIMARY KEY,
    embedding BLOB
);
"""


def init_schema(
    conn: Any,
    has_sqlite_vec: bool = True,
    vec_dim: int = 384,
) -> None:
    """
    Initialize full Renslate schema on an open SQLite connection.
    Executes in a single atomic transaction.
    """
    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE;")
    try:
        cursor.execute(SCHEMA_TRANSLATION_UNITS)
        cursor.execute(SCHEMA_INDEX_OUTBOX)
        cursor.execute(SCHEMA_UNITS_FTS)

        vec_ddl = get_vec_schema(vec_dim=vec_dim, has_sqlite_vec=has_sqlite_vec)
        cursor.execute(vec_ddl)

        cursor.execute(SCHEMA_GLOSSARY)
        cursor.execute(SCHEMA_TRANSLATION_MEMORY)
        cursor.execute(SCHEMA_DAG_TASKS)

        cursor.execute("COMMIT;")
        logger.info("Renslate database schema successfully initialized (vec0=%s).", has_sqlite_vec)
    except Exception:
        cursor.execute("ROLLBACK;")
        raise
