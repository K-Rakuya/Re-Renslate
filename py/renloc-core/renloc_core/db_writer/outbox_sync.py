"""
outbox_sync.py
Atomic batch synchronization mechanism from index_outbox to FTS5 (units_fts)
and vector index (units_vec) within the apsw write cycle.

Provides atomic, consistent index updates without relying on the SQLite Session Extension
(which does not support virtual tables like FTS5 and vec0).
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

EmbedderFn = Callable[[Sequence[str]], Sequence[Union[bytes, Sequence[float], str]]]


def _pack_vector(vec: Union[bytes, Sequence[float], str]) -> Union[bytes, str]:
    """Normalize vector embedding into bytes or JSON string for vec0 / BLOB storage."""
    if isinstance(vec, (bytes, bytearray)):
        return bytes(vec)
    if isinstance(vec, str):
        return vec
    # Sequence of floats -> struct pack to IEEE 754 float32 bytes
    return struct.pack(f"{len(vec)}f", *vec)


def sync_pending_outbox(
    conn: Any,
    batch_size: int = 1000,
    embedder: Optional[EmbedderFn] = None,
    purge: bool = False,
) -> int:
    """
    Synchronizes pending index_outbox entries into `units_fts` and `units_vec`.

    If called within an existing transaction (`conn.getautocommit() == False`),
    executes statements directly as part of that atomic transaction.
    Otherwise, wraps operations in `BEGIN IMMEDIATE` / `COMMIT`.

    Parameters:
        conn: apsw Connection.
        batch_size: Maximum number of outbox rows to process per call.
        embedder: Optional function taking a list of text strings and returning vector representations.
        purge: If True, deletes processed outbox rows; if False, marks them status='PROCESSED'.

    Returns:
        Number of outbox entries processed in this batch.
    """
    cursor = conn.cursor()
    owns_tx = conn.getautocommit()

    if owns_tx:
        cursor.execute("BEGIN IMMEDIATE;")

    try:
        # 1. Fetch pending outbox records ordered by id
        rows = list(
            cursor.execute(
                "SELECT id, unit_id, op FROM index_outbox "
                "WHERE status = 'PENDING' ORDER BY id ASC LIMIT ?;",
                (batch_size,),
            )
        )

        if not rows:
            if owns_tx:
                cursor.execute("COMMIT;")
            return 0

        outbox_ids = [r[0] for r in rows]

        # 2. Determine net effect per unit_id within this batch
        # If an ID is updated multiple times in the same batch, latest op wins
        latest_op_per_unit: dict[int, str] = {}
        for _, unit_id, op in rows:
            latest_op_per_unit[unit_id] = op

        delete_ids = [uid for uid, op in latest_op_per_unit.items() if op == "DELETE"]
        upsert_ids = [uid for uid, op in latest_op_per_unit.items() if op in ("INSERT", "UPDATE")]

        # 3. Handle deletions from units_fts and units_vec
        for uid in delete_ids:
            cursor.execute("DELETE FROM units_fts WHERE rowid = ?;", (uid,))
            try:
                cursor.execute("DELETE FROM units_vec WHERE rowid = ?;", (uid,))
            except Exception as e:
                logger.debug("Failed deleting from units_vec for rowid %s: %s", uid, e)

        # 4. Handle upserts: query current translation_units
        if upsert_ids:
            # Chunk IN clause if large
            chunk_size = 500
            units_data: list[tuple[int, str, Optional[str], Optional[str]]] = []
            for i in range(0, len(upsert_ids), chunk_size):
                chunk = upsert_ids[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                query = (
                    f"SELECT id, source_text, target_text, speaker "
                    f"FROM translation_units WHERE id IN ({placeholders});"
                )
                units_data.extend(cursor.execute(query, chunk))

            found_ids = {u[0] for u in units_data}
            missing_ids = set(upsert_ids) - found_ids

            # Rows that were deleted before outbox sync was executed
            for mid in missing_ids:
                cursor.execute("DELETE FROM units_fts WHERE rowid = ?;", (mid,))
                try:
                    cursor.execute("DELETE FROM units_vec WHERE rowid = ?;", (mid,))
                except Exception:
                    pass

            # Upsert into units_fts
            for uid, src_text, tgt_text, spk in units_data:
                cursor.execute(
                    "INSERT OR REPLACE INTO units_fts(rowid, source_text, target_text, speaker) "
                    "VALUES (?, ?, ?, ?);",
                    (uid, src_text, tgt_text or "", spk or ""),
                )

            # Upsert into units_vec if embedder is provided
            if embedder is not None and units_data:
                texts = [u[1] for u in units_data]
                try:
                    embeddings = embedder(texts)
                    for (uid, _, _, _), emb in zip(units_data, embeddings):
                        packed = _pack_vector(emb)
                        cursor.execute(
                            "INSERT OR REPLACE INTO units_vec(rowid, embedding) VALUES (?, ?);",
                            (uid, packed),
                        )
                except Exception as exc:
                    logger.error("Failed to compute or store embeddings during outbox sync: %s", exc)
                    raise

        # 5. Mark or purge outbox records
        chunk_size = 500
        for i in range(0, len(outbox_ids), chunk_size):
            chunk = outbox_ids[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            if purge:
                cursor.execute(
                    f"DELETE FROM index_outbox WHERE id IN ({placeholders});",
                    chunk,
                )
            else:
                cursor.execute(
                    f"UPDATE index_outbox SET status = 'PROCESSED', processed_at = CURRENT_TIMESTAMP "
                    f"WHERE id IN ({placeholders});",
                    chunk,
                )

        if owns_tx:
            cursor.execute("COMMIT;")

        logger.debug(
            "Outbox sync: processed %d entries (upserts=%d, deletes=%d).",
            len(outbox_ids),
            len(upsert_ids),
            len(delete_ids),
        )
        return len(outbox_ids)

    except Exception:
        if owns_tx:
            cursor.execute("ROLLBACK;")
        raise


def sync_all_pending(
    conn: Any,
    batch_size: int = 1000,
    embedder: Optional[EmbedderFn] = None,
    purge: bool = False,
) -> int:
    """Synchronize all pending outbox entries in batches until queue is empty."""
    total = 0
    while True:
        count = sync_pending_outbox(
            conn,
            batch_size=batch_size,
            embedder=embedder,
            purge=purge,
        )
        total += count
        if count < batch_size:
            break
    return total
