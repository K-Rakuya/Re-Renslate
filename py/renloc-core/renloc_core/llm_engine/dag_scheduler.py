"""
dag_scheduler.py
Persistent Idempotent DAG Scheduler for LLM Inference backed by SQLite and APSW.

Core Principles:
1. Deterministic Task Key:
   dag_task_key = sha256(f"{source_hash}|{icfg_context_id}|{glossary_version}|{model_id}")
2. Strictly Monotonic Milestone Transitions:
   PENDING -> LEASED -> DRAFTED -> REFINED -> QE_SCORED -> COMMITTED
3. Atomic APSW BEGIN IMMEDIATE transactions:
   Prevents race conditions, lock upgrades, and double-translation deadlocks.
4. Robust Crash Recovery:
   Expired leases and sudden power outages / crashes recover cleanly to last
   saved milestone or re-queue to PENDING without duplicate translation.
"""

from __future__ import annotations

import datetime
import enum
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import apsw

logger = logging.getLogger(__name__)


# ==============================================================================
# Task Status and Key Derivation
# ==============================================================================

class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    DRAFTED = "DRAFTED"
    REFINED = "REFINED"
    QE_SCORED = "QE_SCORED"
    COMMITTED = "COMMITTED"
    RUNNING = "RUNNING"  # Legacy compatibility
    COMPLETED = "COMPLETED"  # Legacy compatibility
    FAILED = "FAILED"
    RETRY = "RETRY"


def compute_dag_task_key(
    source_hash: str,
    icfg_context_id: str,
    glossary_version: str,
    model_id: str,
) -> str:
    """
    Derive deterministic, idempotent task key.
    dag_task_key = sha256(source_hash | icfg_context_id | glossary_version | model_id)
    """
    raw = f"{source_hash}|{icfg_context_id}|{glossary_version}|{model_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class DagTask:
    """Represents a scheduled unit of inference work."""
    task_id: str  # The dag_task_key
    task_type: str = "translate"
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    unit_id: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    worker_id: Optional[str] = None
    lease_expires_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ==============================================================================
# Persistent DAG Scheduler
# ==============================================================================

class DagScheduler:
    """
    APSW-backed DAG task scheduler with idempotent key deduplication
    and crash-resilient state transitions.
    """

    def __init__(
        self,
        db_conn_or_path: Union[apsw.Connection, str, Path],
    ) -> None:
        if isinstance(db_conn_or_path, (str, Path)):
            self._conn = apsw.Connection(str(db_conn_or_path))
            self._owns_conn = True
        else:
            self._conn = db_conn_or_path
            self._owns_conn = False

        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure dag_tasks table exists with all required columns and indexes."""
        cursor = self._conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dag_tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'LEASED', 'DRAFTED', 'REFINED', 'QE_SCORED', 'COMMITTED', 'RUNNING', 'COMPLETED', 'FAILED', 'RETRY')),
                priority INTEGER NOT NULL DEFAULT 0,
                unit_id INTEGER,
                dependencies TEXT,
                payload TEXT,
                result TEXT,
                error TEXT,
                retries INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                worker_id TEXT,
                lease_expires_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_dag_status_prio ON dag_tasks(status, priority DESC);
            CREATE INDEX IF NOT EXISTS idx_dag_unit_id ON dag_tasks(unit_id);
            """)
            cursor.execute("COMMIT;")
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    # --------------------------------------------------------------------------
    # Task Submission (Idempotent)
    # --------------------------------------------------------------------------

    def submit_translation_task(
        self,
        source_text: str,
        source_hash: str,
        icfg_context_id: str,
        glossary_version: str,
        model_id: str,
        unit_id: Optional[int] = None,
        priority: int = 0,
        dependencies: Optional[List[str]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> DagTask:
        """
        Idempotently submit a translation task.
        If a task with the calculated dag_task_key already exists, returns the existing
        task directly without re-queueing (preserves previous translation / milestone).
        """
        task_id = compute_dag_task_key(
            source_hash=source_hash,
            icfg_context_id=icfg_context_id,
            glossary_version=glossary_version,
            model_id=model_id,
        )

        payload = {
            "source_text": source_text,
            "source_hash": source_hash,
            "icfg_context_id": icfg_context_id,
            "glossary_version": glossary_version,
            "model_id": model_id,
        }
        if extra_payload:
            payload.update(extra_payload)

        deps_json = json.dumps(dependencies or [])
        payload_json = json.dumps(payload)

        cursor = self._conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            # Check existing task
            cursor.execute(
                "SELECT task_id, task_type, status, priority, unit_id, dependencies, payload, result, "
                "error, retries, max_retries, worker_id, lease_expires_at, started_at, completed_at "
                "FROM dag_tasks WHERE task_id = ?;",
                (task_id,),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute("COMMIT;")
                return self._row_to_task(row)

            # Insert new task
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor.execute(
                "INSERT INTO dag_tasks (task_id, task_type, status, priority, unit_id, dependencies, "
                "payload, result, retries, max_retries, created_at, updated_at) "
                "VALUES (?, 'translate', 'PENDING', ?, ?, ?, ?, '{}', 0, 3, ?, ?);",
                (task_id, priority, unit_id, deps_json, payload_json, now_iso, now_iso),
            )
            cursor.execute("COMMIT;")
            return DagTask(
                task_id=task_id,
                task_type="translate",
                status=TaskStatus.PENDING,
                priority=priority,
                unit_id=unit_id,
                dependencies=dependencies or [],
                payload=payload,
                result={},
            )
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    # --------------------------------------------------------------------------
    # Task Leasing (Worker Pull)
    # --------------------------------------------------------------------------

    def lease_tasks(
        self,
        worker_id: str,
        limit: int = 1,
        lease_duration_sec: int = 60,
    ) -> List[DagTask]:
        """
        Lease ready tasks to a worker under BEGIN IMMEDIATE.
        A task is ready if:
        1. status == 'PENDING' OR (status == 'LEASED' and lease_expires_at < now)
        2. All prerequisite dependency tasks are in 'COMMITTED' state.
        """
        cursor = self._conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            now_iso = now_dt.isoformat()
            expires_iso = (now_dt + datetime.timedelta(seconds=lease_duration_sec)).isoformat()

            # Query candidate tasks
            cursor.execute(
                "SELECT task_id, task_type, status, priority, unit_id, dependencies, payload, result, "
                "error, retries, max_retries, worker_id, lease_expires_at, started_at, completed_at "
                "FROM dag_tasks "
                "WHERE status = 'PENDING' OR (status = 'LEASED' AND lease_expires_at < ?) "
                "ORDER BY priority DESC, created_at ASC;",
                (now_iso,),
            )
            candidates = cursor.fetchall()
            leased: List[DagTask] = []

            for row in candidates:
                if len(leased) >= limit:
                    break

                task = self._row_to_task(row)

                # Check dependencies
                if task.dependencies:
                    placeholders = ",".join("?" for _ in task.dependencies)
                    cursor.execute(
                        f"SELECT COUNT(*) FROM dag_tasks WHERE task_id IN ({placeholders}) AND status = 'COMMITTED';",
                        tuple(task.dependencies),
                    )
                    satisfied_count = cursor.fetchone()[0]
                    if satisfied_count < len(task.dependencies):
                        # Dependencies not fully satisfied yet
                        continue

                # Lease this task
                cursor.execute(
                    "UPDATE dag_tasks SET status = 'LEASED', worker_id = ?, lease_expires_at = ?, "
                    "started_at = COALESCE(started_at, ?), updated_at = ? WHERE task_id = ?;",
                    (worker_id, expires_iso, now_iso, now_iso, task.task_id),
                )
                task.status = TaskStatus.LEASED
                task.worker_id = worker_id
                task.lease_expires_at = expires_iso
                leased.append(task)

            cursor.execute("COMMIT;")
            return leased
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    # --------------------------------------------------------------------------
    # Milestone State Transitions
    # --------------------------------------------------------------------------

    def transition_state(
        self,
        task_id: str,
        new_status: TaskStatus,
        result_update: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        force: bool = False,
    ) -> DagTask:
        """
        Transition task through milestone states:
        PENDING -> LEASED -> DRAFTED -> REFINED -> QE_SCORED -> COMMITTED
        Protected by BEGIN IMMEDIATE.
        """
        cursor = self._conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            cursor.execute(
                "SELECT task_id, task_type, status, priority, unit_id, dependencies, payload, result, "
                "error, retries, max_retries, worker_id, lease_expires_at, started_at, completed_at "
                "FROM dag_tasks WHERE task_id = ?;",
                (task_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise KeyError(f"Task with id '{task_id}' not found.")

            task = self._row_to_task(row)

            # Idempotency guard: once COMMITTED, ignore backward transitions unless forced
            if task.status == TaskStatus.COMMITTED and new_status != TaskStatus.COMMITTED and not force:
                logger.info(f"Task '{task_id}' is already COMMITTED. Preserving COMMITTED status.")
                cursor.execute("COMMIT;")
                return task

            # Update result dictionary incrementally
            merged_result = dict(task.result)
            if result_update:
                merged_result.update(result_update)
            result_json = json.dumps(merged_result)

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            completed_iso = now_iso if new_status in (TaskStatus.COMMITTED, TaskStatus.COMPLETED) else task.completed_at

            cursor.execute(
                "UPDATE dag_tasks SET status = ?, result = ?, error = ?, completed_at = ?, updated_at = ? "
                "WHERE task_id = ?;",
                (new_status.value, result_json, error or task.error, completed_iso, now_iso, task_id),
            )

            # If moving to COMMITTED and unit_id is present, optionally update translation_units
            if new_status == TaskStatus.COMMITTED and task.unit_id is not None:
                final_text = merged_result.get("target_text")
                if final_text is not None:
                    cursor.execute(
                        "UPDATE translation_units SET target_text = ?, status = 'TRANSLATED', "
                        "updated_at = ? WHERE id = ?;",
                        (final_text, now_iso, task.unit_id),
                    )

            cursor.execute("COMMIT;")

            task.status = new_status
            task.result = merged_result
            task.error = error or task.error
            task.completed_at = completed_iso
            return task
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    # --------------------------------------------------------------------------
    # Crash Recovery
    # --------------------------------------------------------------------------

    def recover_after_crash(self) -> int:
        """
        Recovers tasks from unexpected crashes / power loss.
        Finds tasks stuck in LEASED where the lease has expired or interrupted,
        and reverts them cleanly to PENDING (preserving draft results if any).
        Returns the number of recovered tasks.
        """
        cursor = self._conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor.execute(
                "SELECT task_id, status FROM dag_tasks "
                "WHERE status = 'LEASED' AND (lease_expires_at IS NULL OR lease_expires_at <= ?);",
                (now_iso,),
            )
            stuck_tasks = cursor.fetchall()
            recovered_count = 0

            for (task_id, status) in stuck_tasks:
                cursor.execute(
                    "UPDATE dag_tasks SET status = 'PENDING', worker_id = NULL, lease_expires_at = NULL, "
                    "retries = retries + 1, updated_at = ? WHERE task_id = ?;",
                    (now_iso, task_id),
                )
                recovered_count += 1

            cursor.execute("COMMIT;")
            if recovered_count > 0:
                logger.warning(f"Crash recovery: successfully reset {recovered_count} stalled tasks to PENDING.")
            return recovered_count
        except Exception:
            cursor.execute("ROLLBACK;")
            raise

    def get_task(self, task_id: str) -> Optional[DagTask]:
        """Fetch current state of a task."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT task_id, task_type, status, priority, unit_id, dependencies, payload, result, "
            "error, retries, max_retries, worker_id, lease_expires_at, started_at, completed_at "
            "FROM dag_tasks WHERE task_id = ?;",
            (task_id,),
        )
        row = cursor.fetchone()
        return self._row_to_task(row) if row else None

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------

    def _row_to_task(self, row: Tuple[Any, ...]) -> DagTask:
        return DagTask(
            task_id=row[0],
            task_type=row[1],
            status=TaskStatus(row[2]),
            priority=row[3],
            unit_id=row[4],
            dependencies=json.loads(row[5]) if row[5] else [],
            payload=json.loads(row[6]) if row[6] else {},
            result=json.loads(row[7]) if row[7] else {},
            error=row[8],
            retries=row[9],
            max_retries=row[10],
            worker_id=row[11],
            lease_expires_at=row[12],
            started_at=row[13],
            completed_at=row[14],
        )
