"""Guarded operational-storage phase transitions and runtime read routing.

``shadow`` remains the boot/default phase.  Promotion is explicit, transactional
and evidence-backed: every projected session is compared with the legacy source
before ``prefer_v2``; ``v2`` additionally requires every active session to be
fully eligible.  Rollback to shadow/legacy never removes normalized data.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import aiosqlite

from .repository import (
    ProjectionVerification,
    _aiosqlite_raw,
    operational_storage_available,
    projection_coverage,
    record_session_verification,
    session_v2_is_verified,
    tombstone_session,
    verify_session_projection,
)


PHASES = ("legacy", "shadow", "prefer_v2", "v2")
_FORWARD = {
    "legacy": "shadow",
    "shadow": "prefer_v2",
    "prefer_v2": "v2",
}


class StoragePhaseError(RuntimeError):
    """A requested promotion could not satisfy its correctness gates."""


@dataclass(frozen=True)
class PhaseTransition:
    from_phase: str
    to_phase: str
    changed: bool
    verified_sessions: int
    v2_eligible_sessions: int
    writer_epoch: int
    reason: str | None = None


def storage_phase(conn: Any) -> str:
    # ``SqliteDb`` is also used standalone in tests, tools and embedded
    # integrations that never run ``MemoryDB.connect()`` (the owner of the
    # additive v2 migration).  Absence of the normalized schema is the legacy
    # phase, not a read failure: runtime history must remain compatible.
    if not operational_storage_available(conn):
        return "legacy"
    row = conn.execute(
        "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
    ).fetchone()
    phase = str(row[0]) if row is not None else "legacy"
    if phase not in PHASES:
        raise StoragePhaseError(f"unknown operational storage phase {phase!r}")
    return phase


def preferred_session_source(conn: Any, session_id: str) -> str:
    """Return ``v2`` only when this exact session revision is verified."""

    phase = storage_phase(conn)
    if phase not in {"prefer_v2", "v2"}:
        return "legacy"
    # Fail closed per session even in v2. A downgrade/direct legacy writer can
    # arrive while the beta process is alive; the next boot guard demotes the
    # global phase, while this request immediately retains readable history.
    return "v2" if session_v2_is_verified(conn, session_id) else "legacy"


def _drift_reason(conn: Any, *, verify_content: bool = True) -> str | None:
    pending = conn.execute(
        "SELECT 1 FROM legacy_session_changes "
        "WHERE processed_at_ms IS NULL LIMIT 1"
    ).fetchone()
    if pending is not None:
        return "pending_legacy_change"
    missing = conn.execute(
        "SELECT 1 FROM sessions legacy WHERE NOT EXISTS ("
        "SELECT 1 FROM sessions_v2 projected WHERE projected.id=legacy.session_id "
        "AND projected.deleted_at_ms IS NULL) LIMIT 1"
    ).fetchone()
    if missing is not None:
        return "legacy_session_missing_from_v2"
    extra = conn.execute(
        "SELECT 1 FROM sessions_v2 projected WHERE projected.deleted_at_ms IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM sessions legacy "
        "WHERE legacy.session_id=projected.id) LIMIT 1"
    ).fetchone()
    if extra is not None:
        return "v2_session_missing_from_legacy"
    # Triggers are the normal downgrade detector, but a sufficiently old binary
    # can replace the legacy table (and therefore lose those triggers).  A
    # promoted boot pays the deliberate one-time comparison cost so a missing
    # trigger can never leave stale v2 history silently canonical.
    if verify_content:
        for row in conn.execute(
            "SELECT session_id FROM sessions ORDER BY session_id"
        ).fetchall():
            try:
                verification = verify_session_projection(conn, str(row[0]))
            except Exception as exc:
                return f"projection_drift:{type(exc).__name__}"
            if not verification.matches:
                return f"projection_drift:{verification.reason or 'unknown'}"
    return None


def _next_writer_epoch(conn: Any, now_ms: int) -> int:
    row = conn.execute(
        "UPDATE operational_storage_state SET writer_epoch=writer_epoch+1, "
        "updated_at_ms=? WHERE singleton_id=1 RETURNING writer_epoch",
        (now_ms,),
    ).fetchone()
    if row is None:
        raise StoragePhaseError("operational storage state is missing")
    return int(row[0])


def _queue_projection_drift(conn: Any, now_ms: int) -> tuple[int, int]:
    """Repair lost-trigger set drift and enqueue content mismatches.

    Legacy remains canonical throughout beta. A legacy-only id is queued for
    projection; a v2-only id is tombstoned immediately. This lets a promoted
    database recover even when an old binary replaced the trigger-bearing
    legacy table instead of leaving the normal durable journal evidence.
    """

    queued = 0
    for row in conn.execute(
        "SELECT session_id, updated_at FROM sessions ORDER BY session_id"
    ).fetchall():
        session_id = str(row[0])
        try:
            verification = verify_session_projection(
                conn, session_id, now_ms=now_ms
            )
        except Exception:
            verification = None
        if verification is not None and verification.matches:
            continue
        conn.execute(
            "UPDATE sessions_v2 SET legacy_source_hash=NULL WHERE id=?",
            (session_id,),
        )
        pending = conn.execute(
            "SELECT 1 FROM legacy_session_changes WHERE session_id=? "
            "AND processed_at_ms IS NULL LIMIT 1",
            (session_id,),
        ).fetchone()
        if pending is None:
            conn.execute(
                "INSERT INTO legacy_session_changes "
                "(session_id, operation, legacy_updated_at, observed_at_ms) "
                "VALUES (?, 'update', ?, ?)",
                (session_id, row[1], now_ms),
            )
            queued += 1
    tombstoned = 0
    extra_ids = [
        str(row[0])
        for row in conn.execute(
            "SELECT projected.id FROM sessions_v2 projected "
            "WHERE projected.deleted_at_ms IS NULL AND NOT EXISTS ("
            "SELECT 1 FROM sessions legacy WHERE legacy.session_id=projected.id) "
            "ORDER BY projected.id"
        ).fetchall()
    ]
    for session_id in extra_ids:
        if tombstone_session(conn, session_id, now_ms=now_ms).changed:
            tombstoned += 1
    return queued, tombstoned


def _append_phase_event(
    conn: Any,
    *,
    from_phase: str,
    to_phase: str,
    writer_version: str,
    writer_epoch: int,
    now_ms: int,
    details: dict[str, Any],
) -> None:
    conn.execute(
        "INSERT INTO storage_migration_journal "
        "(migration_id, event_type, from_phase, to_phase, writer_version, "
        "writer_epoch, details_json, occurred_at_ms) "
        "VALUES ('operational-storage-v2', 'phase_changed', ?, ?, ?, ?, ?, ?)",
        (
            from_phase,
            to_phase,
            writer_version,
            writer_epoch,
            json.dumps(details, sort_keys=True, separators=(",", ":")),
            now_ms,
        ),
    )


def _apply_phase(
    conn: Any,
    *,
    from_phase: str,
    to_phase: str,
    writer_version: str,
    now_ms: int,
    details: dict[str, Any],
) -> int:
    writer_epoch = _next_writer_epoch(conn, now_ms)
    max_change = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) FROM legacy_session_changes "
        "WHERE processed_at_ms IS NOT NULL"
    ).fetchone()
    changed = conn.execute(
        "UPDATE storage_migration_state SET phase=?, state_version=state_version+1, "
        "last_applied_legacy_change_seq=?, last_writer_version=?, "
        "last_writer_epoch=?, updated_at_ms=? WHERE singleton_id=1 AND phase=?",
        (
            to_phase,
            int(max_change[0] if max_change else 0),
            writer_version,
            writer_epoch,
            now_ms,
            from_phase,
        ),
    )
    if int(changed.rowcount) != 1:
        raise StoragePhaseError("storage phase changed concurrently")
    _append_phase_event(
        conn,
        from_phase=from_phase,
        to_phase=to_phase,
        writer_version=writer_version,
        writer_epoch=writer_epoch,
        now_ms=now_ms,
        details=details,
    )
    return writer_epoch


def guard_storage_phase(
    conn: Any,
    *,
    writer_version: str,
    now_ms: int | None = None,
    audit_content: bool = True,
) -> PhaseTransition:
    """Demote a promoted database to shadow when downgrade drift is observed."""

    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    current = storage_phase(conn)
    if current not in {"prefer_v2", "v2"}:
        epoch_row = conn.execute(
            "SELECT writer_epoch FROM operational_storage_state WHERE singleton_id=1"
        ).fetchone()
        return PhaseTransition(
            current, current, False, 0, 0, int(epoch_row[0] if epoch_row else 0)
        )
    reason = _drift_reason(conn, verify_content=audit_content)
    if reason is None:
        epoch_row = conn.execute(
            "SELECT writer_epoch FROM operational_storage_state WHERE singleton_id=1"
        ).fetchone()
        return PhaseTransition(
            current, current, False, 0, 0, int(epoch_row[0] if epoch_row else 0)
        )
    queued, tombstoned = (
        _queue_projection_drift(conn, effective_now)
        if reason != "pending_legacy_change"
        else (0, 0)
    )
    epoch = _apply_phase(
        conn,
        from_phase=current,
        to_phase="shadow",
        writer_version=writer_version,
        now_ms=effective_now,
        details={
            "automatic_rollback": True,
            "reason": reason,
            "requeued_sessions": queued,
            "tombstoned_sessions": tombstoned,
        },
    )
    return PhaseTransition(current, "shadow", True, 0, 0, epoch, reason)


def transition_storage_phase(
    conn: Any,
    target: str,
    *,
    writer_version: str,
    now_ms: int | None = None,
) -> PhaseTransition:
    """Apply one guarded forward transition or a non-destructive rollback."""

    if target not in PHASES:
        raise StoragePhaseError(f"invalid operational storage phase {target!r}")
    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    current = storage_phase(conn)
    epoch_row = conn.execute(
        "SELECT writer_epoch FROM operational_storage_state WHERE singleton_id=1"
    ).fetchone()
    current_epoch = int(epoch_row[0] if epoch_row else 0)
    if current == target:
        return PhaseTransition(current, target, False, 0, 0, current_epoch)

    rollback = target in {"legacy", "shadow"} and PHASES.index(target) < PHASES.index(current)
    if not rollback and _FORWARD.get(current) != target:
        raise StoragePhaseError(
            f"storage phase must advance one step; cannot move {current} -> {target}"
        )
    if rollback:
        epoch = _apply_phase(
            conn,
            from_phase=current,
            to_phase=target,
            writer_version=writer_version,
            now_ms=effective_now,
            details={"rollback": True},
        )
        return PhaseTransition(current, target, True, 0, 0, epoch, "operator_rollback")

    if current == "legacy":
        # The schema migrator normally owns this edge; retain it here for an
        # explicit recovery of a previously installed additive schema.
        epoch = _apply_phase(
            conn,
            from_phase=current,
            to_phase=target,
            writer_version=writer_version,
            now_ms=effective_now,
            details={"schema_ready": True},
        )
        return PhaseTransition(current, target, True, 0, 0, epoch)

    coverage = projection_coverage(conn)
    if not bool(coverage["complete"]):
        raise StoragePhaseError(
            "cannot promote operational storage while projection coverage is incomplete"
        )
    drift = _drift_reason(conn, verify_content=False)
    if drift is not None:
        raise StoragePhaseError(f"cannot promote operational storage: {drift}")

    session_ids = [
        str(row[0])
        for row in conn.execute(
            "SELECT session_id FROM sessions ORDER BY session_id"
        ).fetchall()
    ]
    verifications: list[ProjectionVerification] = []
    for session_id in session_ids:
        verification = verify_session_projection(
            conn, session_id, now_ms=effective_now
        )
        if not verification.matches:
            raise StoragePhaseError(
                f"session {session_id!r} failed v2 verification: {verification.reason}"
            )
        verifications.append(verification)
    eligible = sum(1 for item in verifications if item.eligible_for_v2)
    if target == "v2" and eligible != len(verifications):
        raise StoragePhaseError(
            "cannot enter v2 while one or more sessions require legacy fallback"
        )

    epoch = _next_writer_epoch(conn, effective_now)
    for verification in verifications:
        record_session_verification(
            conn,
            verification,
            writer_version=writer_version,
            writer_epoch=epoch,
            now_ms=effective_now,
        )
    source_digest = hashlib.sha256(
        "\n".join(
            f"{item.session_id}:{item.source_hash}:{item.source_version}"
            for item in verifications
        ).encode("utf-8")
    ).hexdigest()
    max_change = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) FROM legacy_session_changes "
        "WHERE processed_at_ms IS NOT NULL"
    ).fetchone()
    changed = conn.execute(
        "UPDATE storage_migration_state SET phase=?, state_version=state_version+1, "
        "source_hash=?, last_applied_legacy_change_seq=?, last_writer_version=?, "
        "last_writer_epoch=?, updated_at_ms=? WHERE singleton_id=1 AND phase=?",
        (
            target,
            source_digest,
            int(max_change[0] if max_change else 0),
            writer_version,
            epoch,
            effective_now,
            current,
        ),
    )
    if int(changed.rowcount) != 1:
        raise StoragePhaseError("storage phase changed concurrently")
    _append_phase_event(
        conn,
        from_phase=current,
        to_phase=target,
        writer_version=writer_version,
        writer_epoch=epoch,
        now_ms=effective_now,
        details={
            "verified_sessions": len(verifications),
            "v2_eligible_sessions": eligible,
            "source_hash": source_digest,
        },
    )
    return PhaseTransition(
        current,
        target,
        True,
        len(verifications),
        eligible,
        epoch,
    )


def transition_storage_phase_atomically(
    conn: Any,
    target: str,
    *,
    writer_version: str,
) -> PhaseTransition:
    """Verify and change phase inside one writer-locked SQLite operation.

    The shared aiosqlite worker executes this whole function without yielding
    to another coroutine. ``BEGIN IMMEDIATE`` also prevents another process or
    connection from mutating legacy/v2 state between parity verification and
    the phase CAS. Refuse to inherit an unrelated caller transaction rather
    than committing or rolling it back accidentally.
    """

    if bool(getattr(conn, "in_transaction", False)):
        raise StoragePhaseError(
            "cannot change storage phase inside an existing transaction"
        )
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = transition_storage_phase(
            conn,
            target,
            writer_version=writer_version,
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return result


def guard_storage_phase_atomically(
    conn: Any,
    *,
    writer_version: str,
    audit_content: bool,
) -> PhaseTransition:
    """Run the promoted boot guard in its own writer-locked transaction."""

    if bool(getattr(conn, "in_transaction", False)):
        raise StoragePhaseError(
            "cannot guard storage phase inside an existing transaction"
        )
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = guard_storage_phase(
            conn,
            writer_version=writer_version,
            audit_content=audit_content,
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return result


async def guard_storage_phase_async(
    conn: aiosqlite.Connection,
    *,
    writer_version: str,
    audit_content: bool = True,
) -> PhaseTransition:
    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(
        guard_storage_phase,
        _aiosqlite_raw(conn),
        writer_version=writer_version,
        audit_content=audit_content,
    )


async def guard_storage_phase_atomic_async(
    conn: aiosqlite.Connection,
    *,
    writer_version: str,
    audit_content: bool,
) -> PhaseTransition:
    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(
        guard_storage_phase_atomically,
        _aiosqlite_raw(conn),
        writer_version=writer_version,
        audit_content=audit_content,
    )


async def preferred_session_source_async(
    conn: aiosqlite.Connection,
    session_id: str,
) -> str:
    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(
        preferred_session_source,
        _aiosqlite_raw(conn),
        session_id,
    )


async def transition_storage_phase_async(
    conn: aiosqlite.Connection,
    target: str,
    *,
    writer_version: str,
) -> PhaseTransition:
    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(
        transition_storage_phase_atomically,
        _aiosqlite_raw(conn),
        target,
        writer_version=writer_version,
    )


async def transition_storage_phase_atomic_async(
    conn: aiosqlite.Connection,
    target: str,
    *,
    writer_version: str,
) -> PhaseTransition:
    return await transition_storage_phase_async(
        conn,
        target,
        writer_version=writer_version,
    )
