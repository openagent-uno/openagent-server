#!/usr/bin/env python3
"""Create a hermetic, credential-free local E2E clone of an agent.

The source is opened read-only and copied with SQLite's online backup API, so
WAL state is included consistently.  Identity, coordinator state, logs,
derived indexes and channel credentials are intentionally not copied.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import yaml


_COPY_DIRS = ("memories", "artifacts", "ui")
_IDENTITY_TABLES = (
    "device_certs",
    "network_users",
    "network_devices",
    "network_invitations",
    "network_agents",
    "peer_networks",
)
_REQUEST_TABLES = (
    "task_run_requests",
    "workflow_run_requests",
)


def _inside_temp(path: Path) -> bool:
    root = Path(tempfile.gettempdir()).resolve()
    try:
        path.resolve(strict=False).relative_to(root)
        return path.resolve(strict=False) != root
    except ValueError:
        return False


def _copy_tree_no_symlinks(source: Path, destination: Path) -> None:
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeError(f"fixture input contains a symlink: {candidate}")
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        ignore=shutil.ignore_patterns(".git", ".DS_Store", "*.db", "*.db-*"),
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _sanitize_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=60000")
        for table in _IDENTITY_TABLES:
            if _table_exists(conn, table):
                conn.execute(f'DELETE FROM "{table}"')
        for table in _REQUEST_TABLES:
            if _table_exists(conn, table):
                conn.execute(f'DELETE FROM "{table}"')

        if _table_exists(conn, "network"):
            cols = _columns(conn, "network")
            assignments = []
            if "role" in cols:
                assignments.append("role='standalone'")
            for col in ("coordinator_node_id", "coordinator_pubkey"):
                if col in cols:
                    assignments.append(f'"{col}"=NULL')
            if assignments:
                conn.execute(f"UPDATE network SET {', '.join(assignments)}")

        if _table_exists(conn, "providers"):
            cols = _columns(conn, "providers")
            assignments = []
            if "api_key" in cols:
                assignments.append("api_key=NULL")
            if "enabled" in cols:
                assignments.append("enabled=0")
            if assignments:
                conn.execute(f"UPDATE providers SET {', '.join(assignments)}")
        if _table_exists(conn, "models") and "enabled" in _columns(conn, "models"):
            conn.execute("UPDATE models SET enabled=0")

        if _table_exists(conn, "mcps"):
            cols = _columns(conn, "mcps")
            assignments = []
            if "env_json" in cols:
                assignments.append("env_json='{}'")
            if "headers_json" in cols:
                assignments.append("headers_json='{}'")
            if "oauth" in cols:
                assignments.append("oauth=0")
            if "enabled" in cols:
                assignments.append("enabled=0")
            if assignments:
                conn.execute(f"UPDATE mcps SET {', '.join(assignments)}")

        for table in ("scheduled_tasks", "events", "workflow_tasks", "workflow_schedules"):
            if _table_exists(conn, table) and "enabled" in _columns(conn, table):
                conn.execute(f'UPDATE "{table}" SET enabled=0')
        if _table_exists(conn, "events"):
            cols = _columns(conn, "events")
            assignments = []
            if "secret_enc" in cols:
                assignments.append("secret_enc=''")
            if "secret_hint" in cols:
                assignments.append("secret_hint=NULL")
            if assignments:
                conn.execute(f"UPDATE events SET {', '.join(assignments)}")

        conn.commit()
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("sanitized fixture database failed quick_check")
        # The copied DB must not carry obvious credential-bearing fields.
        if _table_exists(conn, "providers") and conn.execute(
            "SELECT 1 FROM providers WHERE api_key IS NOT NULL AND api_key<>'' LIMIT 1"
        ).fetchone():
            raise RuntimeError("provider credential remained in fixture")
        if _table_exists(conn, "mcps") and conn.execute(
            "SELECT 1 FROM mcps WHERE env_json<>'{}' OR headers_json<>'{}' LIMIT 1"
        ).fetchone():
            raise RuntimeError("MCP credential material remained in fixture")
    finally:
        conn.close()


def _sanitized_config(source: Path, destination: Path) -> None:
    raw: Any = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RuntimeError("agent config must be a mapping")
    config = dict(raw)
    config["name"] = f"{str(config.get('name') or 'openagent')}-local-e2e"
    config["local_e2e_fixture"] = True
    config["auto_update"] = {"enabled": False, "mode": "manual"}
    config["channels"] = {}
    # Provider ids are harmless and useful for read-only settings QA; actual
    # credentials and enabled flags are removed from the database above.
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.chmod(destination, 0o600)


def prepare(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve(strict=False)
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError("source agent directory must be a real directory")
    if not _inside_temp(destination):
        raise RuntimeError("destination must be a child of the OS temporary directory")
    if destination.exists():
        raise RuntimeError("destination already exists")
    source_db = source / "openagent.db"
    source_config = source / "openagent.yaml"
    if not source_db.is_file() or source_db.is_symlink():
        raise RuntimeError("source openagent.db is missing or unsafe")
    if not source_config.is_file() or source_config.is_symlink():
        raise RuntimeError("source openagent.yaml is missing or unsafe")

    destination.mkdir(mode=0o700, parents=True)
    try:
        _sanitized_config(source_config, destination / "openagent.yaml")
        source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        destination_conn = sqlite3.connect(destination / "openagent.db")
        try:
            if source_conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("source database failed quick_check")
            source_conn.backup(destination_conn)
            destination_conn.commit()
        finally:
            destination_conn.close()
            source_conn.close()
        os.chmod(destination / "openagent.db", 0o600)
        _sanitize_database(destination / "openagent.db")
        for name in _COPY_DIRS:
            candidate = source / name
            if candidate.exists():
                if not candidate.is_dir() or candidate.is_symlink():
                    raise RuntimeError(f"unsafe fixture directory: {candidate}")
                _copy_tree_no_symlinks(candidate, destination / name)
        return destination
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        tempfile.mkdtemp(prefix="openagent-local-e2e-")
    )
    # mkdtemp creates the directory; ``prepare`` deliberately refuses an
    # existing target. Remove this brand-new empty leaf without widening scope.
    if args.output is None:
        output.rmdir()
    result = prepare(args.source, output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
