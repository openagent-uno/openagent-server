"""Immutable, human-readable on-disk bundles for Custom View revisions."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping
import uuid


MAX_BUNDLE_FILES = 256
MAX_SCRIPT_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BundleEvidence:
    path: str
    sha256: str
    size_bytes: int


def safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError("invalid Custom View bundle path")
    if "\\" in value or "\x00" in value:
        raise ValueError("invalid Custom View bundle path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid Custom View bundle path")
    return path.as_posix()


class ViewBundleStore:
    """Publish one checksummed, immutable directory per View revision."""

    _CORE_FILES = ("compiled.json", "manifest.json", "view.oaui")

    def __init__(self, db_path: str | Path):
        source = Path(db_path).expanduser()
        self._memory_payloads: dict[tuple[str, int], dict[str, bytes]] = {}
        if str(source) == ":memory:" or str(source).startswith("file::memory:"):
            self.root: Path | None = None
        else:
            self.root = source.resolve().parent / "ui"

    @staticmethod
    def _safe_id(view_id: str) -> str:
        if not view_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in view_id
        ):
            raise ValueError("invalid Custom View id")
        return view_id

    def revision_path(self, view_id: str, revision: int) -> Path | None:
        if self.root is None:
            return None
        safe = self._safe_id(view_id)
        if revision < 1:
            raise ValueError("revision must be positive")
        return self.root / "views" / safe / "revisions" / str(revision)

    @staticmethod
    def _source_bytes(bundle: Mapping[str, Any]) -> bytes:
        markup = bundle.get("markup")
        if isinstance(markup, str):
            return markup.encode("utf-8") + (b"" if markup.endswith("\n") else b"\n")
        return json.dumps(
            bundle.get("spec") or {}, ensure_ascii=False, sort_keys=True,
            indent=2, allow_nan=False,
        ).encode("utf-8") + b"\n"

    @staticmethod
    def _compiled_bytes(bundle: Mapping[str, Any]) -> bytes:
        return json.dumps(
            bundle.get("spec") or {}, ensure_ascii=False, sort_keys=True,
            indent=2, allow_nan=False,
        ).encode("utf-8") + b"\n"

    @staticmethod
    def _normalize_files(
        values: Mapping[str, bytes] | None,
        *,
        directory: str,
        per_file_limit: int,
    ) -> dict[str, bytes]:
        if values is None:
            return {}
        if not isinstance(values, Mapping) or len(values) > MAX_BUNDLE_FILES:
            raise ValueError("Custom View bundle contains too many files")
        result: dict[str, bytes] = {}
        for raw_name, raw_payload in values.items():
            name = safe_relative_path(str(raw_name))
            if not isinstance(raw_payload, bytes) or len(raw_payload) > per_file_limit:
                raise ValueError("Custom View bundle file exceeds its size limit")
            result[f"{directory}/{name}"] = raw_payload
        return result

    @classmethod
    def _entries(
        cls,
        bundle: Mapping[str, Any],
        *,
        scripts: Mapping[str, bytes] | None = None,
        assets: Mapping[str, bytes] | None = None,
    ) -> dict[str, bytes]:
        extra = {
            **cls._normalize_files(
                scripts, directory="scripts", per_file_limit=MAX_SCRIPT_BYTES,
            ),
            **cls._normalize_files(
                assets, directory="assets", per_file_limit=MAX_ASSET_BYTES,
            ),
        }
        file_manifest = [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sizeBytes": len(payload),
            }
            for name, payload in sorted(extra.items())
        ]
        manifest = {
            key: value for key, value in bundle.items()
            if key not in {"markup", "spec"}
        }
        manifest["sourceKind"] = (
            "markup" if isinstance(bundle.get("markup"), str) else "spec"
        )
        manifest["files"] = file_manifest
        entries = {
            "compiled.json": cls._compiled_bytes(bundle),
            "manifest.json": json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
            ).encode("utf-8") + b"\n",
            "view.oaui": cls._source_bytes(bundle),
            **extra,
        }
        if len(extra) > MAX_BUNDLE_FILES or sum(map(len, entries.values())) > MAX_BUNDLE_BYTES:
            raise ValueError("Custom View bundle exceeds its size limit")
        return entries

    @staticmethod
    def _digest(entries: Mapping[str, bytes]) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        for name, payload in sorted(entries.items()):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(payload)
            size += len(payload)
        return digest.hexdigest(), size

    def write_revision(
        self,
        *,
        view_id: str,
        revision: int,
        bundle: dict[str, Any],
        scripts: Mapping[str, bytes] | None = None,
        assets: Mapping[str, bytes] | None = None,
    ) -> BundleEvidence:
        entries = self._entries(bundle, scripts=scripts, assets=assets)
        digest, size = self._digest(entries)
        target = self.revision_path(view_id, revision)
        if target is None:
            evidence = BundleEvidence(
                f"memory://ui/views/{view_id}/revisions/{revision}", digest, size,
            )
            key = (view_id, revision)
            if key in self._memory_payloads:
                if not self.verify(evidence):
                    raise FileExistsError("Custom View revision bundle is immutable")
            else:
                self._memory_payloads[key] = dict(entries)
            return evidence
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            os.chmod(target.parent, 0o700)
        if target.exists():
            if not self.verify(BundleEvidence(str(target), digest, size)):
                raise FileExistsError("Custom View revision bundle is immutable")
            return BundleEvidence(str(target), digest, size)
        temporary = target.parent / f".{revision}.{uuid.uuid4().hex}.staging"
        temporary.mkdir(mode=0o700)
        try:
            (temporary / "scripts").mkdir(mode=0o700)
            (temporary / "assets").mkdir(mode=0o700)
            for name, payload in entries.items():
                path = temporary.joinpath(*PurePosixPath(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(path, 0o500 if name.startswith("scripts/") else 0o400)
            # Published revision directories have no write bit.  New revisions
            # are siblings created through staging, so they never need to
            # mutate an existing bundle in place.
            for directory in sorted(
                (item for item in temporary.rglob("*") if item.is_dir()),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                os.chmod(directory, 0o500)
            directory_fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.replace(temporary, target)
            os.chmod(target, 0o500)
            try:
                directory_fd = os.open(target.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        return BundleEvidence(str(target), digest, size)

    @classmethod
    def _disk_entries(cls, path: Path) -> dict[str, bytes] | None:
        if not path.is_dir() or path.is_symlink():
            return None
        entries: dict[str, bytes] = {}
        total = 0
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                return None
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(path).as_posix()
            if relative not in cls._CORE_FILES and not relative.startswith(
                ("scripts/", "assets/")
            ):
                return None
            try:
                size = candidate.stat().st_size
            except OSError:
                return None
            per_file_limit = (
                MAX_SCRIPT_BYTES
                if relative.startswith("scripts/")
                else MAX_ASSET_BYTES
                if relative.startswith("assets/")
                else MAX_BUNDLE_BYTES
            )
            if size < 0 or size > per_file_limit or total + size > MAX_BUNDLE_BYTES:
                return None
            payload = candidate.read_bytes()
            if len(payload) > per_file_limit or total + len(payload) > MAX_BUNDLE_BYTES:
                return None
            entries[relative] = payload
            total += len(payload)
        if any(name not in entries for name in cls._CORE_FILES):
            return None
        return entries

    def verify(self, evidence: BundleEvidence) -> bool:
        if evidence.path.startswith("memory://"):
            parts = evidence.path.rstrip("/").split("/")
            try:
                key = (parts[-3], int(parts[-1]))
            except (IndexError, ValueError):
                return False
            entries = self._memory_payloads.get(key)
        else:
            entries = self._disk_entries(Path(evidence.path))
        if entries is None:
            return False
        digest, size = self._digest(entries)
        return digest == evidence.sha256 and size == evidence.size_bytes

    def read_revision(
        self,
        evidence: BundleEvidence,
        *,
        view_id: str,
        revision: int,
    ) -> dict[str, Any]:
        if not self.verify(evidence):
            raise ValueError("Custom View bundle checksum verification failed")
        if evidence.path.startswith("memory://"):
            entries = self._memory_payloads[(view_id, revision)]
        else:
            entries = self._disk_entries(Path(evidence.path))
            if entries is None:
                raise FileNotFoundError("Custom View bundle is missing")
        manifest = json.loads(entries["manifest.json"])
        spec = json.loads(entries["compiled.json"])
        source_kind = str(manifest.pop("sourceKind", "markup"))
        markup = entries["view.oaui"].decode("utf-8") if source_kind == "markup" else None
        return {**manifest, "markup": markup, "spec": spec}

    def read_files(
        self,
        evidence: BundleEvidence,
        *,
        view_id: str,
        revision: int,
        directory: str,
    ) -> dict[str, bytes]:
        if directory not in {"scripts", "assets"} or not self.verify(evidence):
            raise ValueError("Custom View bundle checksum verification failed")
        if evidence.path.startswith("memory://"):
            entries = self._memory_payloads[(view_id, revision)]
        else:
            entries = self._disk_entries(Path(evidence.path))
            if entries is None:
                raise FileNotFoundError("Custom View bundle is missing")
        prefix = f"{directory}/"
        return {
            name[len(prefix):]: payload
            for name, payload in entries.items()
            if name.startswith(prefix)
        }


__all__ = [
    "BundleEvidence", "MAX_ASSET_BYTES", "MAX_BUNDLE_BYTES", "MAX_BUNDLE_FILES",
    "MAX_SCRIPT_BYTES", "ViewBundleStore", "safe_relative_path",
]
