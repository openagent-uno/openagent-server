"""Shared helpers for the webhook Events channel: secret generation,
encryption-at-rest, verification, and slug derivation.

Kept dependency-light and outside the DB layer so the three writers of the
``events`` table use identical logic: the in-process ``MemoryDB`` (REST path),
the ``events-manager`` MCP subprocess (raw ``aiosqlite``), and the webhook
auth path that verifies an inbound request.

Why encryption, not a one-way hash:
    Provider webhooks (GitHub / Stripe / Slack / generic-HMAC) authenticate by
    HMAC-ing the raw request body with the shared secret. Verifying that HMAC
    requires the secret *in clear* — a one-way hash cannot reproduce it. So the
    per-event secret is stored ENCRYPTED at rest (Fernet / AES-128-CBC+HMAC)
    under a key held in a 0600 file next to the agent DB. This satisfies the
    "never plaintext at rest" intent while keeping provider HMAC verification
    possible. Only a short ``hint`` (last 4 chars) is kept unencrypted, for the
    UI's ``whsec_…abcd`` display.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet

SECRET_PREFIX = "whsec_"
_KEY_FILENAME = "events.key"

# Process-wide fallback key for ``:memory:`` / path-less DBs (tests). Never
# written to disk; fine because such DBs don't outlive the process.
_ephemeral_key: Optional[bytes] = None


def _resolve_key_path(db_path: Optional[str]) -> Optional[Path]:
    """The ``events.key`` file lives next to the agent DB so the main process
    and the MCP subprocess (both keyed on the same DB path) share it. Returns
    None for in-memory / path-less DBs."""
    if not db_path or db_path == ":memory:" or db_path.startswith("file::memory:"):
        return None
    return Path(db_path).expanduser().resolve().parent / _KEY_FILENAME


def _load_or_create_key(db_path: Optional[str]) -> bytes:
    global _ephemeral_key
    key_path = _resolve_key_path(db_path)
    if key_path is None:
        if _ephemeral_key is None:
            _ephemeral_key = Fernet.generate_key()
        return _ephemeral_key
    if key_path.exists():
        return key_path.read_bytes().strip()
    key = Fernet.generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    # 0600 — the key decrypts every event secret; treat it like the identity key.
    with open(os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(key)
    return key


def _cipher(db_path: Optional[str]) -> Fernet:
    return Fernet(_load_or_create_key(db_path))


def generate_secret() -> str:
    """A fresh URL-safe per-event secret, shown to the user exactly once."""
    return SECRET_PREFIX + secrets.token_urlsafe(32)


def encrypt_secret(clear: str, *, db_path: Optional[str]) -> str:
    """Encrypt a clear secret for storage. Returns a base64 token string."""
    return _cipher(db_path).encrypt(clear.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str, *, db_path: Optional[str]) -> str:
    """Recover the clear secret from its stored token."""
    return _cipher(db_path).decrypt(token.encode("ascii")).decode("utf-8")


def verify_secret(presented: str, *, secret_enc: str, db_path: Optional[str]) -> bool:
    """Constant-time check that ``presented`` matches the stored secret."""
    if not presented or not secret_enc:
        return False
    try:
        clear = decrypt_secret(secret_enc, db_path=db_path)
    except Exception:  # noqa: BLE001 — bad token / wrong key → not authentic
        return False
    return secrets.compare_digest(presented, clear)


def secret_hint(secret: str) -> str:
    """Last 4 chars of the clear secret, for the UI's ``…abcd`` display."""
    return secret[-4:] if len(secret) >= 4 else secret


def make_secret_material(*, db_path: Optional[str]) -> tuple[str, str, str]:
    """Return ``(clear_secret, secret_enc, secret_hint)`` for a brand-new
    secret. The clear value is returned once (to show the user) and never
    stored — only its encrypted form is persisted."""
    clear = generate_secret()
    return clear, encrypt_secret(clear, db_path=db_path), secret_hint(clear)


# ── slug ──

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Derive a URL-safe slug from an event name. Collisions are resolved by
    the caller (``events.slug`` is UNIQUE)."""
    base = _SLUG_STRIP.sub("-", (name or "").lower()).strip("-")
    return base or "event"


def random_slug_suffix() -> str:
    return secrets.token_hex(3)


# Deprecated hash helpers kept as no-ops for any stale import path; the schema
# no longer stores a salted hash. (Unused after the encryption switch.)
def _legacy_hash(secret: str, salt: str) -> str:  # pragma: no cover
    return hashlib.sha256((salt + secret).encode("utf-8")).hexdigest()
