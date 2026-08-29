"""Long-lived Ed25519 identity for an agent or a user device.

Each agent has one identity stored at ``<agent_dir>/identity.key``.
Each CLI/app install carries a separate user-device identity at
``~/.openagent/user/identity.key``. The keys are 32-byte Ed25519 secret
seeds — the same key bytes feed Iroh's NodeId derivation and the
device-cert signing layer (``network.auth.device_cert``).

The file is written 0600 with an atomic rename so a partial write can't
leave a half-written key on disk. Re-reads are cheap (~50 µs) but
callers should still cache the parsed Identity object since it carries
state Iroh's Endpoint binds to.
"""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)


SECRET_KEY_LEN = 32


@dataclass(frozen=True)
class Identity:
    """An Ed25519 keypair plus the raw 32-byte seed Iroh expects.

    ``secret_bytes`` is the canonical representation: hand it to
    ``iroh.SecretKey.from_bytes`` for the Endpoint, and to
    ``Ed25519PrivateKey.from_private_bytes`` for cert signing. Two
    distinct call sites keep needing the bytes; cache the parsed
    private/public objects on this dataclass to avoid re-parsing.
    """

    secret_bytes: bytes
    _private_key: Ed25519PrivateKey
    _public_key: Ed25519PublicKey

    @property
    def public_bytes(self) -> bytes:
        """The 32-byte raw Ed25519 verify key (== Iroh NodeId bytes)."""
        return self._public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    @property
    def public_hex(self) -> str:
        """Hex-encoded public key — used as a stable client_id in the gateway."""
        return self.public_bytes.hex()

    def sign(self, message: bytes) -> bytes:
        return self._private_key.sign(message)

    @classmethod
    def from_secret_bytes(cls, secret: bytes) -> Identity:
        if len(secret) != SECRET_KEY_LEN:
            raise ValueError(f"secret key must be {SECRET_KEY_LEN} bytes, got {len(secret)}")
        priv = Ed25519PrivateKey.from_private_bytes(secret)
        return cls(secret_bytes=secret, _private_key=priv, _public_key=priv.public_key())

    @classmethod
    def generate(cls) -> Identity:
        # ``secrets.token_bytes`` reads from the OS CSPRNG. We deliberately
        # don't use ``Ed25519PrivateKey.generate()`` because we need the raw
        # 32-byte seed (Iroh's SecretKey constructor takes bytes directly,
        # and ``private_bytes_raw`` is only available on newer cryptography
        # releases — generating from ``secrets`` and re-parsing is portable).
        return cls.from_secret_bytes(secrets.token_bytes(SECRET_KEY_LEN))


def _atomic_write_secret(path: Path, data: bytes) -> None:
    """Write *data* to *path* with 0600 perms via atomic rename.

    A crash mid-rename leaves either the old file intact or no file at
    all — never a half-written key. The temp file is created in the
    same directory so the rename is atomic on any POSIX FS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".identity-", dir=str(path.parent))
    try:
        if os.name == "posix":
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _publish_new_secret(path: Path, data: bytes) -> bool:
    """Publish a fully-written secret only when *path* is still absent.

    ``os.replace`` cannot be used for first creation: two processes which both
    observe a missing identity would overwrite each other and return different
    in-memory keys.  Instead, write to a unique tempfile and hard-link that
    complete inode into place.  Creating the destination link is an atomic
    no-replace operation on every supported local filesystem.  Exactly one
    process wins; a loser sees ``FileExistsError`` and must read the winner.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".identity-", dir=str(path.parent))
    try:
        if os.name == "posix":
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _read_identity(path: Path) -> Identity:
    """Read and validate one persisted identity, migrating legacy PEM once."""
    st = path.stat()
    # Ignore permission bits on Windows (they don't map cleanly).
    if os.name == "posix" and (st.st_mode & 0o077) != 0:
        raise PermissionError(
            f"{path} has permissions {oct(st.st_mode & 0o777)}; expected 0600. "
            "Run `chmod 600` on the file or remove it to regenerate."
        )
    secret = path.read_bytes()
    if len(secret) != SECRET_KEY_LEN:
        # PEM-format keys we wrote in an earlier draft show up here;
        # parse them once via cryptography and migrate to raw bytes.
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            priv = load_pem_private_key(secret, password=None)
            if not isinstance(priv, Ed25519PrivateKey):
                raise ValueError("not an Ed25519 PEM key")
            secret = priv.private_bytes(
                Encoding.Raw, PrivateFormat.Raw, NoEncryption(),
            )
            _atomic_write_secret(path, secret)
        except Exception as e:
            raise ValueError(
                f"{path} is not a valid 32-byte Ed25519 seed and isn't a recognised PEM either: {e}"
            ) from e
    return Identity.from_secret_bytes(secret)


def load_or_create_identity(path: Path | str) -> Identity:
    """Read or create the Ed25519 identity file at *path*.

    Permissions are checked on read — a 0644 file is rejected with a
    clear error rather than silently used, since a leaked agent key
    impersonates the agent on the entire network.
    """
    p = Path(path)
    try:
        return _read_identity(p)
    except FileNotFoundError:
        pass

    identity = Identity.generate()
    if _publish_new_secret(p, identity.secret_bytes):
        return identity
    # Another process won first creation.  Never return our discarded
    # candidate: both Desktop and CLI must bind Iroh to the persisted winner.
    return _read_identity(p)


def user_identity_path() -> Path:
    """Where the user-device identity lives (CLI / app installs).

    Distinct from the agent identity (which lives inside the agent dir):
    one user can drive many agents, and the same user identity travels
    with the device, not the agent.
    """
    return Path.home() / ".openagent" / "user" / "identity.key"


def iroh_node_id_public_bytes(node_id: str) -> bytes:
    """Decode an Iroh ``NodeId`` into its raw 32-byte Ed25519 public key.

    Device certificates bind that same key, so comparing this value with a
    certificate's ``device_pubkey`` is the proof-of-possession check that ties
    the application credential to the QUIC peer which presented it.  Keep the
    iroh import local: identity-file operations are also used by lightweight
    tooling that should not import the native transport until needed.
    """
    value = (node_id or "").strip()
    if not value:
        raise ValueError("missing Iroh peer node id")
    try:
        import iroh

        raw = bytes(iroh.PublicKey.from_string(value).to_bytes())
    except Exception as exc:  # noqa: BLE001 - normalise the FFI error surface
        raise ValueError(f"invalid Iroh peer node id: {value!r}") from exc
    if len(raw) != 32:
        raise ValueError(f"Iroh peer key must be 32 bytes, got {len(raw)}")
    return raw
