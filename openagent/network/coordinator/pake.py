"""PAKE backend: SRP-6a today, OPAQUE swap-in tomorrow.

The coordinator never sees the user's password — registration sends a
verifier and login proves knowledge of the password by completing the
SRP-6a exchange. We commit to a small ``PakeBackend`` interface so the
service code can swap to OPAQUE later by replacing ``Srp6aBackend``
with ``OpaqueBackend`` in one place (``coordinator.service``).

SRP-6a parameters: NIST 3072-bit prime + g=5, SHA-256. The constants
match RFC 5054 group ID 4. Both client and server must agree, so do
*not* tune these — agents pin them via ``PAKE_GROUP``.

Implementation uses ``srptools`` (pure-Python). Thread-safe stateless
calls except ``LoginInProgress`` which holds the per-login server
state between ``login_init`` and ``login_finish``.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Protocol

from srptools import (
    SRPContext,
    SRPServerSession,
    constants as srp_constants,
)


def _coerce_bytes(x) -> bytes:
    """Decode srptools' hex-string-as-bytes into raw bytes.

    srptools' ``session.key``, ``key_proof``, ``key_proof_hash`` all
    return *bytes containing ASCII hex characters* (so a 32-byte SHA-256
    digest comes back as 64 ASCII bytes). Calling ``bytes.fromhex`` on
    the decoded ASCII brings us to the raw 32-byte form we actually
    want on the wire.

    Older / newer srptools may flip to returning ``str``; accept both.
    """
    if isinstance(x, str):
        return bytes.fromhex(x)
    if isinstance(x, (bytes, bytearray)):
        try:
            return bytes.fromhex(bytes(x).decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            # If it's already raw, return as-is.
            return bytes(x)
    raise TypeError(f"expected bytes or hex str, got {type(x).__name__}")


PAKE_GROUP = srp_constants.PRIME_3072
PAKE_GENERATOR = srp_constants.PRIME_3072_GEN
import hashlib

# srptools expects the actual hash callable, not a name string.
PAKE_HASH = hashlib.sha256


class PakeError(Exception):
    """Raised on any PAKE-protocol-level failure (bad message, mismatch)."""


@dataclass
class LoginInProgress:
    """Server-side state held between ``login_init`` and ``login_finish``.

    Stored in the coordinator's per-connection memory only — never
    persisted. If the client doesn't complete within a few seconds the
    state is dropped and they have to start over.
    """

    handle: str
    session: SRPServerSession
    created_at: float


class PakeBackend(Protocol):
    """Stateless registration / login surface used by the coordinator.

    All methods take/return raw bytes (the wire-side encoding is the
    coordinator service's responsibility). The PAKE algorithm name is
    stored on the user row alongside the verifier so a future
    multi-algo deployment can pick the right backend per user.
    """

    algo: str

    def register_finalize(self, handle: str, password_verifier_payload: bytes) -> bytes:
        """Validate a registration payload and return the verifier to store.

        For SRP-6a the client sends ``salt || verifier`` directly and we
        store it verbatim — there's no server contribution to
        registration. OPAQUE will compute a derived record here.
        """

    def login_init(self, handle: str, stored_record: bytes, ke1: bytes) -> tuple[LoginInProgress, bytes]:
        """Process the client's first login message; return server response and state."""

    def login_finish(self, state: LoginInProgress, ke3: bytes) -> bytes:
        """Verify the client's proof; return the shared session key."""


class Srp6aBackend:
    """SRP-6a backend wrapping ``srptools``.

    Registration wire: ``u8(salt_len) || salt || verifier(384)``.
    The verifier width is the prime size (384 bytes for 3072-bit) so we
    don't prefix it; the salt length varies (srptools defaults to 8
    bytes, but a future tweak might push it longer).

    Login wire (init):    ``A(384)``                  — client public.
    Login wire (resp):    ``u8(salt_len)||salt||B(384)`` — server's reply.
    Login wire (finish):  ``M1(32)``                  — client proof.
    Login wire (resp):    ``M2(32)``                  — server proof.
    """

    algo = "srp6a"

    VERIFIER_LEN = 384
    PUB_LEN = 384
    # srptools returns the proof as ASCII-hex *bytes* — 64 bytes
    # representing a 32-byte SHA-256 digest. We pass these through
    # unchanged on the wire because srptools' ``verify_proof`` only
    # accepts that exact form (decoding to the raw 32 bytes makes the
    # check return False). The size cost is negligible.
    PROOF_LEN = 64
    MAX_SALT_LEN = 64  # generous bound; srptools defaults to 8

    def _split_record(self, record: bytes) -> tuple[bytes, bytes]:
        if len(record) < 1:
            raise PakeError("empty registration record")
        salt_len = record[0]
        if salt_len == 0 or salt_len > self.MAX_SALT_LEN:
            raise PakeError(f"unreasonable salt_len: {salt_len}")
        if len(record) != 1 + salt_len + self.VERIFIER_LEN:
            raise PakeError(
                f"registration record length mismatch: header says salt={salt_len}, "
                f"expected {1 + salt_len + self.VERIFIER_LEN} total, got {len(record)}",
            )
        salt = record[1 : 1 + salt_len]
        verifier = record[1 + salt_len :]
        return salt, verifier

    def register_finalize(self, handle: str, payload: bytes) -> bytes:
        # Round-trip through the splitter so a malformed payload is
        # rejected at registration time, not later at login.
        self._split_record(payload)
        return payload

    def login_init(
        self, handle: str, stored_record: bytes, ke1: bytes,
    ) -> tuple[LoginInProgress, bytes]:
        import time

        salt, verifier = self._split_record(stored_record)
        if len(ke1) != self.PUB_LEN:
            raise PakeError(f"login_init A must be {self.PUB_LEN} bytes, got {len(ke1)}")

        ctx = SRPContext(
            handle,
            prime=PAKE_GROUP,
            generator=PAKE_GENERATOR,
            hash_func=PAKE_HASH,
        )
        # ``SRPServerSession`` expects hex strings for verifier/salt/A;
        # the bytes-clean wrapper would be nicer but srptools is what
        # we have. Hex encoding is unambiguous and roughly doubles the
        # in-memory size — fine for short-lived state.
        session = SRPServerSession(
            ctx,
            verifier.hex(),
            private=secrets.token_hex(32),
        )
        session.process(ke1.hex(), salt.hex())
        B_hex = session.public
        if not session.key:
            # ``SRPServerSession.process`` derives the shared key, but
            # any failure here is a malformed A. Reject explicitly so
            # the coordinator can rate-limit.
            raise PakeError("login_init: SRP server failed to derive session key")
        B = bytes.fromhex(B_hex)
        if len(B) != self.PUB_LEN:
            # srptools returns the minimum-byte representation; left-pad
            # so the wire format stays fixed-width.
            B = B.rjust(self.PUB_LEN, b"\x00")
        response = bytes([len(salt)]) + salt + B
        state = LoginInProgress(handle=handle, session=session, created_at=time.time())
        return state, response

    def login_finish(self, state: LoginInProgress, ke3: bytes) -> bytes:
        if len(ke3) != self.PROOF_LEN:
            raise PakeError(f"login_finish M1 must be {self.PROOF_LEN} bytes, got {len(ke3)}")
        # ke3 is srptools' ASCII-hex bytes form (see PROOF_LEN comment).
        # ``verify_proof`` only accepts that exact form.
        if not state.session.verify_proof(ke3):
            raise PakeError("login proof verification failed")
        # Server proof goes back as srptools delivered it — same encoding.
        return bytes(state.session.key_proof_hash)


# ── Client-side helpers ───────────────────────────────────────────────────
# Used by the CLI/app login flow. Kept here so both sides agree on the
# wire format (one source of truth per algorithm).


def srp6a_make_registration(handle: str, password: str) -> bytes:
    """Compute the (salt, verifier) registration payload for SRP-6a.

    Wire format: ``u8(salt_len) || salt || verifier(VERIFIER_LEN)``.
    """
    ctx = SRPContext(
        handle,
        password,
        prime=PAKE_GROUP,
        generator=PAKE_GENERATOR,
        hash_func=PAKE_HASH,
    )
    # srptools returns ``(username, verifier, salt)`` — verifier first.
    _, verifier_hex, salt_hex = ctx.get_user_data_triplet()
    salt = bytes.fromhex(salt_hex)
    verifier = bytes.fromhex(verifier_hex)
    if len(verifier) != Srp6aBackend.VERIFIER_LEN:
        verifier = verifier.rjust(Srp6aBackend.VERIFIER_LEN, b"\x00")
    if len(salt) > Srp6aBackend.MAX_SALT_LEN:
        raise PakeError(f"srptools returned an unreasonably long salt: {len(salt)}")
    return bytes([len(salt)]) + salt + verifier


@dataclass
class Srp6aClientLogin:
    """Mutable client-side state across the two-message login exchange."""

    handle: str
    password: str
    a_priv: str  # hex
    A: bytes
    session_key: bytes | None = None

    @classmethod
    def start(cls, handle: str, password: str) -> Srp6aClientLogin:
        ctx = SRPContext(
            handle, password,
            prime=PAKE_GROUP, generator=PAKE_GENERATOR, hash_func=PAKE_HASH,
        )
        from srptools import SRPClientSession

        priv = secrets.token_hex(32)
        session = SRPClientSession(ctx, private=priv)
        A_hex = session.public
        A = bytes.fromhex(A_hex)
        if len(A) != Srp6aBackend.PUB_LEN:
            A = A.rjust(Srp6aBackend.PUB_LEN, b"\x00")
        # Stash the private key we passed in so we can reconstruct the
        # session for the proof step. srptools sessions aren't directly
        # serialisable but are cheap to rebuild.
        return cls(handle=handle, password=password, a_priv=priv, A=A)

    def respond(self, server_response: bytes) -> bytes:
        """Process server's (salt, B) and return our proof M1."""
        if len(server_response) < 1:
            raise PakeError("login response truncated")
        salt_len = server_response[0]
        if salt_len == 0 or salt_len > Srp6aBackend.MAX_SALT_LEN:
            raise PakeError(f"login response has unreasonable salt_len: {salt_len}")
        expected = 1 + salt_len + Srp6aBackend.PUB_LEN
        if len(server_response) != expected:
            raise PakeError(
                f"login response length mismatch: expected {expected}, got {len(server_response)}",
            )
        salt = server_response[1 : 1 + salt_len]
        B = server_response[1 + salt_len :]
        ctx = SRPContext(
            self.handle, self.password,
            prime=PAKE_GROUP, generator=PAKE_GENERATOR, hash_func=PAKE_HASH,
        )
        from srptools import SRPClientSession

        session = SRPClientSession(ctx, private=self.a_priv)
        session.process(B.hex(), salt.hex())
        if not session.key:
            raise PakeError("client: SRP could not derive shared key")
        # Pass srptools' values straight through — they're already in the
        # form ``verify_proof`` expects. See ``PROOF_LEN`` comment.
        self.session_key = bytes(session.key)
        M1 = bytes(session.key_proof)
        if len(M1) != Srp6aBackend.PROOF_LEN:
            raise PakeError(f"client: unexpected proof length {len(M1)}")
        self._expected_M2 = bytes(session.key_proof_hash)
        return M1

    def verify_server(self, M2: bytes) -> None:
        if not hasattr(self, "_expected_M2"):
            raise PakeError("verify_server called before respond")
        if M2 != self._expected_M2:
            raise PakeError("server proof mismatch")
