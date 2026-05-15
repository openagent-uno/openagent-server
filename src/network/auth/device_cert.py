"""Coordinator-signed device certificates.

Format: ``CBOR{payload} || sig`` where:
  - ``payload`` is a CBOR map containing the bound-claims listed in the
    ``DeviceCert`` dataclass, plus a ``v`` version field for future
    format breakage.
  - ``sig`` is a 64-byte Ed25519 signature by the coordinator over the
    raw CBOR-encoded payload bytes.

The cert is presented by the client on the first frame of every
gateway stream. Verification is purely local (no coordinator round-
trip) — gateways pin the coordinator's pubkey at startup and verify
against that. Revocation is enforced by the agent reading
``network_devices.status`` from the local DB on each stream; the cert
TTL bounds how long a leaked cert is usable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cbor2
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


CERT_VERSION = 1
CERT_TTL_SECONDS = 30 * 24 * 3600  # 30 days
SIGNATURE_LEN = 64


class CertVerificationError(Exception):
    """Raised by ``verify_cert`` for any reason a cert is rejected."""


@dataclass(frozen=True)
class DeviceCert:
    """Decoded, *unverified* contents of a device cert.

    Use ``verify_cert`` to obtain one of these — never trust a payload
    decoded directly. The dataclass is frozen so call sites can't
    accidentally mutate the bound-claims after verification.
    """

    handle: str
    device_pubkey: bytes
    network_id: str
    issued_at: float
    expires_at: float
    capabilities: list[str]

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at

    @property
    def device_pubkey_hex(self) -> str:
        return self.device_pubkey.hex()


def _encode_payload(cert: DeviceCert) -> bytes:
    return cbor2.dumps({
        "v": CERT_VERSION,
        "handle": cert.handle,
        "device_pubkey": cert.device_pubkey,
        "network_id": cert.network_id,
        "issued_at": cert.issued_at,
        "expires_at": cert.expires_at,
        "capabilities": list(cert.capabilities),
    })


def _decode_payload(payload: bytes) -> DeviceCert:
    obj = cbor2.loads(payload)
    if not isinstance(obj, dict):
        raise CertVerificationError("cert payload is not a CBOR map")
    if obj.get("v") != CERT_VERSION:
        raise CertVerificationError(f"unsupported cert version: {obj.get('v')}")
    try:
        return DeviceCert(
            handle=str(obj["handle"]),
            device_pubkey=bytes(obj["device_pubkey"]),
            network_id=str(obj["network_id"]),
            issued_at=float(obj["issued_at"]),
            expires_at=float(obj["expires_at"]),
            capabilities=list(obj.get("capabilities") or []),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CertVerificationError(f"cert payload missing/malformed field: {e}") from e


def issue_cert(
    *,
    coordinator_key: Ed25519PrivateKey,
    handle: str,
    device_pubkey: bytes,
    network_id: str,
    capabilities: list[str] | None = None,
    ttl_seconds: int = CERT_TTL_SECONDS,
    now: float | None = None,
) -> bytes:
    """Mint a fresh device cert. Returns the wire bytes (payload || sig)."""
    if len(device_pubkey) != 32:
        raise ValueError(f"device_pubkey must be 32 bytes, got {len(device_pubkey)}")
    issued = now or time.time()
    cert = DeviceCert(
        handle=handle,
        device_pubkey=device_pubkey,
        network_id=network_id,
        issued_at=issued,
        expires_at=issued + ttl_seconds,
        capabilities=capabilities or [],
    )
    payload = _encode_payload(cert)
    sig = coordinator_key.sign(payload)
    # Wire format: 4-byte big-endian payload length || payload || 64-byte sig.
    # The length prefix means the verifier doesn't need to know cert
    # framing in advance — just read the prefix, then payload+sig.
    return len(payload).to_bytes(4, "big") + payload + sig


def verify_cert(
    wire: bytes,
    *,
    coordinator_pubkey: Ed25519PublicKey,
    expected_network_id: str | None = None,
    now: float | None = None,
) -> DeviceCert:
    """Decode + signature-check + expiry-check a cert.

    Raises ``CertVerificationError`` on any failure (bad framing, bad
    sig, expired, network mismatch). The caller is still responsible
    for the *liveness* check (``network_devices.status='active'``) —
    that requires the agent's DB and isn't a property of the cert
    itself.
    """
    if len(wire) < 4 + SIGNATURE_LEN:
        raise CertVerificationError("cert too short")
    payload_len = int.from_bytes(wire[:4], "big")
    if len(wire) != 4 + payload_len + SIGNATURE_LEN:
        raise CertVerificationError(
            f"cert length mismatch: header says {payload_len} payload + {SIGNATURE_LEN} sig, "
            f"got {len(wire) - 4} after header"
        )
    payload = wire[4:4 + payload_len]
    sig = wire[4 + payload_len:]

    try:
        coordinator_pubkey.verify(sig, payload)
    except InvalidSignature as e:
        raise CertVerificationError("invalid coordinator signature") from e

    cert = _decode_payload(payload)
    if cert.is_expired(now=now):
        raise CertVerificationError("cert expired")
    if expected_network_id is not None and cert.network_id != expected_network_id:
        raise CertVerificationError(
            f"cert is for network {cert.network_id!r}, expected {expected_network_id!r}",
        )
    return cert
