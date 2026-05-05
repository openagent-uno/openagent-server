"""Authentication primitives: device certificates + aiohttp middleware.

The coordinator mints CBOR-encoded, Ed25519-signed device certs after
PAKE login. Every inbound gateway request must carry a non-expired,
non-revoked cert signed by the pinned coordinator pubkey.
"""

from openagent.network.auth.device_cert import (
    DeviceCert,
    CertVerificationError,
    issue_cert,
    verify_cert,
    CERT_TTL_SECONDS,
)

__all__ = [
    "DeviceCert",
    "CertVerificationError",
    "issue_cert",
    "verify_cert",
    "CERT_TTL_SECONDS",
]
