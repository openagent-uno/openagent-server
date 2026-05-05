"""Client-side network: dialer + login flow.

Used by the CLI, the desktop-app sidecar, and one OpenAgent calling
into another (federation). The two key types are:

  - ``SessionDialer``: holds a device cert and opens authed Iroh
    streams to a target NodeId, presenting the cert on every stream.
  - ``login()`` / ``register()``: the PAKE flow that exchanges a
    ``handle@network`` + password for a coordinator-signed cert.
"""

from openagent.network.client.login import (
    LoginError,
    register,
    login,
    refresh_cert,
)
from openagent.network.client.session import SessionDialer

__all__ = [
    "LoginError",
    "SessionDialer",
    "register",
    "login",
    "refresh_cert",
]
