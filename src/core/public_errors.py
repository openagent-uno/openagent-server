"""Stable classification for agent-run failures.

``Agent.run_stream`` catches every failure and reports it as a ``done``
frame instead of raising. Without a marker on that frame a failed turn is
shaped exactly like a successful one: ``StreamSession`` republishes the
text as ``OutTextDelta`` and closes the turn ``completed``, so the app, the
CLI and every bridge file the exception message as the model's answer. The
frame now carries ``errored`` plus a code from this module, and the turn
ends on ``TURN_END_ERROR`` through ``OutError`` — the vocabulary
``src/stream/events.py`` already defines for exactly this.

Codes are deliberately coarse. They exist so a client can react — "check
the provider key", "the network is down" — without parsing prose that
varies by provider, version and locale. Classification reads the exception
chain locally and only constants leave this module through
``public_message``; the real diagnostic still travels in the frame's
``text``, because a self-hosted agent is expected to diagnose itself from
what it actually saw (§14). A host that must not show provider detail to
the person in the chat renders ``public_message`` instead, keeping the
code.
"""

from __future__ import annotations

GENERIC = "generic"

PUBLIC_MESSAGES: dict[str, str] = {
    GENERIC: "The agent run failed. Please retry.",
    "dns": (
        "The model provider could not be reached because DNS resolution "
        "failed. Check the host's network and DNS, then retry."
    ),
    "connection": (
        "The model provider could not be reached. Check the network and the "
        "provider endpoint, then retry."
    ),
    "timeout": (
        "The model provider request timed out. Check the connection and the "
        "provider's status before retrying."
    ),
    "auth": (
        "The model provider rejected authentication or access. Check this "
        "agent's provider settings."
    ),
    "rate_limit": (
        "The model provider reported a rate or quota limit. Check usage and "
        "retry when it clears."
    ),
}

_DNS_MARKERS = (
    "could not resolve host",
    "name or service not known",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "nodename nor servname provided",
)

# Bound the walk: ``__cause__``/``__context__`` chains can be long, and a
# self-referential one would otherwise spin.
_MAX_CHAIN = 8


class RunTurnError(RuntimeError):
    """A run failure reconstructed from a ``done`` frame.

    ``run_stream`` already handled the original exception, so the stream
    layer has no exception object to record. This stands in for one, which
    is what ``StreamSession`` needs to report ``TURN_END_ERROR`` instead of
    inferring "answered" from the absence of a raise.
    """

    def __init__(self, message: str, *, code: str = GENERIC) -> None:
        self.code = code if code in PUBLIC_MESSAGES else GENERIC
        super().__init__(message)


class RunErrorClass:
    """A classified failure: a stable code and a diagnostic-free message."""

    __slots__ = ("code",)

    def __init__(self, code: str = GENERIC) -> None:
        self.code = code if code in PUBLIC_MESSAGES else GENERIC

    @property
    def public_message(self) -> str:
        return PUBLIC_MESSAGES[self.code]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RunErrorClass({self.code!r})"


def classify_run_error(error: BaseException | None) -> RunErrorClass:
    """Map a failure onto one of ``PUBLIC_MESSAGES``' codes.

    Walks the exception chain because the useful signal is usually on the
    cause: a provider wrapper carries the status code, the socket error
    underneath carries the DNS text. Unrecognised failures classify as
    ``generic`` rather than guessing.
    """
    seen: set[int] = set()
    for _ in range(_MAX_CHAIN):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))

        if isinstance(error, RunTurnError):
            return RunErrorClass(error.code)

        diagnostic = ""
        if isinstance(error, BaseException):
            diagnostic = str(error).lower()

        if any(marker in diagnostic for marker in _DNS_MARKERS):
            return RunErrorClass("dns")
        if isinstance(error, TimeoutError) or "timed out" in diagnostic:
            return RunErrorClass("timeout")
        if "connection refused" in diagnostic or "failed to connect" in diagnostic:
            return RunErrorClass("connection")

        status = getattr(error, "status_code", None)
        if status is None:
            data = getattr(error, "additional_data", None)
            if isinstance(data, dict):
                status = data.get("status_code")
        if status in (401, 403):
            return RunErrorClass("auth")
        if status == 429:
            return RunErrorClass("rate_limit")

        error = getattr(error, "__cause__", None) or getattr(error, "__context__", None)

    return RunErrorClass()
