"""``classify_run_error`` — stable codes for agent-run failures.

The codes exist so a client can react to a failed turn ("check the
provider key", "the network is down") without parsing prose that varies
by provider, SDK version and locale. Two properties matter and are
guarded here: the classifier reads the *cause* chain, because the useful
signal usually sits under the wrapper a provider raised, and an
unrecognised failure classifies as ``generic`` rather than being guessed
into a category a client would act on wrongly.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _Wrapped(Exception):
    """A provider-style wrapper that carries a status code."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@test("public_errors", "unrecognised failures classify as generic")
async def t_generic(_ctx: TestContext) -> None:
    from src.core.public_errors import classify_run_error

    result = classify_run_error(ValueError("something we have never seen"))
    assert result.code == "generic", f"expected generic, got {result.code!r}"
    assert result.public_message, "every code must have a public message"


@test("public_errors", "status codes map to auth and rate_limit")
async def t_status_codes(_ctx: TestContext) -> None:
    from src.core.public_errors import classify_run_error

    for status, expected in ((401, "auth"), (403, "auth"), (429, "rate_limit")):
        result = classify_run_error(_Wrapped("provider said no", status))
        assert result.code == expected, (
            f"status {status} should classify as {expected!r}, got {result.code!r}"
        )


@test("public_errors", "classification follows the cause chain")
async def t_walks_cause_chain(_ctx: TestContext) -> None:
    """A provider wraps the socket error it hit, so the DNS text is on the
    cause rather than the exception the run actually raised. Reading only
    the outermost exception would classify every one of those as generic."""
    from src.core.public_errors import classify_run_error

    try:
        try:
            raise OSError("Temporary failure in name resolution")
        except OSError as inner:
            raise _Wrapped("model call failed") from inner
    except _Wrapped as outer:
        result = classify_run_error(outer)

    assert result.code == "dns", (
        f"the cause's DNS text must be found, got {result.code!r}"
    )


@test("public_errors", "a self-referential cause chain terminates")
async def t_chain_is_bounded(_ctx: TestContext) -> None:
    """``__context__`` can point back at an exception already visited.
    Without the seen-set the walk would spin instead of returning."""
    from src.core.public_errors import classify_run_error

    a = _Wrapped("a")
    b = _Wrapped("b")
    a.__cause__ = b
    b.__cause__ = a

    result = classify_run_error(a)
    assert result.code == "generic", f"expected generic, got {result.code!r}"


@test("public_errors", "public messages never echo the diagnostic")
async def t_public_message_is_constant(_ctx: TestContext) -> None:
    """The whole point of ``public_message`` is that a host can show it to
    someone who must not see provider internals. A message built from the
    exception would defeat that silently."""
    from src.core.public_errors import PUBLIC_MESSAGES, classify_run_error

    secret = "sk-live-0123456789-should-never-surface"
    result = classify_run_error(_Wrapped(secret, 401))

    assert result.code == "auth", f"expected auth, got {result.code!r}"
    assert secret not in result.public_message, (
        "public_message must not carry the diagnostic it was classified from"
    )
    assert result.public_message == PUBLIC_MESSAGES["auth"], (
        "public_message must be the constant for its code"
    )


@test("public_errors", "RunTurnError round-trips its code")
async def t_run_turn_error_round_trip(_ctx: TestContext) -> None:
    """``StreamSession`` rebuilds an exception from the ``done`` frame so it
    can report ``TURN_END_ERROR``. Re-classifying that stand-in must give
    back the code the agent already determined, not re-derive it from the
    message."""
    from src.core.public_errors import RunTurnError, classify_run_error

    restored = classify_run_error(RunTurnError("opaque", code="rate_limit"))
    assert restored.code == "rate_limit", (
        f"code must survive the round trip, got {restored.code!r}"
    )

    unknown = RunTurnError("opaque", code="not-a-real-code")
    assert unknown.code == "generic", (
        f"an unknown code falls back to generic, got {unknown.code!r}"
    )
