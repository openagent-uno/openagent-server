"""Safety blocklist — the enforcement half of ``safety.approvals``.

These tests exist because this repo has shipped a "safety" feature that was
inert before: ``safety.approvals`` was parsed into env vars nothing read, for
several releases, while ``examples/openagent.full.yaml`` advertised that it
blocked destructive commands and the operator's post-incident notes recorded
enabling it as a mitigation. Zero tests referenced it, so nothing caught it.

So these tests drive ``handlers.shell_exec`` — the real callsite — rather than
asserting things about ``src.core.safety`` in isolation. A test that only
proves ``check_command_allowed`` matches a regex would have passed just as
happily in the years the function was never called. (Cf. ``test_vault_reminder``,
which asserted "on by default" for a feature its only callsite defaulted off.)

The probe command throughout is ``echo "drop database prod;"``. It matches a
built-in blocklist pattern, and executing it prints a string and touches
nothing — so the off-path test can assert the command *actually ran* rather
than merely that no exception was raised.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from ._framework import TestContext, test

# Matches ``\bdrop\s+(?:database|table)\s+``; running it just echoes text.
BLOCKED_PROBE = 'echo "drop database prod;"'


@contextmanager
def _safety_env(**vars: str | None):
    """Set/clear OPENAGENT_SAFETY_* vars and always restore them.

    Restore is mandatory, not tidiness: module load order in ``_TEST_MODULES``
    is significant, and a leaked ``OPENAGENT_SAFETY_APPROVALS=1`` would arm the
    blocklist for every later test module in the same process.
    """
    prev = {k: os.environ.get(k) for k in vars}
    try:
        for k, v in vars.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _reset_shell_hub() -> None:
    from src.mcp.servers.shell import handlers
    handlers._reset_hub_for_tests()


# ── The off-by-default contract ─────────────────────────────────────
# The hard constraint on this feature: an existing deployment must behave
# EXACTLY as it did before the blocklist existed. These are the tests that
# hold that line.


@test("safety", "OFF by default: a blocklisted command still RUNS at the shell_exec callsite")
async def t_off_by_default_still_runs(ctx: TestContext) -> None:
    """The single most important test here.

    With no safety config present, a command the blocklist *would* catch must
    still execute normally — same exit code, same stdout, no exception. If this
    ever fails, the feature has started firing on deployments that never opted
    in, which is exactly the "live outage on someone's nightly cron" this was
    designed to avoid.
    """
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(
        OPENAGENT_SAFETY_APPROVALS=None,          # unset == absent from openagent.yaml
        OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS=None,
    ):
        out = await handlers.shell_exec(BLOCKED_PROBE, session_id="s_off")

    assert out["exit_code"] == 0, f"blocklisted cmd must run untouched when off, got {out}"
    assert "drop database prod;" in out["stdout"], (
        f"command must actually have executed, stdout={out['stdout']!r}"
    )
    assert out["timed_out"] is False


@test("safety", "OFF explicitly (enabled: false): a blocklisted command still runs")
async def t_off_explicit_still_runs(ctx: TestContext) -> None:
    """``safety.approvals.enabled: false`` exports "0" — must behave as unset."""
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(OPENAGENT_SAFETY_APPROVALS="0"):
        out = await handlers.shell_exec(BLOCKED_PROBE, session_id="s_off2")

    assert out["exit_code"] == 0
    assert "drop database prod;" in out["stdout"]


@test("safety", "OFF: block_extra_patterns alone must not arm the blocklist")
async def t_off_extras_inert(ctx: TestContext) -> None:
    """Extras without ``enabled: true`` are inert.

    An operator who lists patterns but never flips ``enabled`` has not opted
    in. Arming on the presence of extras would be a behaviour change for any
    config that carries the commented example stanza.
    """
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(
        OPENAGENT_SAFETY_APPROVALS=None,
        OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS=r"\becho\b",
    ):
        out = await handlers.shell_exec("echo hi", session_id="s_off3")

    assert out["exit_code"] == 0
    assert "hi" in out["stdout"]


@test("safety", "OFF: garbage in the enable flag reads as off, never on")
async def t_off_unrecognised_value(ctx: TestContext) -> None:
    """Fail-open on a typo. A misspelled flag must not silently arm a
    blocklist on a running deployment — the failure mode we can tolerate is
    "the guard I asked for isn't on", not "commands my agent needs are now
    refused at 3am"."""
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(OPENAGENT_SAFETY_APPROVALS="ture"):  # typo for "true"
        out = await handlers.shell_exec(BLOCKED_PROBE, session_id="s_off4")

    assert out["exit_code"] == 0
    assert "drop database prod;" in out["stdout"]


# ── The on-path actually blocks ─────────────────────────────────────


@test("safety", "ON: shell_exec refuses a blocklisted command and spawns nothing")
async def t_on_blocks_foreground(ctx: TestContext) -> None:
    from src.core.safety import BlockedCommandError
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(OPENAGENT_SAFETY_APPROVALS="1", OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS=None):
        try:
            await handlers.shell_exec(BLOCKED_PROBE, session_id="s_on")
        except BlockedCommandError as e:
            assert "drop database" in e.matched.lower(), f"unexpected match: {e.matched!r}"
            assert "openagent.yaml" in str(e), "refusal must tell the model how to proceed"
        else:
            raise AssertionError("blocklisted command must raise when approvals are on")


@test("safety", "ON: a benign command is untouched")
async def t_on_allows_benign(ctx: TestContext) -> None:
    """The blocklist must not become a general-purpose brake."""
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(OPENAGENT_SAFETY_APPROVALS="1"):
        out = await handlers.shell_exec("echo benign", session_id="s_on2")

    assert out["exit_code"] == 0
    assert "benign" in out["stdout"]


@test("safety", "ON: the background path is gated too (no shell registered)")
async def t_on_blocks_background(ctx: TestContext) -> None:
    """``run_in_background=True`` takes a different branch inside shell_exec.

    Gating only the foreground path would leave the trivial bypass of passing
    ``run_in_background=True``, so assert the block lands before spawn AND that
    no shell got registered in the hub.
    """
    from src.core.safety import BlockedCommandError
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(OPENAGENT_SAFETY_APPROVALS="1"):
        try:
            await handlers.shell_exec(
                BLOCKED_PROBE, run_in_background=True, session_id="s_bg",
            )
        except BlockedCommandError:
            pass
        else:
            raise AssertionError("background path must be gated too")

    listed = await handlers.shell_list(session_id="s_bg")
    assert listed == [], f"blocked command must not spawn/register a shell, got {listed}"


@test("safety", "ON: block_extra_patterns from config extend the built-in list")
async def t_on_extra_patterns(ctx: TestContext) -> None:
    """The documented extension point — the yaml's example is terraform."""
    from src.core.safety import BlockedCommandError
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(
        OPENAGENT_SAFETY_APPROVALS="1",
        OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS=r"\bterraform\s+destroy\b",
    ):
        try:
            await handlers.shell_exec("echo terraform destroy -auto-approve", session_id="s_x")
        except BlockedCommandError as e:
            assert "terraform" in e.pattern
        else:
            raise AssertionError("extra pattern must block")

        # …and the built-ins still apply alongside the extras.
        try:
            await handlers.shell_exec(BLOCKED_PROBE, session_id="s_x")
        except BlockedCommandError:
            pass
        else:
            raise AssertionError("built-ins must survive alongside extras")


@test("safety", "ON: an unparseable extra pattern is dropped, not fatal")
async def t_on_bad_extra_pattern(ctx: TestContext) -> None:
    """A typo'd regex must not take the agent down at its first shell call."""
    from src.core.safety import BlockedCommandError
    from src.mcp.servers.shell import handlers

    _reset_shell_hub()
    with _safety_env(
        OPENAGENT_SAFETY_APPROVALS="1",
        OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS="[unclosed",
    ):
        out = await handlers.shell_exec("echo still works", session_id="s_bad")
        assert out["exit_code"] == 0, "a bad extra pattern must not break shell_exec"

        try:
            await handlers.shell_exec(BLOCKED_PROBE, session_id="s_bad")
        except BlockedCommandError:
            pass
        else:
            raise AssertionError("built-ins must still apply despite a bad extra")


@test("safety", "ON: the retired claude_cli blocklist is covered end-to-end")
async def t_on_covers_retired_list(ctx: TestContext) -> None:
    """Every pattern in the list that died with ``e8f5d68`` still blocks.

    Driven through shell_exec (not the regex table) so this asserts the
    protection, not the data. Probes are ``echo``-prefixed: matching is
    substring-based, so this proves the pattern fires without running anything
    destructive if the gate were ever to regress open.
    """
    from src.core.safety import BlockedCommandError
    from src.mcp.servers.shell import handlers

    probes = [
        "echo rm -rf /",
        "echo sudo rm foo",
        "echo dd if=/dev/zero of=/dev/sda",
        "echo mkfs.ext4 /dev/sda1",
        "echo chmod -R 777 /etc",
        "echo git push --force origin main",
        "echo git push -f origin main",
        "echo git reset --hard origin/main",
        "echo kubectl delete pod x",
        "echo docker system prune -a",
        "echo drop table users",
        "echo shutdown now",
        "echo poweroff",
    ]
    _reset_shell_hub()
    with _safety_env(OPENAGENT_SAFETY_APPROVALS="1", OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS=None):
        for cmd in probes:
            try:
                await handlers.shell_exec(cmd, session_id="s_cov")
            except BlockedCommandError:
                continue
            raise AssertionError(f"should have been blocked: {cmd!r}")


# ── The config file must not lie again ──────────────────────────────


@test("safety", "no write-only OPENAGENT_SAFETY_* env var exists in src/")
async def t_no_write_only_safety_env(ctx: TestContext) -> None:
    """Guards the exact defect class that motivated this work.

    ``OPENAGENT_SAFETY_{GUARDRAILS,APPROVALS,BLOCK_EXTRA_PATTERNS,COMPRESSION,
    COMPRESSION_THRESHOLD_TOKENS}`` were all set by server.py and read by
    nothing — for the whole life of the config stanza. A safety-shaped env var
    with no reader is worse than no env var: it greps like a live mitigation,
    which is precisely how ``OPENAGENT_SAFETY_APPROVALS=1`` ended up in a
    post-incident note while doing nothing.

    So: every OPENAGENT_SAFETY_* name written under src/ must also be read
    under src/.
    """
    import re
    import tokenize
    from pathlib import Path

    def _code_only(path: Path) -> str:
        """Source with comments blanked out.

        Load-bearing: the server.py stanza *discusses* the retired
        ``OPENAGENT_SAFETY_GUARDRAILS`` / ``OPENAGENT_SAFETY_COMPRESSION`` vars
        in prose. Counting a comment as a reader would let a resurrected
        write-only export hide behind its own explanatory comment — which is
        how this test first shipped, and it was vacuous until a planted defect
        exposed it.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        try:
            with open(path, "rb") as f:
                toks = list(tokenize.tokenize(f.readline))
        except (tokenize.TokenError, SyntaxError, IndentationError):
            return text
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                lines[row - 1] = lines[row - 1][:col]
        return "\n".join(lines)

    root = Path(__file__).resolve().parents[2] / "src"
    written: dict[str, str] = {}
    read: set[str] = set()

    # A write is `os.environ["NAME"] = ...`.
    w_re = re.compile(r"os\.environ\[\s*[\"'](OPENAGENT_SAFETY_[A-Z_]+)[\"']\s*\]\s*=[^=]")
    # A read is an explicit lookup, or a module constant bound to the name
    # (``_APPROVALS_ENV = "OPENAGENT_SAFETY_APPROVALS"``) — which is how
    # safety.py names the vars it reads. Deliberately NOT matching bare
    # mentions: a docstring naming a var is not a reader.
    r_re = re.compile(
        r"(?:os\.environ\.get\(|os\.getenv\()\s*[\"'](OPENAGENT_SAFETY_[A-Z_]+)[\"']"
        r"|os\.environ\[\s*[\"'](OPENAGENT_SAFETY_[A-Z_]+)[\"']\s*\](?!\s*=[^=])"
        r"|^\s*[A-Za-z_]\w*\s*=\s*[\"'](OPENAGENT_SAFETY_[A-Z_]+)[\"']\s*$",
        re.MULTILINE,
    )
    for py in root.rglob("*.py"):
        code = _code_only(py)
        for m in w_re.finditer(code):
            written.setdefault(m.group(1), str(py))
        for m in r_re.finditer(code):
            read.add(next(g for g in m.groups() if g))

    assert written, "found no OPENAGENT_SAFETY_* writers — the scanner is broken"
    orphans = {name: loc for name, loc in written.items() if name not in read}
    assert not orphans, (
        "these OPENAGENT_SAFETY_* vars are written but never read — either wire "
        f"a reader or delete the export: {orphans}"
    )


@test("safety", "examples/openagent.full.yaml documents only safety keys the code reads")
async def t_example_yaml_safety_is_true(ctx: TestContext) -> None:
    """The shipped reference config must not advertise a protection that
    doesn't fire — the whole point of this change.

    Asserts the safety block is limited to keys ``_build_agent`` actually
    consumes, and that the shipped default stays OFF (a default flip here
    would silently arm the blocklist for anyone copying the reference).
    """
    from pathlib import Path

    import yaml

    p = Path(__file__).resolve().parents[2] / "examples" / "openagent.full.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    safety = cfg.get("safety") or {}

    assert set(safety) == {"approvals"}, (
        "safety block documents keys the server does not read "
        f"(guardrails/compression are retired): {sorted(safety)}"
    )
    assert safety["approvals"]["enabled"] is False, (
        "the shipped reference must keep approvals OFF by default"
    )
    # The documented extension point must parse as the code expects.
    for frag in safety["approvals"]["block_extra_patterns"]:
        __import__("re").compile(frag)


@test("safety", "an allow pattern carves an exception out of the blocklist")
async def t_allow_pattern_exempts(ctx: TestContext) -> None:
    """Without this the stanza is all-or-nothing, and therefore unused.

    ``git push --force`` ships in the default block list, but an autonomous
    agent that owns its own branch force-pushes it on every run by design
    (rebase, then force-push ``agent/<ticket>-…``). Such an operator's only
    choices were "blocklist off entirely" or "break my agent" — so they pick
    off, and every other protection in the list goes off with it.
    """
    import os

    from src.core import safety

    safety._compile.cache_clear()
    safety._compile_allow.cache_clear()
    prev_on = os.environ.get(safety._APPROVALS_ENV)
    prev_allow = os.environ.get(safety._ALLOW_PATTERNS_ENV)
    os.environ[safety._APPROVALS_ENV] = "1"
    os.environ[safety._ALLOW_PATTERNS_ENV] = r"git push --force[a-z-]* origin lyra-agent/"
    try:
        # The agent's own branch — its normal workflow keeps running.
        safety.check_command_allowed("git push --force origin lyra-agent/482-fix")
        safety.check_command_allowed("git push --force-with-lease origin lyra-agent/482")

        # The exception is SCOPED. This is the whole value: the May-2026
        # incident put production signing on master via a force-push, and a
        # blanket "allow git push --force" would re-permit exactly that.
        for blocked in (
            "git push --force origin master",
            "git push --force origin publish",
        ):
            try:
                safety.check_command_allowed(blocked)
            except safety.BlockedCommandError:
                pass
            else:
                raise AssertionError(
                    f"{blocked!r} was allowed — the exemption leaked past the "
                    "branch it was scoped to."
                )

        # Unrelated defaults are untouched by the exemption.
        for blocked in ("rm -rf /", "kubectl delete deployment lyra-api"):
            try:
                safety.check_command_allowed(blocked)
            except safety.BlockedCommandError:
                pass
            else:
                raise AssertionError(f"{blocked!r} was allowed by an unrelated exemption")
    finally:
        safety._compile.cache_clear()
        safety._compile_allow.cache_clear()
        for k, v in ((safety._APPROVALS_ENV, prev_on), (safety._ALLOW_PATTERNS_ENV, prev_allow)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@test("safety", "an allow pattern is inert while approvals are off")
async def t_allow_pattern_inert_when_off(ctx: TestContext) -> None:
    """The off path must return before touching either pattern list — an
    operator who leaves ``allow_patterns`` in their yaml and turns the
    stanza off must get today's byte-identical behaviour, not a half-armed
    policy that evaluates allows and then lets everything through anyway.
    """
    import os

    from src.core import safety

    safety._compile.cache_clear()
    safety._compile_allow.cache_clear()
    prev_on = os.environ.get(safety._APPROVALS_ENV)
    prev_allow = os.environ.get(safety._ALLOW_PATTERNS_ENV)
    os.environ.pop(safety._APPROVALS_ENV, None)
    os.environ[safety._ALLOW_PATTERNS_ENV] = "this-is-not-valid-regex((("
    try:
        # A syntactically broken allow pattern must not even be compiled
        # when the feature is off, let alone raise.
        safety.check_command_allowed("rm -rf /")
        safety.check_command_allowed("git push --force origin master")
    finally:
        safety._compile.cache_clear()
        safety._compile_allow.cache_clear()
        for k, v in ((safety._APPROVALS_ENV, prev_on), (safety._ALLOW_PATTERNS_ENV, prev_allow)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@test("safety", "a typo'd allow pattern fails closed, and says so")
async def t_bad_allow_pattern_audited(ctx: TestContext) -> None:
    """Inverted failure mode vs a block pattern, and worse.

    A typo'd BLOCK pattern fails safe: that one command stays unblocked and
    the rest of the list still applies. A typo'd ALLOW pattern fails CLOSED
    — the exception silently stops applying and the agent starts getting
    blocked on work that ran yesterday. That must be auditable rather than
    mysterious.
    """
    import os

    from src.core import safety

    safety._compile_allow.cache_clear()
    prev_on = os.environ.get(safety._APPROVALS_ENV)
    prev_allow = os.environ.get(safety._ALLOW_PATTERNS_ENV)
    os.environ[safety._APPROVALS_ENV] = "1"
    os.environ[safety._ALLOW_PATTERNS_ENV] = "git push --force origin lyra-agent/((("
    try:
        # Broken fragment is dropped, not raised — the agent must not die on
        # its first shell call because of a yaml typo.
        try:
            safety.check_command_allowed("git push --force origin lyra-agent/482")
        except safety.BlockedCommandError:
            pass  # fails CLOSED, as documented
        else:
            raise AssertionError(
                "a broken allow pattern silently allowed the command — it must "
                "fail closed, or a typo becomes a permission grant"
            )
    finally:
        safety._compile_allow.cache_clear()
        for k, v in ((safety._APPROVALS_ENV, prev_on), (safety._ALLOW_PATTERNS_ENV, prev_allow)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
