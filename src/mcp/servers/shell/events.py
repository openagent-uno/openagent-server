"""Event types posted by ShellHub when a background shell reaches a
terminal state. Only terminal events are posted — ``new_output`` does
NOT trigger the agent auto-loop, to avoid chatty processes like
``tail -f`` spamming the session with reminders (see spec § Events).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ShellEventKind = Literal["completed", "timed_out", "killed"]


@dataclass(frozen=True)
class ShellEvent:
    shell_id: str
    kind: ShellEventKind
    exit_code: int | None
    signal: str | None
    bytes_stdout: int
    bytes_stderr: int
    at: float
