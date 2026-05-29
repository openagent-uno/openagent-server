from typing import Any, Union

from src.memory.sessions.agent import AgentSession
from src.memory.sessions.summary import SessionSummaryManager
from src.memory.sessions.team import TeamSession


# WorkflowSession kept as a name-only stub: OpenAgent's workflow engine
# (src/workflow/) does not persist runs as a SessionStore row, but the
# session-store code's type signatures reference the symbol. Aliasing it
# to ``Any`` lets those signatures continue to type-check without
# carrying a real WorkflowSession schema.
WorkflowSession = Any

Session = Union[AgentSession, TeamSession]

__all__ = [
    "AgentSession",
    "TeamSession",
    "WorkflowSession",
    "Session",
    "SessionSummaryManager",
]
