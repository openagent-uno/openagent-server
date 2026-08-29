"""Normalized operational history, activity, and derived-search support.

The Memory Vault deliberately does not live in this package.  Operational
history is canonical SQLite state; its FTS database is a disposable projection.
"""

from .schema import (
    OPERATIONAL_SCHEMA_VERSION,
    OperationalMigrationError,
    ensure_operational_storage,
)
from .phase import (
    PhaseTransition,
    StoragePhaseError,
    transition_storage_phase,
    transition_storage_phase_async,
)

__all__ = [
    "OPERATIONAL_SCHEMA_VERSION",
    "OperationalMigrationError",
    "ensure_operational_storage",
    "PhaseTransition",
    "StoragePhaseError",
    "transition_storage_phase",
    "transition_storage_phase_async",
]
