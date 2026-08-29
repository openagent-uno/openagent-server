"""Server-owned Custom Views and live dashboard runtime.

The package deliberately keeps the agent-authored OA-UI source, canonical
SQLite metadata, and the live subscription runtime separate.  Clients only
receive a validated JSON AST; they never execute markup, JavaScript, or shell
snippets supplied by a model or a data source.
"""

from .compiler import OAUIValidationError, compile_oaui
from .repository import (
    CustomViewConflict,
    CustomViewNotFound,
    CustomViewRateLimited,
    CustomViewRepository,
)
from .service import CustomViewService, service_for_db, service_for_gateway

__all__ = [
    "CustomViewConflict",
    "CustomViewNotFound",
    "CustomViewRateLimited",
    "CustomViewRepository",
    "CustomViewService",
    "OAUIValidationError",
    "compile_oaui",
    "service_for_db",
    "service_for_gateway",
]
