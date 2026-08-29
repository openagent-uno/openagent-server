"""Principal-bound tools for creating and managing OpenAgent Custom Views.

This MCP stays in the server process: it needs the authenticated on-behalf-of
identity and shares the live data runtime with the gateway.  The model can
author OA-UI, bundle scripts/assets, and server-side source/action definitions,
but it never receives raw bundle paths, secrets, or another principal's objects.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Mapping

from src.core.on_behalf_context import current_on_behalf_identity
from src.custom_views.compiler import (
    COMPONENTS,
    MAX_DEPTH,
    MAX_NODES,
    MAX_SOURCE_BYTES,
    SCHEMA_VERSION,
)
from src.custom_views.bundles import (
    MAX_ASSET_BYTES,
    MAX_BUNDLE_BYTES,
    MAX_BUNDLE_FILES,
    MAX_SCRIPT_BYTES,
    safe_relative_path,
)
from src.custom_views.repository import (
    CustomViewConflict,
    CustomViewError,
    CustomViewImmutable,
    CustomViewInputError,
    CustomViewNotFound,
)
from src.custom_views.service import service_for_db
from src.memory.operational.access import AccessContext


logger = logging.getLogger(__name__)


def build_runtime_toolkit(pool: Any) -> Any:
    """Build the ui-manager toolkit around the canonical DB and live runtime."""

    from src.mcp._runtime import Toolkit

    db = getattr(pool, "_db", None)
    if db is None:
        raise RuntimeError("ui-manager requires the canonical database")
    service = service_for_db(db, pool=pool)

    async def access_for_turn() -> AccessContext:
        identity = current_on_behalf_identity()
        if identity is not None:
            return AccessContext.from_on_behalf_identity(identity)

        # Scheduled tasks and detached workflow/event runs intentionally have
        # no human device certificate.  They execute as the local agent in its
        # installation tenant, never as a caller-supplied owner.
        conn = await db._ensure_connected()
        tenant_row = await (
            await conn.execute("SELECT NULLIF(network_id, '') FROM network LIMIT 1")
        ).fetchone()
        if tenant_row is not None and tenant_row[0]:
            tenant = str(tenant_row[0])
        else:
            state = await (
                await conn.execute(
                    "SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1"
                )
            ).fetchone()
            if state is None:
                raise PermissionError("installation identity is unavailable")
            tenant = f"installation:{state[0]}"
        handle = "openagent"
        principal_id = f"agent:{handle}"
        return AccessContext(
            tenant_id=tenant,
            principal_id=principal_id,
            principal_type="agent",
            handle=handle,
            device_id="internal-runtime",
            principal_ids=frozenset({principal_id, "agent:internal-runtime"}),
            grant_identities=frozenset(
                {
                    ("agent", handle),
                    ("agent", principal_id),
                    ("installation", tenant),
                }
            ),
        )

    def safe_error(exc: Exception) -> dict[str, Any]:
        if isinstance(exc, CustomViewConflict):
            return {"ok": False, "error": "revision_conflict", "message": str(exc)}
        if isinstance(exc, CustomViewNotFound):
            return {"ok": False, "error": "not_found", "message": str(exc)}
        if isinstance(exc, CustomViewImmutable):
            return {"ok": False, "error": "immutable", "message": str(exc)}
        if isinstance(exc, CustomViewInputError):
            return {"ok": False, "error": "invalid_view", "message": str(exc)}
        logger.error("ui-manager operation failed (details suppressed)")
        return {
            "ok": False,
            "error": "temporarily_unavailable",
            "message": "Custom Views are temporarily unavailable.",
        }

    def decode_scripts(value: Mapping[str, Any] | None) -> dict[str, str] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or len(value) > MAX_BUNDLE_FILES:
            raise CustomViewInputError("scripts contains too many files")
        total = 0
        result: dict[str, str] = {}
        for raw_name, payload in value.items():
            try:
                name = safe_relative_path(raw_name)
            except (TypeError, ValueError) as exc:
                raise CustomViewInputError("script path is invalid") from exc
            if not isinstance(payload, str) or len(payload) > MAX_SCRIPT_BYTES:
                raise CustomViewInputError("script exceeds the supported size")
            encoded = payload.encode("utf-8")
            if len(encoded) > MAX_SCRIPT_BYTES:
                raise CustomViewInputError("script exceeds the supported size")
            total += len(encoded)
            if total > MAX_BUNDLE_BYTES:
                raise CustomViewInputError("scripts exceed the supported bundle size")
            result[name] = payload
        return result

    def decode_assets(value: Mapping[str, Any] | None) -> dict[str, bytes] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or len(value) > MAX_BUNDLE_FILES:
            raise CustomViewInputError("assets contains too many files")
        max_encoded = 4 * ((MAX_ASSET_BYTES + 2) // 3)
        prepared: list[tuple[str, str]] = []
        total = 0
        # Bound every encoded file and the aggregate before expanding even the
        # first payload in memory. Strict base64 validation still runs in the
        # decode pass below.
        for raw_name, encoded in value.items():
            try:
                name = safe_relative_path(raw_name)
            except (TypeError, ValueError) as exc:
                raise CustomViewInputError("asset path is invalid") from exc
            if not isinstance(encoded, str) or len(encoded) > max_encoded:
                raise CustomViewInputError("asset exceeds the supported size")
            try:
                encoded.encode("ascii")
            except UnicodeEncodeError as exc:
                raise CustomViewInputError("asset base64 payload is invalid") from exc
            if len(encoded) % 4:
                raise CustomViewInputError("asset base64 payload is invalid")
            padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
            predicted_size = (len(encoded) // 4) * 3 - padding
            if predicted_size > MAX_ASSET_BYTES:
                raise CustomViewInputError("asset exceeds the supported size")
            total += predicted_size
            if total > MAX_BUNDLE_BYTES:
                raise CustomViewInputError("assets exceed the supported bundle size")
            prepared.append((name, encoded))

        result: dict[str, bytes] = {}
        for name, encoded in prepared:
            try:
                payload = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise CustomViewInputError("asset base64 payload is invalid") from exc
            if len(payload) > MAX_ASSET_BYTES:
                raise CustomViewInputError("asset exceeds the supported size")
            result[name] = payload
        return result

    async def ui_get_schema() -> dict[str, Any]:
        """Return the OA-UI v1 component, binding, lifecycle, and limit schema.

        Call this before authoring an unfamiliar component or data source.
        OA-UI is declarative only: HTML, JavaScript, CSS and iframes are not
        supported. Media must use artifact:/asset: references.
        """

        return {
            "ok": True,
            "schemaVersion": SCHEMA_VERSION,
            "components": sorted(COMPONENTS),
            "drivers": ["static", "push", "file_watch", "command_poll", "command_stream"],
            "activation": ["while_visible", "always", "manual"],
            "actions": [
                "command", "mcp_tool", "refresh_source", "set_data",
                "run_workflow", "run_scheduled_task", "trigger_event",
            ],
            "bindings": ["{{data.source.path}}", "{{state.controlName}}"],
            "subViews": {
                "local": "use child components",
                "referenced": "viewId always requires an explicit immutable revision",
            },
            "surfaces": {
                "inline": "immutable layout revision pinned to one chat session",
                "sidebar": "durable page with optimistic revisions and soft delete",
            },
            "limits": {
                "markupBytes": MAX_SOURCE_BYTES,
                "nodes": MAX_NODES,
                "depth": MAX_DEPTH,
            },
            "example": (
                '<stack gap="3"><heading level="2">Overview</heading>'
                '<metric label="CPU" value.bind="{{data.host.cpu}}" unit="%" />'
                '<button action="refresh">Refresh</button></stack>'
            ),
        }

    async def ui_create_view(
        title: str,
        markup: str,
        surface: str = "sidebar",
        description: str = "",
        icon: str | None = None,
        session_id: str | None = None,
        initial_data: dict[str, Any] | None = None,
        sources: dict[str, Any] | None = None,
        actions: dict[str, Any] | None = None,
        scripts: dict[str, str] | None = None,
        assets_base64: dict[str, str] | None = None,
        visibility: str | None = None,
        sidebar_order: int = 0,
        sidebar_group: str | None = None,
        expires_at: int | None = None,
        frozen: bool = False,
    ) -> dict[str, Any]:
        """Create a validated OA-UI View.

        ``inline`` requires the exact current session_id and becomes layout-
        immutable; call ui_snapshot_to_chat afterwards. ``sidebar`` creates a
        durable page that can later be revised or deleted. Dynamic values
        belong in initial_data/sources, not interpolated into markup.
        """

        try:
            if surface == "inline" and not session_id:
                raise CustomViewInputError(
                    "inline Views require the exact current session_id"
                )
            access = await access_for_turn()
            effective_visibility = visibility or (
                "installation_shared"
                if access.principal_type == "agent" and access.device_id == "internal-runtime"
                else "private"
            )
            view = await service.create(
                access,
                surface=surface,
                title=title,
                description=description,
                icon=icon,
                markup=markup,
                session_id=session_id,
                initial_data=initial_data,
                sources=sources,
                actions=actions,
                scripts=decode_scripts(scripts),
                assets=decode_assets(assets_base64),
                visibility=effective_visibility,
                sidebar_order=sidebar_order,
                sidebar_group=sidebar_group,
                expires_at=expires_at,
                frozen=frozen,
            )
            return {"ok": True, "view": view}
        except Exception as exc:
            return safe_error(exc)

    async def ui_update_view(
        view_id: str,
        expected_revision: int,
        markup: str | None = None,
        title: str | None = None,
        description: str | None = None,
        icon: str | None = None,
        clear_icon: bool = False,
        actions: dict[str, Any] | None = None,
        scripts: dict[str, str] | None = None,
        assets_base64: dict[str, str] | None = None,
        visibility: str | None = None,
        sidebar_order: int | None = None,
        sidebar_group: str | None = None,
        clear_sidebar_group: bool = False,
        expires_at: int | None = None,
        clear_expires_at: bool = False,
        frozen: bool | None = None,
    ) -> dict[str, Any]:
        """Create a new revision of a sidebar View using optimistic locking.

        Read the latest view first and pass its revision. Inline definitions
        cannot be changed; only their data may be updated.
        """

        try:
            access = await access_for_turn()
            kwargs: dict[str, Any] = {"expected_revision": expected_revision}
            if markup is not None:
                kwargs["markup"] = markup
            if title is not None:
                kwargs["title"] = title
            if description is not None:
                kwargs["description"] = description
            if clear_icon:
                kwargs["icon"] = None
            elif icon is not None:
                kwargs["icon"] = icon
            if actions is not None:
                kwargs["actions"] = actions
            if scripts is not None:
                kwargs["scripts"] = decode_scripts(scripts)
            if assets_base64 is not None:
                kwargs["assets"] = decode_assets(assets_base64)
            if visibility is not None:
                kwargs["visibility"] = visibility
            if sidebar_order is not None:
                kwargs["sidebar_order"] = sidebar_order
            if clear_sidebar_group:
                kwargs["sidebar_group"] = None
            elif sidebar_group is not None:
                kwargs["sidebar_group"] = sidebar_group
            if clear_expires_at:
                kwargs["expires_at"] = None
            elif expires_at is not None:
                kwargs["expires_at"] = expires_at
            if frozen is not None:
                kwargs["frozen"] = frozen
            view = await service.update(view_id, access, **kwargs)
            return {"ok": True, "view": view}
        except Exception as exc:
            return safe_error(exc)

    async def ui_delete_view(view_id: str, expected_revision: int) -> dict[str, Any]:
        """Soft-delete a sidebar View at the expected revision."""

        try:
            access = await access_for_turn()
            await service.delete(view_id, access, expected_revision=expected_revision)
            return {"ok": True, "viewId": view_id}
        except Exception as exc:
            return safe_error(exc)

    async def ui_get_view(view_id: str, revision: int | None = None) -> dict[str, Any]:
        """Read one authorized View; revision pins an immutable historical layout."""

        try:
            view = await service.get(view_id, await access_for_turn(), revision=revision)
            return {"ok": True, "view": view}
        except Exception as exc:
            return safe_error(exc)

    async def ui_list_views(
        surface: str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List authorized Custom Views, optionally filtered by surface/session/text."""

        try:
            items, has_more = await service.list(
                await access_for_turn(),
                surface=surface,
                query=query,
                session_id=session_id,
                limit=limit,
                offset=offset,
            )
            return {
                "ok": True,
                "views": items,
                "hasMore": has_more,
                "nextOffset": offset + len(items) if has_more else None,
            }
        except Exception as exc:
            return safe_error(exc)

    async def ui_snapshot_to_chat(
        view_id: str,
        session_id: str,
        revision: int | None = None,
    ) -> dict[str, Any]:
        """Return the marker that embeds an existing inline View in this chat.

        Include the returned marker verbatim in a normal text response. The
        view must be inline and bound to ``session_id``; sidebar Views are
        linked from the sidebar and cannot be smuggled into channel replies.
        """

        try:
            access = await access_for_turn()
            view = await service.get(view_id, access, revision=revision)
            if view.get("surface") != "inline" or view.get("sessionId") != session_id:
                raise CustomViewInputError(
                    "snapshot-to-chat requires an inline View bound to this session"
                )
            selected = int(view["revision"])
            part = await service.resolve_inline_ref(
                view_id, selected, session_id=session_id, access=access,
            )
            if part is None:
                raise CustomViewNotFound("inline Custom View revision not found")
            marker = f"[OPENAGENT_UI:{view_id}@{selected}]"
            return {
                "ok": True,
                "marker": marker,
                "part": part,
                "instruction": (
                    "Include the marker verbatim exactly once in a normal textual reply. "
                    "Do not use it on Telegram, WhatsApp, Discord, Slack, webhook, or CLI."
                ),
            }
        except Exception as exc:
            return safe_error(exc)

    async def ui_set_data(
        view_id: str,
        key: str,
        value: Any,
        expected_version: int | None = None,
        expires_at: int | None = None,
        mode: str = "replace",
        max_items: int = 1000,
    ) -> dict[str, Any]:
        """Replace, shallow-merge, or ring-buffer append one named JSON value.

        ``append`` is bounded by ``max_items``; every mode notifies visible
        subscribers after the canonical checkpoint has been serialized.
        """

        try:
            access = await access_for_turn()
            item = await service.set_data(
                view_id,
                key,
                value,
                access,
                expected_version=expected_version,
                expires_at=expires_at,
                mode=mode,
                max_items=max_items,
            )
            return {"ok": True, "data": item}
        except Exception as exc:
            return safe_error(exc)

    async def ui_configure_source(
        view_id: str,
        key: str,
        driver: str,
        expected_revision: int,
        activation: str = "while_visible",
        config: dict[str, Any] | None = None,
        enabled: bool = True,
        expires_at: int | None = None,
        output_schema: dict[str, Any] | None = None,
        scripts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create or replace a sidebar source as a new View revision.

        Commands use argv arrays; never construct a shell string from UI data.
        A config may name a bundled script with ``script`` plus a literal
        ``args`` array; pass its UTF-8 contents in ``scripts``. Read the latest
        View first and pass its revision to detect concurrent edits.
        Inline source definitions are immutable after creation.
        """

        try:
            source = await service.configure_source(
                view_id,
                key,
                {
                    "driver": driver,
                    "activation": activation,
                    "config": config or {},
                    "enabled": enabled,
                    "expiresAt": expires_at,
                    "outputSchema": output_schema,
                },
                await access_for_turn(),
                expected_revision=expected_revision,
                scripts=decode_scripts(scripts),
            )
            return {"ok": True, "source": source}
        except Exception as exc:
            return safe_error(exc)

    async def ui_delete_source(
        view_id: str,
        key: str,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Delete a sidebar source by creating a new View revision."""

        try:
            revision = await service.delete_source(
                view_id,
                key,
                await access_for_turn(),
                expected_revision=expected_revision,
            )
            return {"ok": True, "viewId": view_id, "key": key, "revision": revision}
        except Exception as exc:
            return safe_error(exc)

    async def ui_refresh_source(view_id: str, key: str) -> dict[str, Any]:
        """Run one authorized source immediately and publish its next value."""

        try:
            result = await service.refresh_source(
                view_id, key, await access_for_turn()
            )
            return {"ok": True, **result}
        except Exception as exc:
            return safe_error(exc)

    async def ui_reactivate_view(
        view_id: str,
        expected_revision: int,
        expires_at: int | None = None,
    ) -> dict[str, Any]:
        """Reactivate a frozen/expired View without changing an inline layout."""

        try:
            view = await service.reactivate(
                view_id,
                await access_for_turn(),
                expected_revision=expected_revision,
                expires_at=expires_at,
            )
            return {"ok": True, "view": view}
        except Exception as exc:
            return safe_error(exc)

    async def ui_list_grants(view_id: str) -> dict[str, Any]:
        """List explicit grants for a View owned by the current principal."""

        try:
            result = await service.list_grants(view_id, await access_for_turn())
            return {"ok": True, **result}
        except Exception as exc:
            return safe_error(exc)

    async def ui_set_grant(
        view_id: str,
        principal_type: str,
        principal_id: str,
        permissions: list[str],
        expected_acl_version: int,
    ) -> dict[str, Any]:
        """Create or replace one View grant using optimistic ACL locking."""

        try:
            result = await service.set_grant(
                view_id,
                await access_for_turn(),
                principal_type=principal_type,
                principal_id=principal_id,
                permissions=permissions,
                expected_acl_version=expected_acl_version,
            )
            return {"ok": True, **result}
        except Exception as exc:
            return safe_error(exc)

    async def ui_delete_grant(
        view_id: str,
        principal_type: str,
        principal_id: str,
        expected_acl_version: int,
    ) -> dict[str, Any]:
        """Delete one View grant using optimistic ACL locking."""

        try:
            result = await service.delete_grant(
                view_id,
                await access_for_turn(),
                principal_type=principal_type,
                principal_id=principal_id,
                expected_acl_version=expected_acl_version,
            )
            return {"ok": True, **result}
        except Exception as exc:
            return safe_error(exc)

    return Toolkit(
        name="ui-manager",
        tools=[
            ui_get_schema,
            ui_create_view,
            ui_update_view,
            ui_delete_view,
            ui_get_view,
            ui_list_views,
            ui_snapshot_to_chat,
            ui_set_data,
            ui_configure_source,
            ui_delete_source,
            ui_refresh_source,
            ui_reactivate_view,
            ui_list_grants,
            ui_set_grant,
            ui_delete_grant,
        ],
    )


__all__ = ["build_runtime_toolkit"]
