"""Regression coverage for websocket-scoped interactive PTYs.

Desktop and CLI commonly use the same device certificate.  The certificate
is authentication/audit identity, not terminal ownership: each chat websocket
must own an independent PTY namespace even when both clients choose the same
``terminal_id``.
"""

from __future__ import annotations

import base64
import inspect
import json
from types import SimpleNamespace
from typing import Any

from ._framework import TestContext, test


class _FakeTerminalSession:
    """In-memory stand-in that exercises TerminalManager without a real PTY."""

    _next_pid = 1000

    def __init__(
        self,
        *,
        terminal_id: str,
        connection_id: str,
        device_id: str,
        on_output,
        on_exit,
        shell: str | None = None,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        **_kwargs: Any,
    ) -> None:
        self.terminal_id = terminal_id
        self.connection_id = connection_id
        self.device_id = device_id
        self._on_output = on_output
        self._on_exit = on_exit
        self.shell = shell or "/bin/sh"
        self.cwd = cwd or "/tmp"
        self.cols = cols
        self.rows = rows
        self.writes: list[bytes] = []
        self.signals: list[str] = []
        self.started = False
        self.closed = False
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid

    @property
    def is_running(self) -> bool:
        return self.started and not self.closed

    async def start(self) -> None:
        self.started = True

    def rebind(self, on_output, on_exit) -> None:
        self._on_output = on_output
        self._on_exit = on_exit

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def resize(self, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows

    def send_signal(self, name: str) -> None:
        self.signals.append(name)

    async def close(self) -> None:
        self.closed = True


class _Request(dict):
    def __init__(self, gateway, cert, *, query: dict[str, str] | None = None) -> None:
        super().__init__(device_cert=cert)
        self.app = {"gateway": gateway}
        self.query = query or {}
        self.headers: dict[str, str] = {}


@test(
    "gateway_terminals",
    "Desktop and CLI sharing a device own independent PTY namespaces",
)
async def t_same_device_connections_do_not_collide(ctx: TestContext) -> None:
    import src.gateway.terminals as terminal_module
    from src.gateway import protocol as P
    from src.gateway.server import Gateway

    original_session = terminal_module.TerminalSession
    original_supported = terminal_module.PTY_SUPPORTED
    terminal_module.TerminalSession = _FakeTerminalSession
    terminal_module.PTY_SUPPORTED = True
    try:
        gateway = Gateway.__new__(Gateway)
        gateway.terminals = terminal_module.TerminalManager()
        sent: dict[object, list[dict[str, Any]]] = {}

        async def _capture(ws, payload: dict[str, Any]) -> None:
            sent.setdefault(ws, []).append(payload)

        gateway._safe_ws_send_json = _capture
        desktop_ws = object()
        cli_ws = object()
        device_id = "same-device-certificate"
        terminal_id = "same-client-generated-id"
        open_frame = {
            "type": P.TERMINAL_OPEN,
            "terminal_id": terminal_id,
            "cols": 80,
            "rows": 24,
        }

        await gateway._handle_terminal_frame(
            desktop_ws, "desktop-connection", open_frame, device_id=device_id,
        )
        await gateway._handle_terminal_frame(
            cli_ws, "cli-connection", open_frame, device_id=device_id,
        )

        desktop = gateway.terminals.get("desktop-connection", terminal_id)
        cli = gateway.terminals.get("cli-connection", terminal_id)
        assert desktop is not None and cli is not None
        assert desktop is not cli, "the second websocket rebound the first PTY"
        assert desktop.device_id == cli.device_id == device_id
        assert gateway.terminals.count() == 2
        assert gateway.terminals.list_for_connection("desktop-connection") == [desktop]
        assert gateway.terminals.list_for_connection("cli-connection") == [cli]

        await gateway._handle_terminal_frame(
            desktop_ws,
            "desktop-connection",
            {
                "type": P.TERMINAL_INPUT,
                "terminal_id": terminal_id,
                "data": base64.b64encode(b"desktop").decode("ascii"),
            },
            device_id=device_id,
        )
        await gateway._handle_terminal_frame(
            cli_ws,
            "cli-connection",
            {
                "type": P.TERMINAL_INPUT,
                "terminal_id": terminal_id,
                "data": base64.b64encode(b"cli").decode("ascii"),
            },
            device_id=device_id,
        )
        await gateway._handle_terminal_frame(
            desktop_ws,
            "desktop-connection",
            {
                "type": P.TERMINAL_RESIZE,
                "terminal_id": terminal_id,
                "cols": 132,
                "rows": 40,
            },
            device_id=device_id,
        )
        await gateway._handle_terminal_frame(
            cli_ws,
            "cli-connection",
            {
                "type": P.TERMINAL_SIGNAL,
                "terminal_id": terminal_id,
                "signal": "INT",
            },
            device_id=device_id,
        )
        assert desktop.writes == [b"desktop"]
        assert cli.writes == [b"cli"]
        assert (desktop.cols, desktop.rows) == (132, 40)
        assert (cli.cols, cli.rows) == (80, 24)
        assert desktop.signals == []
        assert cli.signals == ["INT"]

        # An explicit close from Desktop must leave the same-id CLI PTY alive.
        await gateway._handle_terminal_frame(
            desktop_ws,
            "desktop-connection",
            {"type": P.TERMINAL_CLOSE, "terminal_id": terminal_id},
            device_id=device_id,
        )
        assert desktop.closed is True
        assert gateway.terminals.get("desktop-connection", terminal_id) is None
        assert gateway.terminals.get("cli-connection", terminal_id) is cli
        assert cli.closed is False

        # The websocket-finally path is connection-scoped as well. Give
        # Desktop another shell, then disconnect it; CLI must still survive.
        await gateway._handle_terminal_frame(
            desktop_ws,
            "desktop-connection",
            {**open_frame, "terminal_id": "desktop-second"},
            device_id=device_id,
        )
        assert await gateway.terminals.close_for_connection("desktop-connection") == 1
        assert gateway.terminals.get("cli-connection", terminal_id) is cli
        assert cli.closed is False

        # Pin the actual gateway cleanup callsite, not only manager behaviour.
        handler_source = inspect.getsource(Gateway._handle_ws)
        assert "close_for_connection(connection_id)" in handler_source
        assert "close_for_client(client_id)" not in handler_source

        await gateway.terminals.close_all()
    finally:
        terminal_module.TerminalSession = original_session
        terminal_module.PTY_SUPPORTED = original_supported


@test(
    "gateway_terminals",
    "terminal REST listing never mixes simultaneous same-device connections",
)
async def t_terminal_list_is_connection_scoped(ctx: TestContext) -> None:
    from src.gateway.api.terminals import handle_list

    desktop = SimpleNamespace(
        terminal_id="desktop-terminal",
        pid=101,
        shell="/bin/zsh",
        cwd="/desktop",
        cols=80,
        rows=24,
        is_running=True,
    )
    cli = SimpleNamespace(
        terminal_id="cli-terminal",
        pid=202,
        shell="/bin/bash",
        cwd="/cli",
        cols=120,
        rows=40,
        is_running=True,
    )

    class _TerminalLists:
        def list_for_connection(self, connection_id: str):
            return {
                "desktop-connection": [desktop],
                "cli-connection": [cli],
            }.get(connection_id, [])

    gateway = SimpleNamespace(
        terminals=_TerminalLists(),
        _chat_client_devices={
            "desktop-connection": "device-a",
            "cli-connection": "device-a",
            "other-connection": "device-b",
        },
    )
    cert = SimpleNamespace(device_pubkey_hex="device-a")

    # Legacy request is deliberately empty when the same certificate has two
    # active sockets: guessing would leak the sibling client's terminal list.
    ambiguous = await handle_list(_Request(gateway, cert))
    ambiguous_payload = json.loads(ambiguous.text)
    assert ambiguous_payload == {"connection_id": None, "terminals": []}

    exact = await handle_list(_Request(
        gateway, cert, query={"connection_id": "desktop-connection"},
    ))
    exact_payload = json.loads(exact.text)
    assert exact_payload["connection_id"] == "desktop-connection"
    assert [item["terminal_id"] for item in exact_payload["terminals"]] == [
        "desktop-terminal",
    ]

    forged = await handle_list(_Request(
        gateway, cert, query={"connection_id": "other-connection"},
    ))
    assert json.loads(forged.text) == {"terminals": []}
