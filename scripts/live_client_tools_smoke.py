#!/usr/bin/env python3
"""One-shot acceptance smoke using a real model and a real client broker.

This is deliberately opt-in because CI has no paid-provider credential.  It
creates an isolated capability-host home, exposes only that temporary client to
an actual :class:`Agent`, asks the configured model to write and read a
sentinel through ``client:filesystem``, and then proves routing, execution-host
metadata and content-free local audit records.  It never touches the user's
persistent local-tools consent or broker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any


class _Wire:
    closed = False

    async def close(self, **_kwargs: Any) -> None:
        self.closed = True


async def _wait_for_broker(server: Any, task: asyncio.Task[Any]) -> None:
    from openagent_host_tools.local_broker import LocalBrokerClient

    deadline = asyncio.get_running_loop().time() + 10.0
    last_error: BaseException | None = None
    while asyncio.get_running_loop().time() < deadline:
        if task.done():
            await task
        probe = LocalBrokerClient(server.paths)
        try:
            await probe.connect()
            await probe.close()
            return
        except (OSError, EOFError, ConnectionRefusedError, FileNotFoundError) as exc:
            last_error = exc
            await probe.close()
            await asyncio.sleep(0.02)
    raise RuntimeError(f"isolated client broker did not start: {last_error}")


def _audit_rows(path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM audit ORDER BY seq")
        ]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from openagent_host_tools import HostPaths, LocalCapabilityClient
    from openagent_host_tools.local_broker import LocalBrokerServer
    from openagent_host_tools.types import HostError

    from src.core.agent import Agent
    from src.core.execution_origin import execution_origin_scope
    from src.gateway.capabilities import CapabilityRegistry
    from src.mcp.pool import MCPPool
    from src.models.native_provider import NativeProvider

    api_key = os.environ.get(args.api_key_env) or "local"
    provider_name = args.model.split(":", 1)[0]
    if not provider_name or ":" not in args.model:
        raise ValueError("--model must be a canonical provider:model id")
    if provider_name == "deterministic":
        raise ValueError("the live acceptance smoke refuses deterministic models")

    class RecordingRegistry(CapabilityRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.results: list[dict[str, Any]] = []

        async def call_tool(self, *call_args: Any, **call_kwargs: Any) -> Any:
            value = await super().call_tool(*call_args, **call_kwargs)
            if isinstance(value, dict):
                self.results.append(value)
            return value

    token = "OA_LIVE_CLIENT_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="openagent-live-agent-smoke-") as raw:
        root = Path(raw)
        paths = HostPaths.discover(root / "client")
        target = root / "client-sentinel.txt"
        broker = LocalBrokerServer(paths)
        broker_task = asyncio.create_task(
            broker.run(), name="live-client-tools-smoke-broker",
        )
        host: LocalCapabilityClient | None = None
        agent: Agent | None = None
        delivery_tasks: set[asyncio.Task[Any]] = set()
        try:
            await _wait_for_broker(broker, broker_task)
            host = LocalCapabilityClient(paths=paths)
            await host.start()
            await host.set_consent(True)
            servers = await host.catalog()
            registry = RecordingRegistry()
            connection: dict[str, Any] = {}
            network_id = "live-smoke-network"
            instance_id = "live-smoke-cli"
            principal = {
                "network_id": network_id,
                "account_id": network_id,
                "device_id": "live-smoke-device",
                "client_instance_id": instance_id,
                "generation": 1,
            }

            async def deliver(payload: dict[str, Any]) -> None:
                if payload.get("type") == "client_tool_cancel":
                    await host.cancel(str(payload.get("call_id") or ""))
                    return
                if payload.get("type") != "client_tool_call":
                    return
                try:
                    result = await host.call(
                        str(payload["server"]),
                        str(payload["tool"]),
                        dict(payload.get("args") or {}),
                        principal=principal,
                        call_id=str(payload["call_id"]),
                        idempotency_key=payload.get("idempotency_key"),
                        deadline_ms=payload.get("deadline_ms"),
                        arguments_sha256=payload.get("arguments_sha256"),
                    )
                    frame = {
                        "type": "client_tool_result",
                        "call_id": payload["call_id"],
                        "generation": payload["generation"],
                        "result": result.to_wire(),
                    }
                except HostError as exc:
                    frame = {
                        "type": "client_tool_result",
                        "call_id": payload["call_id"],
                        "generation": payload["generation"],
                        "error": exc.to_wire(),
                    }
                registry.resolve_result(connection["value"], frame)

            async def send_json(_ws: Any, payload: dict[str, Any]) -> bool:
                task = asyncio.create_task(deliver(payload))
                delivery_tasks.add(task)
                task.add_done_callback(delivery_tasks.discard)
                return True

            conn = await registry.register(
                device_id="live-smoke-device",
                account_id=network_id,
                client_instance_id=instance_id,
                generation=1,
                device_label="Live Smoke CLI",
                ws=_Wire(),
                send_json=send_json,
                servers=servers,
                network_id=network_id,
            )
            connection["value"] = conn
            origin = registry.origin_for("live-smoke-device", instance_id)
            if origin is None:
                raise RuntimeError("trusted live-smoke origin was not registered")

            pool = MCPPool.from_config(
                mcp_config=[{"builtin": "tool-search"}],
                include_defaults=False,
            )
            provider_config = [{
                "name": provider_name,
                "framework": "api-based",
                "api_key": api_key,
                "base_url": args.base_url,
            }]
            model = NativeProvider(
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                providers_config=provider_config,
                db_path=str(root / "runtime.db"),
            )
            agent = Agent(
                name="live-client-tools-smoke",
                model=model,
                system_prompt=(
                    "This is an acceptance smoke. Obey the user exactly. "
                    "Client-local paths are never server paths."
                ),
                mcp_pool=pool,
                memory=None,
            )
            prompt = (
                "Use the local tool host on this client. You MUST call "
                "tool-search.call_tool with server='client:filesystem' and "
                f"tool='write_file' to write the exact supplied sentinel to {str(target)!r}. "
                f"The exact sentinel is {token!r}. Then call server='client:filesystem', "
                "tool='read_text_file' on the same path and verify it. Never use "
                "server:filesystem. Only after both calls succeed, reply with "
                "LIVE_AGENT_CLIENT_OK."
            )
            with execution_origin_scope(origin):
                response = await asyncio.wait_for(
                    agent.run(
                        prompt,
                        user_id="live-smoke",
                        session_id="chat:live-client-tools",
                    ),
                    timeout=args.timeout,
                )

            if target.read_text(encoding="utf-8") != token:
                raise AssertionError("the real agent did not write the exact client sentinel")
            if "LIVE_AGENT_CLIENT_OK" not in response:
                raise AssertionError(f"unexpected real-agent response: {response!r}")
            model_id = agent.last_response_meta(
                "chat:live-client-tools",
            ).get("model")
            if not model_id or str(model_id).startswith("deterministic"):
                raise AssertionError(f"non-live response model: {model_id!r}")

            rows = _audit_rows(paths.audit_db)
            filesystem_tools = [
                row["tool"]
                for row in rows
                if row["server"] == "filesystem" and row["outcome"] == "success"
            ]
            if filesystem_tools[-2:] != ["write_file", "read_text_file"]:
                raise AssertionError(f"unexpected client audit sequence: {filesystem_tools}")
            hosts = [
                result.get("execution_host")
                for result in registry.results
                if isinstance(result.get("execution_host"), dict)
            ]
            if len(hosts) < 2 or any(
                host_info.get("kind") != "client"
                or host_info.get("client_instance_id") != instance_id
                for host_info in hosts[-2:]
            ):
                raise AssertionError(f"wrong execution host metadata: {hosts}")
            for audit_file in paths.internal.glob("audit.sqlite3*"):
                if token.encode("utf-8") in audit_file.read_bytes():
                    raise AssertionError(f"audit leaked sentinel content: {audit_file.name}")

            return {
                "ok": True,
                "model": model_id,
                "response_marker": "LIVE_AGENT_CLIENT_OK",
                "client_tools": filesystem_tools,
                "execution_host": hosts[-1],
                "audit_rows": len(rows),
            }
        finally:
            if delivery_tasks:
                await asyncio.gather(*delivery_tasks, return_exceptions=True)
            if agent is not None:
                with suppress(Exception):
                    await agent.shutdown()
            if host is not None:
                with suppress(Exception):
                    await host.set_consent(False)
                with suppress(Exception):
                    await host.close()
            broker_task.cancel()
            await asyncio.gather(broker_task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible /v1 endpoint for a real model provider",
    )
    parser.add_argument(
        "--model",
        default="local:claude-sonnet-4-6",
        help="Canonical provider:model runtime id",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAGENT_LIVE_SMOKE_API_KEY",
        help="Environment variable containing the provider key (never a CLI value)",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), sort_keys=True))


if __name__ == "__main__":
    main()
