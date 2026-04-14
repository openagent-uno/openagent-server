"""Explicit stdio transport lifecycle for MCP subprocesses."""

from __future__ import annotations

from typing import Any

import anyio
from anyio.streams.text import TextReceiveStream
from mcp import StdioServerParameters, types
from mcp.client.stdio import (
    _create_platform_compatible_process,
    _get_executable_command,
    _terminate_process_tree,
    get_default_environment,
)
from mcp.shared.message import SessionMessage


class ManagedStdioTransport:
    """Asyncio-friendly stdio transport wrapper for MCP subprocesses."""

    def __init__(self, server: StdioServerParameters):
        self.server = server
        self.process: Any | None = None
        self._task_group: anyio.abc.TaskGroup | None = None
        self.read_stream = None
        self.write_stream = None
        self._read_stream_writer = None
        self._write_stream_reader = None

    async def start(self):
        self._read_stream_writer, self.read_stream = anyio.create_memory_object_stream(0)
        self.write_stream, self._write_stream_reader = anyio.create_memory_object_stream(0)

        command = _get_executable_command(self.server.command)
        env = (
            {**get_default_environment(), **self.server.env}
            if self.server.env is not None
            else get_default_environment()
        )
        self.process = await _create_platform_compatible_process(
            command=command,
            args=self.server.args,
            env=env,
            cwd=self.server.cwd,
        )
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._stdout_reader)
        self._task_group.start_soon(self._stdin_writer)
        return self.read_stream, self.write_stream

    async def _stdout_reader(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        assert self._read_stream_writer is not None
        try:
            async with self._read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(
                    self.process.stdout,
                    encoding=self.server.encoding,
                    errors=self.server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await self._read_stream_writer.send(exc)
                            continue
                        await self._read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def _stdin_writer(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        assert self._write_stream_reader is not None
        try:
            async with self._write_stream_reader:
                async for session_message in self._write_stream_reader:
                    payload = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await self.process.stdin.send(
                        (payload + "\n").encode(
                            encoding=self.server.encoding,
                            errors=self.server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def aclose(self) -> None:
        process = self.process
        task_group = self._task_group
        read_stream = self.read_stream
        write_stream = self.write_stream
        read_writer = self._read_stream_writer
        write_reader = self._write_stream_reader

        self.process = None
        self._task_group = None
        self.read_stream = None
        self.write_stream = None
        self._read_stream_writer = None
        self._write_stream_reader = None

        try:
            if process and process.stdin:
                try:
                    await process.stdin.aclose()
                except Exception:
                    pass
        finally:
            if task_group is not None:
                task_group.cancel_scope.cancel()
                try:
                    await task_group.__aexit__(None, None, None)
                except Exception:
                    pass
            if process is not None and process.returncode is None:
                try:
                    await _terminate_process_tree(process, timeout_seconds=1.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            for stream in (read_stream, write_stream, read_writer, write_reader):
                if stream is None:
                    continue
                try:
                    await stream.aclose()
                except Exception:
                    pass
            if process is not None:
                try:
                    await process.aclose()
                except Exception:
                    pass
