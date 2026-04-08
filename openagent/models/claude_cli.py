"""Claude model via the Claude Code CLI (subprocess)."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

from openagent.models.base import BaseModel, ModelResponse, ToolCall


class ClaudeCLI(BaseModel):
    """Claude via the `claude` CLI tool.

    Requires `claude` to be installed and authenticated.
    Uses `claude -p` (--print) for non-interactive single-shot responses.
    Prompt is piped via stdin to avoid shell escaping issues.
    Uses --no-session-persistence to avoid loading old session context.

    MCP servers from OpenAgent are passed via --mcp-config so Claude CLI
    can use them alongside its own built-in MCPs.
    """

    def __init__(
        self,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict[str, dict] | None = None,
        permission_mode: str = "auto",
    ):
        self.model = model
        self.allowed_tools = allowed_tools or []
        self.mcp_servers = mcp_servers or {}
        self.permission_mode = permission_mode  # "auto", "bypass", or "default"
        self._mcp_config_path: str | None = None

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        """Set MCP server configs to pass to Claude CLI.

        Format: {"name": {"command": "node", "args": ["/path/to/server.js"]}, ...}
        """
        self.mcp_servers = servers
        self._mcp_config_path = None  # invalidate cached config file

    def _get_mcp_config_path(self) -> str | None:
        """Write MCP config to a temp file and return path."""
        if not self.mcp_servers:
            return None
        if self._mcp_config_path and Path(self._mcp_config_path).exists():
            return self._mcp_config_path

        config = {"mcpServers": self.mcp_servers}
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="openagent_mcp_",
            delete=False,
        )
        json.dump(config, tmp, indent=2)
        tmp.close()
        self._mcp_config_path = tmp.name
        return self._mcp_config_path

    def _build_cmd(self, output_format: str = "json") -> list[str]:
        """Build the base claude command."""
        cmd = ["claude", "-p", "--output-format", output_format, "--no-session-persistence"]
        if self.model:
            cmd.extend(["--model", self.model])

        # Permission mode: bypass all tool confirmations for agent use
        if self.permission_mode == "bypass":
            cmd.append("--dangerously-skip-permissions")

        for tool in self.allowed_tools:
            cmd.extend(["--allowedTools", tool])

        # Pass OpenAgent MCP servers to Claude CLI
        mcp_config = self._get_mcp_config_path()
        if mcp_config:
            cmd.extend(["--mcp-config", mcp_config])

        return cmd

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        # Build the prompt: system context + messages, all piped via stdin
        prompt_parts = []
        if system:
            prompt_parts.append(f"<system>\n{system}\n</system>")
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")

        prompt = "\n\n".join(prompt_parts)

        cmd = self._build_cmd("json")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=prompt.encode("utf-8"))

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else f"claude CLI exited with code {proc.returncode}"
            return ModelResponse(content=f"Error: {error_msg}")

        output = stdout.decode().strip()
        try:
            data = json.loads(output)
            text = data.get("result", output)
            return ModelResponse(
                content=text,
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
            )
        except json.JSONDecodeError:
            return ModelResponse(content=output)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        prompt_parts = []
        if system:
            prompt_parts.append(f"<system>\n{system}\n</system>")
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(msg.get("content", ""))

        prompt = "\n\n".join(prompt_parts)

        cmd = self._build_cmd("stream-json")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        async for line in proc.stdout:
            text = line.decode().strip()
            if not text:
                continue
            try:
                event = json.loads(text)
                if event.get("type") == "assistant" and "content" in event:
                    yield event["content"]
            except json.JSONDecodeError:
                yield text

        await proc.wait()
