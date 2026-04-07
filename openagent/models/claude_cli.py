"""Claude model via the Claude Code CLI (subprocess)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from openagent.models.base import BaseModel, ModelResponse, ToolCall


class ClaudeCLI(BaseModel):
    """Claude via the `claude` CLI tool.

    Requires `claude` to be installed and authenticated.
    Uses `claude -p` (--print) for non-interactive single-shot responses.
    Prompt is piped via stdin to avoid shell escaping issues with long prompts.
    Uses --no-session-persistence to avoid loading old session context.
    """

    def __init__(self, model: str | None = None, allowed_tools: list[str] | None = None):
        self.model = model
        self.allowed_tools = allowed_tools or []

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        # Build the prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")

        prompt = "\n\n".join(prompt_parts)

        # Build command
        cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence"]
        if self.model:
            cmd.extend(["--model", self.model])
        if system:
            cmd.extend(["--append-system-prompt", system])
        for tool in self.allowed_tools:
            cmd.extend(["--allowedTools", tool])

        # Pipe prompt via stdin (avoids shell escaping issues)
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
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(msg.get("content", ""))

        prompt = "\n\n".join(prompt_parts)

        cmd = ["claude", "-p", "--output-format", "stream-json", "--no-session-persistence"]
        if self.model:
            cmd.extend(["--model", self.model])
        if system:
            cmd.extend(["--append-system-prompt", system])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Send prompt and close stdin
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
