"""Media generation MCP server.

Exposes ``generate_image`` and ``generate_video`` to the agent so it can
render visuals inline (Telegram replies, vault notes, workflow outputs).
Subprocess MCP (stdio transport) — same pattern as the scheduler.

Providers:
  • image:  ``openai`` (DALL·E-3 / gpt-image-1) — uses ``OPENAI_API_KEY``
  • video:  ``fal``     (pixverse / runway etc.)  — uses ``FAL_KEY``

Both providers are auto-detected from env vars. When a provider's key is
missing, the corresponding tool surfaces a clear "not configured" error
instead of silently failing — the model can fall back or tell the user.

Files are written under ``~/.cache/openagent/media/<timestamp>-<hash>.{ext}``
so disk usage stays predictable; the tool returns the local path AND the
remote URL (when present) so callers can pick.

Off by default. Enable by mounting in ``openagent.yaml``:
  mcps:
    - builtin: media-gen
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("media-gen")


_CACHE_DIR = Path(os.path.expanduser("~/.cache/openagent/media"))


def _cache_path(prefix: str, prompt: str, ext: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()[:8]
    ts = int(time.time())
    return _CACHE_DIR / f"{prefix}-{ts}-{h}.{ext}"


async def _save_url_to_file(url: str, dest: Path, timeout_s: float = 60.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(url)
            r.raise_for_status()
            dest.write_bytes(r.content)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("media-gen download failed: %s", e)
        return False


# ── Image generation ─────────────────────────────────────────────────


@mcp.tool()
async def generate_image(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "standard",
) -> dict[str, Any]:
    """Generate an image from a text prompt via OpenAI.

    Args:
        prompt: What to draw — natural-language description.
        model: ``gpt-image-1`` (recommended) or ``dall-e-3``.
        size: ``1024x1024`` / ``1792x1024`` / ``1024x1792``.
        quality: ``standard`` (faster) or ``hd`` (slower, sharper).

    Returns ``{ok, local_path, remote_url, model, size}``. ``ok`` is
    False with a ``reason`` field when the OpenAI key is missing or
    the request failed."""
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return {
            "ok": False,
            "reason": "OPENAI_API_KEY not set — configure under "
                      "media_gen.openai.api_key or as env var.",
        }
    try:
        # Lazy import so missing-dependency installs don't fail import-time.
        from openai import AsyncOpenAI  # type: ignore
    except Exception as e:
        return {"ok": False, "reason": f"openai SDK missing: {e}"}

    client = AsyncOpenAI(api_key=api_key)
    try:
        resp = await client.images.generate(
            model=model,
            prompt=prompt[:4000],
            size=size,
            quality=quality,
            n=1,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"OpenAI image generation failed: {e}"}

    if not resp.data:
        return {"ok": False, "reason": "OpenAI returned no images"}
    item = resp.data[0]
    url = getattr(item, "url", None)
    b64 = getattr(item, "b64_json", None)
    local = _cache_path("img", prompt, "png")

    if b64:
        import base64
        try:
            local.write_bytes(base64.b64decode(b64))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"b64 decode failed: {e}"}
    elif url:
        if not await _save_url_to_file(url, local):
            return {"ok": False, "reason": "failed to download generated image"}
    else:
        return {"ok": False, "reason": "OpenAI returned neither url nor b64"}

    return {
        "ok": True,
        "local_path": str(local),
        "remote_url": url,
        "model": model,
        "size": size,
    }


# ── Video generation ─────────────────────────────────────────────────


@mcp.tool()
async def generate_video(
    prompt: str,
    model: str = "fal-ai/pixverse/v3.5/text-to-video",
    duration_s: int = 5,
) -> dict[str, Any]:
    """Generate a short video from a text prompt via Fal.

    Args:
        prompt: What to render.
        model: A Fal model id (default Pixverse v3.5 text-to-video).
        duration_s: Length in seconds (most models cap at 5-8).

    Returns ``{ok, local_path, remote_url}`` with ``ok=False`` and a
    ``reason`` when the Fal key is missing or generation fails."""
    api_key = (os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY") or "").strip()
    if not api_key:
        return {
            "ok": False,
            "reason": "FAL_KEY not set — configure under "
                      "media_gen.fal.api_key or as env var.",
        }
    # Minimal direct HTTP submit — keeps the dependency surface small
    # (no fal-client SDK requirement). For long-running jobs we poll
    # the queue endpoint, capped at ``_QUEUE_TIMEOUT_S``.
    _QUEUE_TIMEOUT_S = 180
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt[:2000], "duration": str(duration_s)}
    submit_url = f"https://queue.fal.run/{model}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(submit_url, headers=headers, json=payload)
            r.raise_for_status()
            queue = r.json()
            status_url = queue.get("status_url")
            response_url = queue.get("response_url")
            if not status_url or not response_url:
                return {"ok": False, "reason": f"Fal returned no status_url: {queue}"}
            # Poll until completion.
            deadline = time.monotonic() + _QUEUE_TIMEOUT_S
            while time.monotonic() < deadline:
                sr = await client.get(status_url, headers=headers)
                sr.raise_for_status()
                status_payload = sr.json()
                if status_payload.get("status") == "COMPLETED":
                    final = await client.get(response_url, headers=headers)
                    final.raise_for_status()
                    data = final.json()
                    video = (data.get("video") or {})
                    url = video.get("url") or data.get("url")
                    if not url:
                        return {"ok": False, "reason": f"Fal completed without video url: {data}"}
                    local = _cache_path("vid", prompt, "mp4")
                    if not await _save_url_to_file(url, local, timeout_s=120):
                        return {"ok": False, "reason": "failed to download generated video"}
                    return {"ok": True, "local_path": str(local), "remote_url": url}
                if status_payload.get("status") == "FAILED":
                    return {"ok": False, "reason": f"Fal job failed: {status_payload}"}
                await asyncio.sleep(2.0)
        return {"ok": False, "reason": "Fal job timed out"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"Fal request failed: {e}"}


def main() -> None:
    """Entry point matched by ``builtins.py`` python_module pattern."""
    mcp.run()


if __name__ == "__main__":
    main()
