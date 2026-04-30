"""Local Piper TTS — fallback when no cloud TTS row is configured.

Covers:

* ``is_available()`` returns False when piper isn't installed.
* ``synthesize()`` returns None gracefully when the package can't
  load (import path is missing / install failed).
* Voice-name parser converts ``en_US-amy-medium`` → the rhasspy
  ``piper-voices`` URL path correctly. Sentinel for the regression
  where we'd silently 404 against a malformed URL.
* ``resolve_tts_provider`` returns the synthetic local config when no
  DB row exists AND piper is available, ``None`` otherwise — the
  branch that toggles voice mode's "out of the box" experience.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

from ._framework import TestContext, test


# ── Helpers ─────────────────────────────────────────────────────────


class _FakeDBNoRows:
    """Stand-in for the gateway DB whose latest_audio_model is empty."""
    async def latest_audio_model(self, kind: str):
        return None


# ── Tests ───────────────────────────────────────────────────────────


@test("tts_local", "is_available reports False when piper not importable")
async def t_is_available_when_missing(_ctx: TestContext) -> None:
    from openagent.channels import tts_local

    # Force the import inside is_available() to fail by removing piper
    # from sys.modules and shadowing the import path. The function
    # catches ImportError and returns False.
    saved = sys.modules.pop("piper", None)
    sys.modules["piper"] = None  # forces ImportError on the inner import
    try:
        assert tts_local.is_available() is False, (
            "is_available must return False when piper import is shadowed"
        )
    finally:
        if saved is not None:
            sys.modules["piper"] = saved
        else:
            sys.modules.pop("piper", None)


@test("tts_local", "synthesize returns None when piper isn't loadable")
async def t_synthesize_no_backend(_ctx: TestContext) -> None:
    from openagent.channels import tts_local

    # Patch is_available to True but make the actual import fail —
    # mirrors a partial install where the package metadata exists but
    # the native ONNX runtime won't load.
    with patch.object(tts_local, "_load_voice", return_value=None):
        result = await tts_local.synthesize("Hello world", voice="en_US-amy-medium")
    assert result is None, f"expected None when load fails, got {result!r}"


@test("tts_local", "_voice_url_path parses canonical voice names correctly")
async def t_voice_url_path(_ctx: TestContext) -> None:
    from openagent.channels.tts_local import _voice_url_path

    # Standard cases — must match the rhasspy/piper-voices repo layout.
    assert _voice_url_path("en_US-amy-medium") == "en/en_US/amy/medium/en_US-amy-medium"
    assert _voice_url_path("it_IT-paola-medium") == "it/it_IT/paola/medium/it_IT-paola-medium"
    # Multi-word quality (rare but valid in the repo).
    assert _voice_url_path("en_GB-alan-low-quality") == "en/en_GB/alan/low-quality/en_GB-alan-low-quality"
    # Malformed inputs — must return None so callers can degrade
    # instead of issuing a broken HTTP request.
    assert _voice_url_path("nonsense") is None
    assert _voice_url_path("foo-bar") is None
    assert _voice_url_path("nolang-baz-medium") is None


@test("tts_local", "resolve_tts_provider returns local Piper config when piper is available")
async def t_resolve_tts_uses_piper(_ctx: TestContext) -> None:
    from openagent.channels import tts as tts_mod
    from openagent.channels import tts_local

    # No DB row + piper available → synthetic local config wins.
    with patch.object(tts_local, "is_available", return_value=True):
        cfg = await tts_mod.resolve_tts_provider(_FakeDBNoRows())
    assert cfg is not None, "expected a fallback config when piper is available"
    assert cfg.vendor == tts_mod.LOCAL_PIPER_VENDOR, cfg
    assert cfg.response_format == "wav", cfg
    assert cfg.voice_id, "voice_id should resolve to the default Piper voice"


@test("tts_local", "resolve_tts_provider returns None when no row AND no piper")
async def t_resolve_tts_none_when_no_backends(_ctx: TestContext) -> None:
    from openagent.channels import tts as tts_mod
    from openagent.channels import tts_local

    with patch.object(tts_local, "is_available", return_value=False):
        cfg = await tts_mod.resolve_tts_provider(_FakeDBNoRows())
    assert cfg is None, f"expected None when neither cloud row nor piper available: {cfg}"


@test("tts_local", "_resolve_voice_name picks language-matched voice when nothing pinned")
async def t_resolve_voice_name_language(_ctx: TestContext) -> None:
    from openagent.channels import tts_local

    # Wrap in patch so an existing OPENAGENT_PIPER_VOICE in the env
    # doesn't override our test (CI environments shouldn't set it,
    # but local dev might).
    with patch.dict("os.environ", {}, clear=False):
        # Direct env scrub — patch.dict doesn't unset existing keys.
        import os as _os
        _os.environ.pop("OPENAGENT_PIPER_VOICE", None)

        # No language → default English voice (today's behaviour).
        assert tts_local._resolve_voice_name(None) == tts_local.DEFAULT_VOICE
        # Italian transcription → Italian voice (the user complaint).
        assert tts_local._resolve_voice_name(None, language="it") == "it_IT-paola-medium"
        # ISO code with region — only the lang prefix matters.
        assert tts_local._resolve_voice_name(None, language="es-AR") == "es_ES-mls_9972-low"
        # Unknown language → fall back to default rather than crashing.
        assert tts_local._resolve_voice_name(None, language="xx") == tts_local.DEFAULT_VOICE


@test("tts_local", "explicit voice overrides language hint")
async def t_resolve_voice_name_explicit_wins(_ctx: TestContext) -> None:
    from openagent.channels import tts_local

    # Explicit voice must beat the language hint — covers the case
    # where a user wants a specific Piper voice across languages.
    assert tts_local._resolve_voice_name("en_GB-alan-medium", language="it") == "en_GB-alan-medium"


@test("tts_local", "OPENAGENT_PIPER_VOICE env var overrides language hint")
async def t_resolve_voice_name_env_wins(_ctx: TestContext) -> None:
    from openagent.channels import tts_local

    # Same priority as explicit > env > language > default — the env
    # var should still win when no explicit voice is passed but a
    # language hint is. Otherwise users who set the env var would see
    # it silently ignored as soon as language plumbing engaged.
    with patch.dict("os.environ", {"OPENAGENT_PIPER_VOICE": "en_GB-alan-medium"}):
        assert tts_local._resolve_voice_name(None, language="it") == "en_GB-alan-medium"
