"""Model input-capability routing and bounded document extraction.

``models.metadata_json.input_modalities`` is the source of truth for what a
configured LLM can receive natively.  The dispatcher uses this module before a
provider is invoked so unsupported media is never silently dropped (or handed
to a text-only API which will reject it).

Text-like files and PDFs have a deliberately small, local fallback pipeline:
their text is extracted into the latest user message and the binary ``files``
argument is removed.  Images, audio, video, and non-extractable documents must
have a natively compatible configured model.
"""

from __future__ import annotations

import asyncio
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


INPUT_MODALITIES = frozenset({"text", "image", "audio", "video", "file"})
_MODALITY_ALIASES = {
    "images": "image",
    "audios": "audio",
    "videos": "video",
    "files": "file",
    "document": "file",
    "documents": "file",
}

# Extraction is a fallback for models that do not accept native documents.
# Keep the allow-list narrow so binary data is never decoded into garbage.
_TEXT_MIMES = frozenset({
    "application/json",
    "application/ld+json",
    "application/x-javascript",
    "application/x-python",
    "application/xml",
    "application/x-yaml",
    "text/javascript",
    "text/x-python",
    "text/plain",
    "text/html",
    "text/css",
    "text/markdown",
    "text/csv",
    "text/xml",
    "text/rtf",
    "text/yaml",
})
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".ndjson", ".csv",
    ".tsv", ".xml", ".html", ".htm", ".css", ".py", ".pyi", ".js",
    ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".log", ".sh", ".zsh", ".bash", ".sql",
    ".rs", ".go", ".java", ".kt", ".swift", ".c", ".h", ".cc", ".cpp",
    ".hpp", ".rb", ".php", ".scala", ".lua", ".r", ".graphql",
})

# Bound both parser work and context growth.  These are per-turn totals; a
# single pathological upload cannot monopolise the event loop or model window.
MAX_EXTRACT_SOURCE_BYTES = 10 * 1024 * 1024
MAX_EXTRACT_TEXT_BYTES = 256 * 1024
MAX_EXTRACT_CHARS_PER_FILE = 100_000
MAX_EXTRACT_CHARS_TOTAL = 200_000
MAX_PDF_PAGES = 32


class MediaCapabilityError(RuntimeError):
    """A configured model cannot consume the turn's media safely."""


@dataclass(frozen=True)
class PreparedMedia:
    """A provider-ready turn after capability checks and optional extraction."""

    messages: list[dict[str, Any]]
    files: list[Any] | None
    images: list[Any] | None
    audio: list[Any] | None
    videos: list[Any] | None
    native_modalities: frozenset[str]
    extracted_files: int = 0


@dataclass(frozen=True)
class SelectedMediaModel:
    """The compatible catalog entry plus the exact payload it should receive."""

    entry: Any
    prepared: PreparedMedia
    fell_back: bool


def normalize_input_modalities(value: Any, *, require_text: bool = True) -> list[str]:
    """Validate and canonicalise a metadata ``input_modalities`` value.

    The persisted representation is a deterministic list.  Singular and plural
    spellings are accepted at API boundaries, but unknown values fail loudly so
    a typo cannot make the dispatcher claim a capability the model lacks.
    """
    if isinstance(value, str):
        raw_values: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_values = value
    else:
        raise ValueError("input_modalities must be a list of modality names")

    normalized: set[str] = set()
    for raw in raw_values:
        item = str(raw or "").strip().lower()
        if not item:
            continue
        item = _MODALITY_ALIASES.get(item, item)
        if item not in INPUT_MODALITIES:
            raise ValueError(
                f"unknown input modality {item!r}; expected one of "
                f"{sorted(INPUT_MODALITIES)}"
            )
        normalized.add(item)
    if require_text and "text" not in normalized:
        raise ValueError("LLM input_modalities must include 'text'")
    if not normalized:
        raise ValueError("input_modalities cannot be empty")
    return [item for item in ("text", "image", "audio", "video", "file") if item in normalized]


def default_input_modalities(provider: str, model: str) -> list[str]:
    """Conservative capabilities for legacy/new rows without metadata.

    Unknown and local models default to text only.  We opt models into binary
    modalities only for stable, recognisable multimodal families; operators can
    override this explicitly in ``metadata_json`` when a proxy or fine-tune has
    different capabilities.
    """
    p = str(provider or "").strip().lower()
    m = str(model or "").strip().lower()

    # OpenRouter model ids may retain their upstream ``vendor/model`` prefix.
    if p == "openrouter" and "/" in m:
        upstream, upstream_model = m.split("/", 1)
        upstream = {"x-ai": "xai", "mistralai": "mistral"}.get(upstream, upstream)
        return default_input_modalities(upstream, upstream_model)

    modalities = {"text"}
    if p == "google" and "gemini" in m:
        modalities.update({"image", "audio", "video", "file"})
    elif p == "anthropic" and "claude" in m and any(
        marker in m for marker in ("claude-3", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4")
    ):
        modalities.update({"image", "file"})
    elif p == "openai" and any(
        marker in m for marker in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")
    ):
        modalities.update({"image", "file"})
        if "audio" in m or "realtime" in m:
            modalities.add("audio")
    elif p == "mistral" and any(marker in m for marker in ("pixtral", "vision")):
        modalities.add("image")
    elif p == "qwen" and any(marker in m for marker in ("-vl", "qwen-vl", "vision")):
        modalities.add("image")
        if "video" in m:
            modalities.add("video")
    elif p == "zai" and any(marker in m for marker in ("glm-4v", "glm-4.5v", "vision", "-vl")):
        modalities.add("image")
    elif p == "moonshot" and "vision" in m:
        modalities.add("image")

    return [item for item in ("text", "image", "audio", "video", "file") if item in modalities]


def normalize_model_metadata(
    metadata: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    strict: bool = True,
) -> dict[str, Any]:
    """Return metadata with a persisted, validated capability declaration."""
    out = dict(metadata or {})
    raw = out.get("input_modalities")
    if raw is None:
        out["input_modalities"] = default_input_modalities(provider, model)
        return out
    try:
        out["input_modalities"] = normalize_input_modalities(raw)
    except ValueError:
        if strict:
            raise
        out["input_modalities"] = default_input_modalities(provider, model)
    return out


def input_modalities_for(entry: Any) -> frozenset[str]:
    """Resolve one catalog entry's declared or inferred capabilities."""
    metadata = dict(getattr(entry, "metadata", None) or {})
    raw = metadata.get("input_modalities")
    if raw is None:
        raw = default_input_modalities(
            getattr(entry, "provider", ""), getattr(entry, "model_id", ""),
        )
    try:
        return frozenset(normalize_input_modalities(raw))
    except ValueError:
        return frozenset(default_input_modalities(
            getattr(entry, "provider", ""), getattr(entry, "model_id", ""),
        ))


def native_modalities_for_payload(
    *,
    files: Sequence[Any] | None = None,
    images: Sequence[Any] | None = None,
    audio: Sequence[Any] | None = None,
    videos: Sequence[Any] | None = None,
) -> frozenset[str]:
    required: set[str] = set()
    if files:
        required.add("file")
    if images:
        required.add("image")
    if audio:
        required.add("audio")
    if videos:
        required.add("video")
    return frozenset(required)


def _file_name(file: Any) -> str:
    value = (
        getattr(file, "filename", None)
        or getattr(file, "name", None)
        or (Path(str(getattr(file, "filepath", ""))).name if getattr(file, "filepath", None) else None)
        or "attachment"
    )
    return str(value)


def _file_mime(file: Any) -> str:
    mime = str(getattr(file, "mime_type", None) or "").split(";", 1)[0].strip().lower()
    if mime:
        return mime
    suffix = Path(_file_name(file)).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".json":
        return "application/json"
    if suffix in _TEXT_EXTENSIONS:
        return "text/plain"
    return "application/octet-stream"


def _local_file_bytes(file: Any) -> bytes:
    """Read local/in-memory content with a hard source-size ceiling.

    Remote URL fetching is intentionally excluded from the extraction fallback:
    provider-native URL handling may be used by a compatible model, while a
    text-only route must not perform an unbounded server-side download.
    """
    content = getattr(file, "content", None)
    if isinstance(content, str):
        data = content.encode("utf-8")
    elif isinstance(content, (bytes, bytearray, memoryview)):
        data = bytes(content)
    elif getattr(file, "filepath", None):
        path = Path(str(file.filepath))
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise MediaCapabilityError(
                f"{_file_name(file)} cannot be read for text extraction: {exc}"
            ) from exc
        if size > MAX_EXTRACT_SOURCE_BYTES:
            raise MediaCapabilityError(
                f"{_file_name(file)} is too large for text extraction "
                f"({size} bytes; limit {MAX_EXTRACT_SOURCE_BYTES})"
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise MediaCapabilityError(
                f"{_file_name(file)} cannot be read for text extraction: {exc}"
            ) from exc
    elif getattr(file, "url", None):
        raise MediaCapabilityError(
            f"{_file_name(file)} is remote; bounded local extraction is unavailable"
        )
    else:
        raise MediaCapabilityError(f"{_file_name(file)} has no readable content")
    if len(data) > MAX_EXTRACT_SOURCE_BYTES:
        raise MediaCapabilityError(
            f"{_file_name(file)} is too large for text extraction "
            f"({len(data)} bytes; limit {MAX_EXTRACT_SOURCE_BYTES})"
        )
    return data


def _extract_file_text_sync(file: Any) -> tuple[str, bool]:
    """Extract one supported document. Returns ``(text, truncated)``."""
    filename = _file_name(file)
    mime = _file_mime(file)
    suffix = Path(filename).suffix.lower()
    data = _local_file_bytes(file)
    truncated = False

    if mime == "application/pdf" or suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency is mandatory
            raise MediaCapabilityError("PDF extraction is unavailable (pypdf is not installed)") from exc
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if getattr(reader, "is_encrypted", False):
                try:
                    unlocked = reader.decrypt("")
                except Exception:
                    unlocked = 0
                if not unlocked:
                    raise MediaCapabilityError(f"{filename} is encrypted")
            chunks: list[str] = []
            used = 0
            page_count = len(reader.pages)
            for index in range(min(page_count, MAX_PDF_PAGES)):
                page = reader.pages[index]
                page_text = page.extract_text() or ""
                if not page_text:
                    continue
                remaining = MAX_EXTRACT_CHARS_PER_FILE - used
                if remaining <= 0:
                    truncated = True
                    break
                chunks.append(page_text[:remaining])
                used += min(len(page_text), remaining)
                if len(page_text) > remaining:
                    truncated = True
                    break
            if page_count > MAX_PDF_PAGES:
                truncated = True
            text = "\n\n".join(chunks).strip()
        except MediaCapabilityError:
            raise
        except Exception as exc:
            raise MediaCapabilityError(f"failed to extract PDF {filename}: {exc}") from exc
        if not text:
            raise MediaCapabilityError(
                f"{filename} contains no extractable text (it may be a scanned PDF)"
            )
        return text, truncated

    if not (mime.startswith("text/") or mime in _TEXT_MIMES or suffix in _TEXT_EXTENSIONS):
        raise MediaCapabilityError(
            f"{filename} ({mime}) is not a supported text/PDF document"
        )
    original_size = len(data)
    if original_size > MAX_EXTRACT_TEXT_BYTES:
        data = data[:MAX_EXTRACT_TEXT_BYTES]
        truncated = True
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_EXTRACT_CHARS_PER_FILE:
        text = text[:MAX_EXTRACT_CHARS_PER_FILE]
        truncated = True
    if not text.strip():
        raise MediaCapabilityError(f"{filename} is empty or contains no readable text")
    return text, truncated


def _attachment_block(file: Any, text: str, *, truncated: bool) -> str:
    filename = html.escape(_file_name(file), quote=True)
    mime = html.escape(_file_mime(file), quote=True)
    trailer = "\n[content truncated by OpenAgent]" if truncated else ""
    return (
        f'<attachment filename="{filename}" mime="{mime}" extracted="true">\n'
        f"{text}{trailer}\n"
        "</attachment>"
    )


def _append_extracted_text(
    messages: Sequence[dict[str, Any]], blocks: Sequence[str],
) -> list[dict[str, Any]]:
    copied = [dict(message) for message in messages]
    block = "\n\n".join(item for item in blocks if item).strip()
    if not block:
        return copied
    for index in range(len(copied) - 1, -1, -1):
        message = copied[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [*content, {"type": "text", "text": block}]
        elif isinstance(content, str) and content:
            message["content"] = f"{content}\n\n{block}"
        else:
            message["content"] = block
        return copied
    copied.append({"role": "user", "content": block})
    return copied


async def prepare_media_for_entry(
    entry: Any,
    *,
    messages: Sequence[dict[str, Any]],
    files: Sequence[Any] | None = None,
    images: Sequence[Any] | None = None,
    audio: Sequence[Any] | None = None,
    videos: Sequence[Any] | None = None,
) -> PreparedMedia:
    """Validate/transform a media turn for one catalog entry."""
    supported = input_modalities_for(entry)
    missing_binary: list[str] = []
    if images and "image" not in supported:
        missing_binary.append("image")
    if audio and "audio" not in supported:
        missing_binary.append("audio")
    if videos and "video" not in supported:
        missing_binary.append("video")
    if missing_binary:
        raise MediaCapabilityError(
            f"{getattr(entry, 'runtime_id', 'model')} does not support "
            f"{', '.join(missing_binary)} input"
        )

    prepared_messages = [dict(message) for message in messages]
    prepared_files = list(files) if files else None
    extracted_count = 0
    if files and "file" not in supported:
        blocks: list[str] = []
        total_chars = 0
        for file in files:
            remaining = MAX_EXTRACT_CHARS_TOTAL - total_chars
            if remaining <= 0:
                raise MediaCapabilityError(
                    f"document extraction exceeds the per-turn limit of "
                    f"{MAX_EXTRACT_CHARS_TOTAL} characters"
                )
            text, truncated = await asyncio.to_thread(_extract_file_text_sync, file)
            if len(text) > remaining:
                text = text[:remaining]
                truncated = True
            blocks.append(_attachment_block(file, text, truncated=truncated))
            total_chars += len(text)
            extracted_count += 1
        prepared_messages = _append_extracted_text(prepared_messages, blocks)
        prepared_files = None

    native = native_modalities_for_payload(
        files=prepared_files, images=images, audio=audio, videos=videos,
    )
    if not native.issubset(supported):
        missing = sorted(native - supported)
        raise MediaCapabilityError(
            f"{getattr(entry, 'runtime_id', 'model')} does not support "
            f"{', '.join(missing)} input"
        )
    return PreparedMedia(
        messages=prepared_messages,
        files=prepared_files,
        images=list(images) if images else None,
        audio=list(audio) if audio else None,
        videos=list(videos) if videos else None,
        native_modalities=native,
        extracted_files=extracted_count,
    )


async def select_media_model(
    selected: Any,
    catalog: Sequence[Any],
    *,
    messages: Sequence[dict[str, Any]],
    files: Sequence[Any] | None = None,
    images: Sequence[Any] | None = None,
    audio: Sequence[Any] | None = None,
    videos: Sequence[Any] | None = None,
) -> SelectedMediaModel:
    """Use ``selected`` when safe, otherwise first configured compatible LLM."""
    ordered = [selected]
    ordered.extend(
        entry for entry in catalog
        if getattr(entry, "runtime_id", None) != getattr(selected, "runtime_id", None)
    )
    errors: list[str] = []
    for entry in ordered:
        try:
            prepared = await prepare_media_for_entry(
                entry,
                messages=messages,
                files=files,
                images=images,
                audio=audio,
                videos=videos,
            )
        except MediaCapabilityError as exc:
            errors.append(str(exc))
            continue
        return SelectedMediaModel(
            entry=entry,
            prepared=prepared,
            fell_back=(
                getattr(entry, "runtime_id", None)
                != getattr(selected, "runtime_id", None)
            ),
        )

    required = sorted(native_modalities_for_payload(
        files=files, images=images, audio=audio, videos=videos,
    ))
    detail = "; ".join(errors[:3])
    raise MediaCapabilityError(
        "No enabled configured model can process the attached media"
        + (f" (required: {', '.join(required)})" if required else "")
        + ". Configure models.metadata.input_modalities or enable a compatible model."
        + (f" Details: {detail}" if detail else "")
    )


def metadata_json_for_storage(
    metadata: dict[str, Any] | None, *, provider: str, model: str,
) -> str:
    """Canonical JSON helper used by DB writes and idempotent backfills."""
    return json.dumps(
        normalize_model_metadata(metadata, provider=provider, model=model),
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "INPUT_MODALITIES",
    "MAX_EXTRACT_CHARS_PER_FILE",
    "MAX_EXTRACT_CHARS_TOTAL",
    "MAX_PDF_PAGES",
    "MediaCapabilityError",
    "PreparedMedia",
    "SelectedMediaModel",
    "default_input_modalities",
    "input_modalities_for",
    "metadata_json_for_storage",
    "native_modalities_for_payload",
    "normalize_input_modalities",
    "normalize_model_metadata",
    "prepare_media_for_entry",
    "select_media_model",
]
