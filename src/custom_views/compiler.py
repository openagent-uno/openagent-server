"""Parser, validator, and compiler for the non-executable OA-UI v1 DSL."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping
from urllib.parse import urlparse
from xml.etree import ElementTree


SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 256 * 1024
MAX_SPEC_BYTES = 512 * 1024
MAX_NODES = 512
MAX_DEPTH = 32
MAX_CHILDREN = 100
MAX_PROPS = 64
MAX_STRING_CHARS = 16_384

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_BINDING_RE = re.compile(r"^(?P<source>[A-Za-z][A-Za-z0-9_.-]{0,63}):(?P<path>/.*|)$")
_MUSTACHE_BINDING_RE = re.compile(
    r"^\{\{\s*(?P<root>data|state)(?P<tail>(?:\.[A-Za-z0-9_-]+)+)\s*\}\}$"
)

CONTAINER_COMPONENTS = frozenset({
    "stack", "row", "grid", "card", "scroll", "tabs", "sub-view",
})
LEAF_COMPONENTS = frozenset({
    "divider", "spacer", "heading", "text", "markdown", "code", "badge",
    "metric", "status", "progress", "image", "icon", "file-link", "button",
    "toggle", "select", "segmented", "text-input", "table", "list", "key-value",
    "line-chart", "bar-chart", "area-chart", "pie-chart", "donut-chart",
    "scatter-chart", "gauge", "sparkline",
    "loading-state", "empty-state", "stale-state", "error-state",
})
COMPONENTS = CONTAINER_COMPONENTS | LEAF_COMPONENTS

_COMMON_PROPS = frozenset({
    "label", "title", "subtitle", "description", "text", "value", "fallback",
    "icon", "tone", "size", "align", "justify", "gap", "padding", "width",
    "height", "minWidth", "maxWidth", "minHeight", "maxHeight", "grow", "wrap",
    "hidden", "disabled", "loading", "emptyText", "ariaLabel", "tooltip",
    "format", "unit", "precision", "truncate", "copyable", "selectable",
    "action", "href", "src", "alt", "assetId", "fileId", "target", "variant",
})
_COMPONENT_PROPS: dict[str, frozenset[str]] = {
    "stack": frozenset({"direction"}),
    "row": frozenset({"direction"}),
    "grid": frozenset({"columns", "minColumnWidth", "rowGap", "columnGap"}),
    "card": frozenset({"footer", "collapsible", "collapsed", "tight", "rail"}),
    "scroll": frozenset({"axis"}),
    "tabs": frozenset({"active", "items"}),
    "sub-view": frozenset({"name", "active", "viewId", "revision"}),
    "heading": frozenset({"level"}),
    "text": frozenset({"muted"}),
    "markdown": frozenset({"allowLinks"}),
    "code": frozenset({"code", "language", "lineNumbers"}),
    "metric": frozenset({"detail", "trend"}),
    "status": frozenset({"status"}),
    "progress": frozenset({"min", "max", "indeterminate"}),
    "image": frozenset({"fit", "aspectRatio", "caption"}),
    "icon": frozenset({"name"}),
    "file-link": frozenset({"filename", "kind"}),
    "button": frozenset({"kind", "input"}),
    "toggle": frozenset({"checked", "name", "input"}),
    "select": frozenset({"options", "selected", "multiple", "name", "input"}),
    "segmented": frozenset({"options", "selected", "name", "input"}),
    "text-input": frozenset({
        "name", "placeholder", "multiline", "inputType", "submitLabel", "maxLength", "input",
    }),
    "table": frozenset({"rows", "columns", "dense", "sortable", "pageSize"}),
    "list": frozenset({"items", "ordered"}),
    "key-value": frozenset({"items"}),
    "line-chart": frozenset({"data", "values", "series", "xKey", "yKey", "legend", "stacked"}),
    # A stacked bar is represented as `bar-chart` with `stacked=true`; it is
    # not a second renderer component or a free-form chart subtype.
    "bar-chart": frozenset({"data", "values", "series", "xKey", "yKey", "legend", "stacked", "orientation"}),
    "area-chart": frozenset({"data", "values", "series", "xKey", "yKey", "legend", "stacked"}),
    "pie-chart": frozenset({"data", "values", "series", "nameKey", "valueKey", "legend"}),
    "donut-chart": frozenset({"data", "values", "series", "nameKey", "valueKey", "legend", "centerLabel"}),
    "scatter-chart": frozenset({"data", "values", "series", "xKey", "yKey", "legend"}),
    "gauge": frozenset({"data", "values", "series", "min", "max", "thresholds"}),
    "sparkline": frozenset({"data", "values", "series", "xKey", "yKey"}),
}


class OAUIValidationError(ValueError):
    """Safe validation error; it never includes source or data values."""


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise OAUIValidationError("OA-UI values must be finite JSON") from exc


def _validate_json(value: Any, *, depth: int = 0) -> None:
    if depth > 16:
        raise OAUIValidationError("OA-UI property nesting exceeds 16 levels")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise OAUIValidationError("OA-UI numbers must be finite")
        return
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            raise OAUIValidationError("OA-UI strings exceed the supported size")
        return
    if isinstance(value, list):
        if len(value) > 1000:
            raise OAUIValidationError("OA-UI arrays exceed 1000 items")
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 256:
            raise OAUIValidationError("OA-UI objects exceed 256 fields")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128 or key in {"__proto__", "prototype", "constructor"}:
                raise OAUIValidationError("OA-UI object contains an invalid key")
            _validate_json(item, depth=depth + 1)
        return
    raise OAUIValidationError("OA-UI values must be JSON-compatible")


def _validate_href(value: str) -> None:
    if any(ord(char) < 0x20 for char in value) or "\\" in value:
        raise OAUIValidationError("OA-UI link is invalid")
    parsed = urlparse(value)
    if value.lstrip().lower().startswith(("javascript:", "data:", "file:")):
        raise OAUIValidationError("OA-UI link scheme is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise OAUIValidationError("OA-UI links cannot contain credentials")
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    if not parsed.scheme and not parsed.netloc and (
        (value.startswith("/") and not value.startswith("//")) or value.startswith("#")
    ):
        return
    if parsed.scheme in {"http", "https"}:
        raise OAUIValidationError("OA-UI remote URLs must use https")
    raise OAUIValidationError("OA-UI links must use https, loopback http, or an app path")


def _validate_asset_ref(value: str) -> None:
    if not isinstance(value, str):
        raise OAUIValidationError("OA-UI media src must be an artifact or bundle asset reference")
    artifact = re.fullmatch(
        r"artifact:(?://)?[A-Za-z0-9][A-Za-z0-9_-]{0,255}", value,
    )
    if artifact is not None:
        return
    asset = re.fullmatch(
        r"asset:(?://)?(?P<path>[A-Za-z0-9][A-Za-z0-9_./-]{0,1023})", value,
    )
    if asset is not None and all(
        part not in {"", ".", ".."} for part in asset.group("path").split("/")
    ):
        return
    raise OAUIValidationError("OA-UI media src must be an artifact or bundle asset reference")


def _binding(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        stripped = value.strip()
        mustache = _MUSTACHE_BINDING_RE.fullmatch(stripped)
        if mustache is not None:
            pointer = "/" + "/".join(mustache.group("tail").lstrip(".").split("."))
            return {"$bind": {"source": mustache.group("root"), "path": pointer}}
        match = _BINDING_RE.fullmatch(stripped)
        if match is None:
            raise OAUIValidationError("binding must be SOURCE:/json/pointer")
        return {"$bind": {"source": match.group("source"), "path": match.group("path")}}
    if not isinstance(value, Mapping):
        raise OAUIValidationError("binding must be an object")
    raw = value.get("$bind", value)
    if not isinstance(raw, Mapping):
        raise OAUIValidationError("binding must contain $bind")
    source = raw.get("source")
    path = raw.get("path", "")
    if not isinstance(source, str) or not _NAME_RE.fullmatch(source):
        raise OAUIValidationError("binding source is invalid")
    if not isinstance(path, str) or (path and not path.startswith("/")) or len(path) > 2048:
        raise OAUIValidationError("binding path must be a JSON Pointer")
    result: dict[str, Any] = {"$bind": {"source": source, "path": path}}
    fallback = value.get("fallback") if "$bind" in value else None
    if "fallback" in value:
        _validate_json(fallback)
        result["fallback"] = fallback
    return result


def _validate_prop(component: str, name: str, value: Any) -> Any:
    allowed = _COMMON_PROPS | _COMPONENT_PROPS.get(component, frozenset())
    if name not in allowed:
        raise OAUIValidationError(f"property {name!r} is not supported by {component}")
    if name in {"action", "target"} and (
        (isinstance(value, Mapping) and "$bind" in value)
        or (isinstance(value, str) and _MUSTACHE_BINDING_RE.fullmatch(value.strip()))
    ):
        raise OAUIValidationError(f"{name} cannot be data-bound")
    if isinstance(value, Mapping) and "$bind" in value:
        return _binding(value)
    if isinstance(value, str) and _MUSTACHE_BINDING_RE.fullmatch(value.strip()):
        return _binding(value)
    _validate_json(value)
    if name == "href" and isinstance(value, str):
        if component == "file-link":
            _validate_asset_ref(value)
        else:
            _validate_href(value)
    if name in {"src", "assetId", "fileId"} and isinstance(value, str):
        _validate_asset_ref(value)
    if name == "action" and (not isinstance(value, str) or not _NAME_RE.fullmatch(value)):
        raise OAUIValidationError("action references must use a safe identifier")
    if name == "target" and value not in {"_self", "_blank"}:
        raise OAUIValidationError("link target is not supported")
    if component == "text-input" and name == "inputType" and value not in {
        "text", "number", "email", "url", "search",
    }:
        raise OAUIValidationError("text-input inputType is not supported")
    if component == "text-input" and name == "multiline" and not isinstance(value, bool):
        raise OAUIValidationError("text-input multiline must be boolean")
    if component == "text-input" and name == "maxLength" and (
        not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1_000_000
    ):
        raise OAUIValidationError("text-input maxLength must be a positive integer")
    if component == "sub-view" and name == "revision" and (
        not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
        raise OAUIValidationError("sub-view revision must be a positive integer")
    if component == "sub-view" and name == "viewId" and (
        not isinstance(value, str) or not _NAME_RE.fullmatch(value)
    ):
        raise OAUIValidationError("sub-view viewId is invalid")
    return value


def validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of an OA-UI JSON AST."""

    if not isinstance(spec, Mapping):
        raise OAUIValidationError("OA-UI spec must be an object")
    unknown = set(spec) - {"schemaVersion", "root", "states"}
    if unknown:
        raise OAUIValidationError("OA-UI spec contains unsupported top-level fields")
    if spec.get("schemaVersion", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise OAUIValidationError("unsupported OA-UI schemaVersion")
    root = spec.get("root")
    if not isinstance(root, Mapping):
        raise OAUIValidationError("OA-UI spec requires a root node")
    count = 0
    seen_ids: set[str] = set()

    def visit(raw: Mapping[str, Any], depth: int) -> dict[str, Any]:
        nonlocal count
        count += 1
        if count > MAX_NODES:
            raise OAUIValidationError(f"OA-UI exceeds {MAX_NODES} nodes")
        if depth > MAX_DEPTH:
            raise OAUIValidationError(f"OA-UI exceeds depth {MAX_DEPTH}")
        if set(raw) - {"type", "id", "props", "children"}:
            raise OAUIValidationError("OA-UI node contains unsupported fields")
        component = raw.get("type")
        if not isinstance(component, str) or component not in COMPONENTS:
            raise OAUIValidationError("OA-UI node type is not supported")
        output: dict[str, Any] = {"type": component}
        node_id = raw.get("id")
        if node_id is not None:
            if not isinstance(node_id, str) or not _NAME_RE.fullmatch(node_id):
                raise OAUIValidationError("OA-UI node id is invalid")
            if node_id in seen_ids:
                raise OAUIValidationError("OA-UI node ids must be unique")
            seen_ids.add(node_id)
            output["id"] = node_id
        props = raw.get("props", {})
        if not isinstance(props, Mapping) or len(props) > MAX_PROPS:
            raise OAUIValidationError(f"OA-UI node props exceed {MAX_PROPS} fields")
        if props:
            output["props"] = {
                str(key): _validate_prop(component, str(key), value)
                for key, value in props.items()
            }
        children = raw.get("children", [])
        if not isinstance(children, list) or len(children) > MAX_CHILDREN:
            raise OAUIValidationError(f"OA-UI node children exceed {MAX_CHILDREN}")
        if children and component not in CONTAINER_COMPONENTS:
            raise OAUIValidationError(f"{component} cannot contain child components")
        if component == "sub-view":
            referenced = isinstance(props.get("viewId"), str)
            if referenced and "revision" not in props:
                raise OAUIValidationError(
                    "referenced sub-view requires an immutable revision"
                )
            if not referenced and "revision" in props:
                raise OAUIValidationError(
                    "sub-view revision requires viewId"
                )
            if referenced and children:
                raise OAUIValidationError(
                    "sub-view must use either viewId or child components"
                )
            if not referenced and not children:
                raise OAUIValidationError(
                    "sub-view requires viewId or child components"
                )
        if children:
            if any(not isinstance(child, Mapping) for child in children):
                raise OAUIValidationError("OA-UI children must be nodes")
            output["children"] = [visit(child, depth + 1) for child in children]
        return output

    normalized = {"schemaVersion": SCHEMA_VERSION, "root": visit(root, 1)}
    states = spec.get("states")
    if states is not None:
        if not isinstance(states, Mapping) or set(states) - {"loading", "empty", "stale", "error"}:
            raise OAUIValidationError("OA-UI states may contain loading, empty, stale, and error")
        if any(not isinstance(node, Mapping) for node in states.values()):
            raise OAUIValidationError("OA-UI state entries must be nodes")
        normalized["states"] = {
            str(name): visit(node, 1) for name, node in states.items()
        }
    if _json_size(normalized) > MAX_SPEC_BYTES:
        raise OAUIValidationError("compiled OA-UI exceeds the supported size")
    return normalized


def _parse_literal(value: str) -> Any:
    raw = value.strip()
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    if raw.startswith(("{", "[", '"')):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", raw):
        try:
            return int(raw)
        except ValueError:
            pass
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]+", raw):
        try:
            return float(raw)
        except ValueError:
            pass
    return value


def parse_markup(markup: str) -> dict[str, Any]:
    if not isinstance(markup, str) or not markup.strip():
        raise OAUIValidationError("OA-UI markup must not be empty")
    if len(markup.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise OAUIValidationError("OA-UI markup exceeds the supported size")
    lowered = markup.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise OAUIValidationError("OA-UI declarations and entities are not allowed")
    try:
        element = ElementTree.fromstring(markup)
    except ElementTree.ParseError as exc:
        raise OAUIValidationError("OA-UI markup is not well formed") from exc

    def convert(node: ElementTree.Element) -> dict[str, Any]:
        if not isinstance(node.tag, str) or node.tag not in COMPONENTS:
            raise OAUIValidationError("OA-UI markup contains an unsupported component")
        output: dict[str, Any] = {"type": node.tag}
        props: dict[str, Any] = {}
        for raw_name, raw_value in node.attrib.items():
            if raw_name == "id":
                output["id"] = raw_value
                continue
            if raw_name.endswith(".bind"):
                props[raw_name[:-5]] = _binding(raw_value)
            else:
                props[raw_name] = _parse_literal(raw_value)
        text = (node.text or "").strip()
        if text:
            if node.tag not in {"heading", "text", "markdown", "code", "badge", "button"}:
                raise OAUIValidationError("text content is only allowed in text components")
            target = "label" if node.tag == "button" else "text"
            if target in props:
                raise OAUIValidationError("text cannot be provided twice")
            props[target] = text
        if props:
            output["props"] = props
        children = list(node)
        if children:
            output["children"] = [convert(child) for child in children]
        for child in children:
            if (child.tail or "").strip():
                raise OAUIValidationError("mixed OA-UI text and child content is not supported")
        return output

    return validate_spec({"schemaVersion": SCHEMA_VERSION, "root": convert(element)})


def compile_oaui(*, markup: str | None = None, spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compile markup or validate an AST, requiring exactly one source form."""

    if (markup is None) == (spec is None):
        raise OAUIValidationError("provide exactly one of markup or spec")
    return parse_markup(markup) if markup is not None else validate_spec(spec or {})
