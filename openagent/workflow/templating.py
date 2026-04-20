"""Template resolution for block config values.

Any string anywhere in a block's ``config`` may contain ``{{expr}}``
placeholders. At runtime they are resolved against:

    ctx = {
        "inputs": <workflow-level inputs>,
        "vars":   <graph_json.variables, mutable by set-variable>,
        "nodes":  {node_id: {output, status, error?}},
        "now":    <ISO-8601 UTC of the current tick>,
        "run_id": <workflow_runs.id>,
    }

Evaluation uses jinja2's ``SandboxedEnvironment`` — no subprocess
spawning, no attribute access to Python internals, no arbitrary
imports. Only the whitelisted filters below are registered.

``resolve_templates`` walks arbitrarily nested ``dict``/``list``
structures and resolves every string it encounters. Non-string values
pass through untouched. A single placeholder that spans the entire
string (``"{{expr}}"`` with nothing else around it) preserves the
resolved value's native type — this lets the AI write
``"args": {"count": "{{n2.output.count}}"}`` and still send an int
into the tool.
"""

from __future__ import annotations

import json
import re
from typing import Any

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment


# Exactly ``{{<expr>}}`` with no surrounding text — when this matches we
# return the expression's native result instead of coercing to str.
_WHOLE_EXPR = re.compile(r"^\s*\{\{\s*(?P<expr>.+?)\s*\}\}\s*$", re.DOTALL)


class _OpenAgentSandbox(SandboxedEnvironment):
    """Sandboxed environment that prefers item access on dicts.

    Stock Jinja tries ``getattr(obj, name)`` first and only falls back
    to ``obj[name]`` on AttributeError. For dicts, that means
    ``inputs.items`` returns the ``dict.items`` method instead of the
    ``"items"`` key. Our workflows lean heavily on ``inputs.x``,
    ``vars.y``, ``nodes.nX.output.z`` — always dict access — so we
    flip the priority: dicts get item access first, everything else
    stays default.
    """

    def getattr(self, obj: Any, attribute: str) -> Any:  # noqa: N802
        if isinstance(obj, dict) and attribute in obj:
            return obj[attribute]
        return super().getattr(obj, attribute)


def _build_env() -> _OpenAgentSandbox:
    env = _OpenAgentSandbox(
        autoescape=False,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["json"] = lambda v: json.dumps(v, default=str)
    env.filters["truncate_str"] = lambda v, n=80: (str(v)[: int(n)] + "…") if len(str(v)) > int(n) else str(v)
    env.filters["default_if_none"] = lambda v, d="": d if v is None else v
    return env


_ENV = _build_env()


def _eval_whole(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate a jinja expression and return its native value.

    ``Template.render`` always stringifies, which is wrong for
    `{{n.output.count}}` when we want the int. Use ``compile_expression``
    which returns the raw Python value.
    """
    fn = _ENV.compile_expression(expr)
    return fn(**ctx)


def resolve_templates(value: Any, ctx: dict[str, Any]) -> Any:
    """Recursively resolve any ``{{...}}`` placeholders inside ``value``.

    - ``dict`` / ``list`` / ``tuple`` are walked; strings inside them are
      resolved.
    - A string consisting of exactly one ``{{expr}}`` returns the
      expression's native type (so ints stay ints).
    - A string with mixed literal + placeholder content is resolved as
      a Jinja template (result is always a string).
    - Non-string leaves pass through.
    """
    if isinstance(value, str):
        whole = _WHOLE_EXPR.match(value)
        if whole is not None:
            try:
                return _eval_whole(whole.group("expr"), ctx)
            except Exception as exc:  # pragma: no cover - propagate with context
                raise TemplateError(
                    f"failed to evaluate template {value!r}: {exc}"
                ) from exc
        if "{{" not in value:
            return value
        try:
            return _ENV.from_string(value).render(**ctx)
        except Exception as exc:  # pragma: no cover
            raise TemplateError(
                f"failed to render template {value!r}: {exc}"
            ) from exc
    if isinstance(value, dict):
        return {k: resolve_templates(v, ctx) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        resolved = [resolve_templates(item, ctx) for item in value]
        return type(value)(resolved) if isinstance(value, tuple) else resolved
    return value


def evaluate_expression(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate a bare jinja expression (no surrounding ``{{ }}``) and
    return the native Python result. Used by the ``if`` block for
    truthiness checks and by ``loop.items_expr`` for list resolution.
    """
    # Allow the caller to pass either "ctx.var" or "{{ctx.var}}".
    stripped = _WHOLE_EXPR.match(expr)
    if stripped is not None:
        expr = stripped.group("expr")
    try:
        return _eval_whole(expr, ctx)
    except Exception as exc:
        raise TemplateError(
            f"failed to evaluate expression {expr!r}: {exc}"
        ) from exc


class TemplateError(ValueError):
    """Raised when a template / expression fails to resolve."""
