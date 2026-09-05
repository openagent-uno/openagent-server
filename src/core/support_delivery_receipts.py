"""Compact operational receipts without customer text or credentials."""
from __future__ import annotations
import re
from typing import Any
from src.core.support_turn import receipt_objects, delivery_state

# These are labels, never excerpts from the provider's message. Classification
# is a diagnostic hint; it does not authorize retries or change delivery state.
_HINTS = (
    ('newer_inbound', r'newer.*inbound|thread changed|re-read thread_brief'),
    ('already_answered', r'already answered|already replied|outbound.*newer'),
    ('idempotency', r'idempotenc|duplicate delivery'),
    ('validation_error', r'validation error|field required|unexpected keyword'),
    ('permission_denied', r'forbidden|unauthorized|permission denied'),
    ('rate_limit', r'rate.limit|too many requests|cooldown'),
    ('timeout', r'timeout|timed out'),
    ('tool_missing', r'tool not found|no tool named|not loaded'),
)


def summarize(actions: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [a for a in actions if a.get('kind') in ('customer_reply', 'customer_draft')]
    result: dict[str, Any] = {'state': delivery_state(actions), 'attempts': len(attempts)}
    if not attempts:
        return result
    last = attempts[-1]
    result['action_success'] = last.get('success') is True
    objects = receipt_objects(last.get('receipt'))
    for name in ('sent', 'blocked', 'uncertain', 'isError', 'ok', 'success'):
        values = [o[name] for o in objects if isinstance(o.get(name), bool)]
        if values:
            result[name] = values[-1]
    for obj in objects:
        category = obj.get('category')
        if isinstance(category, str) and re.fullmatch(r'[a-z_]{1,80}', category):
            result['category'] = category
        status = obj.get('status')
        if isinstance(status, int) and 400 <= status < 600:
            result['http_status'] = status
    if result['state'] not in ('sent', 'simulated', 'not_attempted'):
        # Inspect failure text in process, but never retain the text itself.
        texts = []
        for obj in objects:
            for key in ('error', 'reason', 'detail', 'text'):
                if isinstance(obj.get(key), str): texts.append(obj[key])
        receipt = last.get('receipt')
        if isinstance(receipt, str): texts.append(receipt)
        text = '\n'.join(texts)[:12000]
        result['error_hints'] = [name for name, pattern in _HINTS if re.search(pattern, text, re.I)]
        match = re.search(r'(?:HTTP(?:Exception|Error)?[ :]+|status(?: code)?[ :=]+)([45]\d\d)\b', text, re.I)
        if match: result['http_status'] = int(match[1])
        error_type = next((o['error_type'] for o in objects if o.get('error_type') in ('RuntimeError', 'TimeoutError', 'ValueError', 'ConnectionError', 'MCPResultEnvelopeLimitError')), None)
        if error_type: result['error_type'] = error_type
    return result
