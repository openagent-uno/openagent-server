"""Shared channel utilities — formatting, voice transcription, response parsing.

The actual platform implementations are in openagent/bridges/.
"""

from openagent.channels.base import (
    parse_response_markers,
    split_preserving_code_blocks,
    is_blocked_attachment,
)

__all__ = ["parse_response_markers", "split_preserving_code_blocks", "is_blocked_attachment"]
