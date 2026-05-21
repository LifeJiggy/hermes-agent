"""Parser for Kimi K2 inline tool-call format.

Kimi K2 (via OpenRouter and Moonshot) sometimes returns tool calls
embedded inside the content string as tagged JSON instead of in the
structured ``tool_calls`` field.  This module detects and normalises
that format so the agent loop receives proper ``ToolCall`` objects.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.transports.types import ToolCall

# Kimi K2 inline markers
_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_SECTION_END = "<|tool_calls_section_end|>"
_CALL_BEGIN = "<|tool_call_begin|>"
_CALL_END = "<|tool_call_end|>"

# Matches a single tool call: <|tool_call_begin|>{...json...}<|tool_call_end|>
_CALL_RE = re.compile(
    re.escape(_CALL_BEGIN) + r"(\{.*?\})" + re.escape(_CALL_END),
    re.DOTALL,
)


def _has_k2_markers(content: str | None) -> bool:
    """Return ``True`` if *content* has Kimi K2 tool-call markers."""
    return bool(content) and _SECTION_BEGIN in content


def _strip_section(content: str) -> str:
    """Remove all ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>`` blocks."""
    result = content
    while True:
        si = result.find(_SECTION_BEGIN)
        if si == -1:
            break
        ei = result.find(_SECTION_END, si)
        if ei == -1:
            break
        result = result[:si] + result[ei + len(_SECTION_END):]
    return result


def _json_arg(value: Any) -> str:
    """Normalise *value* to a JSON string for ``ToolCall.arguments``."""
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, str):
        return value
    return str(value)


def _build_call(data: dict[str, Any]) -> ToolCall | None:
    """Build a ``ToolCall`` from a parsed K2 JSON dict, or ``None`` if invalid."""
    name = data.get("name", "")
    if not name:
        return None
    arguments = _json_arg(data.get("arguments", {}))
    call_id = data.get("id")
    return ToolCall(
        id=str(call_id) if call_id else None,
        name=name,
        arguments=arguments,
    )


def parse_k2_tool_calls(content: str) -> tuple[str | None, list[ToolCall]]:
    """Extract ``ToolCall`` objects embedded in a Kimi K2 content string.

    Returns ``(cleaned_content, tool_calls)`` where *cleaned_content* is
    *content* with all tool-call sections removed (or ``None`` if nothing
    remains after stripping).
    """
    tool_calls: list[ToolCall] = []

    for match in _CALL_RE.finditer(content):
        raw = match.group(1)
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        tc = _build_call(data)
        if tc is not None:
            tool_calls.append(tc)

    cleaned = _strip_section(content).strip() or None
    return cleaned, tool_calls


def try_parse(content: str | None) -> tuple[str | None, list[ToolCall] | None]:
    """Attempt to parse Kimi K2 tool calls from *content*.

    Returns ``(cleaned_content, tool_calls)`` when K2 markers are found,
    or ``(content, None)`` when they are not (so callers can use the
    return value unconditionally).
    """
    if not _has_k2_markers(content):
        return content, None
    return parse_k2_tool_calls(content)  # type: ignore[arg-type]
