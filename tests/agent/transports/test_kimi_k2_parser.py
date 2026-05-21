"""Tests for the Kimi K2 inline tool-call parser."""

import json

import pytest
from agent.transports.kimi_k2_parser import (
    try_parse,
    parse_k2_tool_calls,
    _has_k2_markers,
    _strip_section,
)


class TestHasK2Markers:
    def test_detects_section_begin(self):
        assert _has_k2_markers("<|tool_calls_section_begin|>...")

    def test_none_content(self):
        assert not _has_k2_markers(None)

    def test_empty_content(self):
        assert not _has_k2_markers("")

    def test_plain_text(self):
        assert not _has_k2_markers("Hello world")


class TestStripSection:
    def test_strip_single_section(self):
        result = _strip_section("a<|tool_calls_section_begin|>stuff<|tool_calls_section_end|>b")
        assert result == "ab"

    def test_strip_trailing_text(self):
        result = _strip_section("lead<|tool_calls_section_begin|>inner<|tool_calls_section_end|>trail")
        assert result == "leadtrail"

    def test_strip_multiple_sections(self):
        result = _strip_section(
            "a<|tool_calls_section_begin|>1<|tool_calls_section_end|>"
            "b<|tool_calls_section_begin|>2<|tool_calls_section_end|>c"
        )
        assert result == "abc"

    def test_no_section_passthrough(self):
        assert _strip_section("hello") == "hello"


class TestParseK2ToolCalls:
    def test_basic(self):
        content = (
            "I'll search.\n"
            "<|tool_calls_section_begin|>\n"
            '<|tool_call_begin|>{"name": "web_search", "arguments": {"q": "hi"}, "id": "c1"}<|tool_call_end|>\n'
            "<|tool_calls_section_end|>"
        )
        cleaned, calls = parse_k2_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].id == "c1"
        assert '"q": "hi"' in calls[0].arguments
        assert cleaned == "I'll search."

    def test_multiple_tool_calls(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "a", "arguments": {}, "id": "1"}<|tool_call_end|>'
            '<|tool_call_begin|>{"name": "b", "arguments": {"x": 1}, "id": "2"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        cleaned, calls = parse_k2_tool_calls(content)
        assert len(calls) == 2
        assert calls[0].name == "a"
        assert calls[1].name == "b"
        assert cleaned is None  # content was only tags

    def test_no_id(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "search", "arguments": {"q": "test"}}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, calls = parse_k2_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].id is None

    def test_arguments_as_dict(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "f", "arguments": {"key": "value"}, "id": "c1"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, calls = parse_k2_tool_calls(content)
        assert json.loads(calls[0].arguments) == {"key": "value"}

    def test_arguments_as_string(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "f", "arguments": "{\\"key\\": \\"value\\"}", "id": "c1"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, calls = parse_k2_tool_calls(content)
        assert json.loads(calls[0].arguments) == {"key": "value"}

    def test_empty_name_skipped(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "", "arguments": {}, "id": "c1"}<|tool_call_end|>'
            '<|tool_call_begin|>{"name": "real", "arguments": {}, "id": "c2"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, calls = parse_k2_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "real"

    def test_malformed_json_skipped(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{not-json}<|tool_call_end|>'
            '<|tool_call_begin|>{"name": "ok", "arguments": {}, "id": "c1"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, calls = parse_k2_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "ok"

    def test_not_a_dict_skipped(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>"just a string"<|tool_call_end|>'
            '<|tool_call_begin|>{"name": "ok", "arguments": {}, "id": "c1"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, calls = parse_k2_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "ok"

    def test_extra_fields_preserved(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "f", "type": "function", "arguments": {"x": 1}, "id": "c1", "extra": "val"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, calls = parse_k2_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "f"
        assert calls[0].id == "c1"


class TestTryParse:
    def test_no_markers_returns_original(self):
        content, calls = try_parse("Hello world")
        assert content == "Hello world"
        assert calls is None

    def test_none_passthrough(self):
        content, calls = try_parse(None)
        assert content is None
        assert calls is None

    def test_empty_passthrough(self):
        content, calls = try_parse("")
        assert content == ""
        assert calls is None

    def test_with_markers_parses(self):
        content = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "search", "arguments": {"q": "x"}, "id": "c1"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        cleaned, calls = try_parse(content)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].name == "search"

    def test_content_before_and_after(self):
        content = (
            "Before.\n"
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>{"name": "a", "arguments": {}, "id": "1"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
            "\nAfter."
        )
        cleaned, calls = try_parse(content)
        assert cleaned == "Before.\n\nAfter."
        assert len(calls) == 1



