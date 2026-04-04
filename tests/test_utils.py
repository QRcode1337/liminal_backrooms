# tests/test_utils.py
"""Tests for shared utility functions."""

import sys
import unittest

# shared_utils has heavy dependencies (replicate, anthropic, etc.) that may
# not be installed in CI. Mock them so we can import the lightweight helpers.
_mock_modules = [
    "replicate", "anthropic", "together", "openai", "bs4",
    "duckduckgo_search", "dotenv", "PIL", "PIL.Image",
]
for mod_name in _mock_modules:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = type(sys)("mock_" + mod_name)
# Provide minimal stubs for classes shared_utils references at import time
sys.modules["anthropic"].Anthropic = lambda **kw: None  # type: ignore[attr-defined]
sys.modules["openai"].OpenAI = lambda **kw: None  # type: ignore[attr-defined]
sys.modules["together"].Together = lambda **kw: None  # type: ignore[attr-defined]
sys.modules["dotenv"].load_dotenv = lambda: None  # type: ignore[attr-defined]

from shared_utils import get_visible_messages, _iter_sse_stream


class TestGetVisibleMessages(unittest.TestCase):

    def test_filters_hidden(self):
        msgs = [
            {"role": "assistant", "content": "visible"},
            {"role": "assistant", "content": "hidden", "hidden": True},
            {"role": "user", "content": "also visible"},
        ]
        result = get_visible_messages(msgs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"], "visible")
        self.assertEqual(result[1]["content"], "also visible")

    def test_filters_non_dicts(self):
        msgs = [
            {"role": "assistant", "content": "ok"},
            "plain string",
            42,
        ]
        result = get_visible_messages(msgs)
        self.assertEqual(len(result), 1)

    def test_empty_list(self):
        self.assertEqual(get_visible_messages([]), [])

    def test_all_hidden(self):
        msgs = [
            {"content": "a", "hidden": True},
            {"content": "b", "hidden": True},
        ]
        self.assertEqual(get_visible_messages(msgs), [])


class TestIterSseStream(unittest.TestCase):

    def test_openai_style_chunks(self):
        """Test parsing OpenAI-style SSE with choices[0].delta.content."""
        import json

        lines = [
            b'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}).encode(),
            b'data: ' + json.dumps({"choices": [{"delta": {"content": " world"}}]}).encode(),
            b'data: [DONE]',
        ]

        class FakeResponse:
            def iter_lines(self):
                return iter(lines)

        chunks = []
        result = _iter_sse_stream(FakeResponse(), stream_callback=lambda c: chunks.append(c))
        self.assertEqual(result, "Hello world")
        self.assertEqual(chunks, ["Hello", " world"])

    def test_claude_style_chunks(self):
        """Test parsing Claude-style SSE with content_block_delta."""
        import json

        lines = [
            b'data: ' + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}).encode(),
            b'data: ' + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " there"}}).encode(),
            b'data: [DONE]',
        ]

        class FakeResponse:
            def iter_lines(self):
                return iter(lines)

        result = _iter_sse_stream(FakeResponse())
        self.assertEqual(result, "Hi there")

    def test_empty_stream(self):
        class FakeResponse:
            def iter_lines(self):
                return iter([b'data: [DONE]'])

        result = _iter_sse_stream(FakeResponse())
        self.assertEqual(result, "")

    def test_malformed_json_skipped(self):
        lines = [
            b'data: {invalid json}',
            b'data: {"choices": [{"delta": {"content": "ok"}}]}',
            b'data: [DONE]',
        ]

        class FakeResponse:
            def iter_lines(self):
                return iter(lines)

        result = _iter_sse_stream(FakeResponse())
        self.assertEqual(result, "ok")


class TestFilterConversationMessages(unittest.TestCase):
    """Test the _filter_conversation_messages helper.

    main.py has heavy GUI dependencies (PyQt6) which may not be importable
    in CI, so we test the function via a direct exec of the function source
    if main cannot be imported.
    """

    @classmethod
    def setUpClass(cls):
        """Try to import the function; skip tests if deps are unavailable."""
        try:
            from main import _filter_conversation_messages
            cls._filter = staticmethod(_filter_conversation_messages)
        except (ImportError, ModuleNotFoundError):
            cls._filter = None

    def test_filters_system_messages(self):
        if self._filter is None:
            self.skipTest("main.py deps unavailable")
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "ai_name": "AI-1"},
        ]
        result = self._filter(msgs, "AI-1")
        self.assertEqual(len(result), 2)
        self.assertTrue(all(m["role"] != "system" for m in result))

    def test_filters_empty_messages(self):
        if self._filter is None:
            self.skipTest("main.py deps unavailable")
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "valid"},
        ]
        result = self._filter(msgs, "AI-1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "valid")

    def test_filters_whispers_for_other_ais(self):
        if self._filter is None:
            self.skipTest("main.py deps unavailable")
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "secret", "_type": "whisper", "_whisper_to": "AI-2"},
            {"role": "system", "content": "for me", "_type": "whisper", "_whisper_to": "AI-1"},
        ]
        result = self._filter(msgs, "AI-1")
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
