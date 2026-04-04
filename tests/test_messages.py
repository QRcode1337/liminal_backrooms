# tests/test_messages.py
"""Tests for the messages module."""

import unittest
from messages import (
    MessageType,
    make_system_notification,
    make_agent_notification,
    make_assistant_message,
    make_user_message,
    make_whisper,
    is_visible,
    is_from_ai,
    get_type,
)


class TestMessageFactories(unittest.TestCase):

    def test_system_notification(self):
        msg = make_system_notification("Test notification")
        self.assertEqual(msg["role"], "system")
        self.assertEqual(msg["content"], "Test notification")
        self.assertEqual(msg["_type"], MessageType.SYSTEM_NOTIFICATION)

    def test_agent_notification_success(self):
        msg = make_agent_notification("Command ran", success=True)
        self.assertTrue(msg["_command_success"])

    def test_agent_notification_failure(self):
        msg = make_agent_notification("Command failed", success=False)
        self.assertFalse(msg["_command_success"])

    def test_agent_notification_pending(self):
        msg = make_agent_notification("Generating...", success=None)
        self.assertIsNone(msg["_command_success"])

    def test_assistant_message(self):
        msg = make_assistant_message("Hello", "AI-1", "claude-opus")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["ai_name"], "AI-1")
        self.assertEqual(msg["model"], "claude-opus")

    def test_user_message(self):
        msg = make_user_message("Hi there", "Bob")
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["_user_name"], "Bob")

    def test_whisper(self):
        msg = make_whisper("Secret", "AI-1", "AI-2")
        self.assertEqual(msg["_type"], MessageType.WHISPER)
        self.assertEqual(msg["_whisper_from"], "AI-1")
        self.assertEqual(msg["_whisper_to"], "AI-2")

    def test_extra_kwargs(self):
        msg = make_assistant_message("Hi", "AI-1", "model", custom_field="value")
        self.assertEqual(msg["custom_field"], "value")


class TestMessageHelpers(unittest.TestCase):

    def test_is_visible_normal(self):
        self.assertTrue(is_visible({"role": "assistant", "content": "hi"}))

    def test_is_visible_hidden(self):
        self.assertFalse(is_visible({"role": "assistant", "content": "hi", "hidden": True}))

    def test_is_visible_non_dict(self):
        self.assertFalse(is_visible("not a dict"))

    def test_is_from_ai(self):
        msg = {"ai_name": "AI-1", "content": "hello"}
        self.assertTrue(is_from_ai(msg, "AI-1"))
        self.assertFalse(is_from_ai(msg, "AI-2"))

    def test_get_type(self):
        self.assertEqual(get_type({"_type": "whisper"}), "whisper")
        self.assertIsNone(get_type({"content": "no type"}))


if __name__ == "__main__":
    unittest.main()
