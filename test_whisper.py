#!/usr/bin/env python3
"""Tests for !whisper parsing and participant targeting."""

import unittest

from command_parser import parse_commands, resolve_participant_target

ROSTER = [
    ("AI-1", "agy/claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("AI-2", "cx/gpt-5.6-luna-medium", "GPT 5.6 Luna"),
    ("AI-3", "xai/grok-4.3", "Grok 4.3"),
    ("AI-4", "xai/grok-4.20-0309-non-reasoning", "Grok 4.20"),
    ("AI-5", "agy/gemini-3.5-flash-high", "Gemini 3.5 Flash"),
]


class ParseWhisperTests(unittest.TestCase):
    def test_quoted_model_id(self):
        _, cmds = parse_commands(
            '!whisper "xai/grok-4.20-0309-non-reasoning" "blind spot is the will"'
        )
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "whisper")
        self.assertEqual(cmds[0].params["target"], "xai/grok-4.20-0309-non-reasoning")
        self.assertEqual(cmds[0].params["message"], "blind spot is the will")

    def test_quoted_display_name(self):
        _, cmds = parse_commands('!whisper "grok 4.20" "psst"')
        self.assertEqual(cmds[0].params["target"], "grok 4.20")
        self.assertEqual(cmds[0].params["message"], "psst")

    def test_unquoted_slot(self):
        _, cmds = parse_commands('!whisper AI-4 "hello"')
        self.assertEqual(cmds[0].params["target"], "AI-4")
        self.assertEqual(cmds[0].params["message"], "hello")

    def test_strips_command_from_visible_text(self):
        cleaned, cmds = parse_commands('Public line\n!whisper "AI-4" "secret"\nMore public')
        self.assertEqual(len(cmds), 1)
        self.assertNotIn("secret", cleaned)
        self.assertNotIn("!whisper", cleaned)


class ResolveTargetTests(unittest.TestCase):
    def test_slot_and_number(self):
        self.assertEqual(resolve_participant_target("AI-4", ROSTER)[0], "AI-4")
        self.assertEqual(resolve_participant_target("4", ROSTER)[0], "AI-4")

    def test_model_id(self):
        name, err = resolve_participant_target(
            "xai/grok-4.20-0309-non-reasoning", ROSTER
        )
        self.assertIsNone(err)
        self.assertEqual(name, "AI-4")

    def test_display_name_and_spaced_version(self):
        self.assertEqual(resolve_participant_target("Grok 4.20", ROSTER)[0], "AI-4")
        self.assertEqual(resolve_participant_target("grok 4.20", ROSTER)[0], "AI-4")
        self.assertEqual(resolve_participant_target("grok-4.20", ROSTER)[0], "AI-4")

    def test_ambiguous_grok(self):
        name, err = resolve_participant_target("grok", ROSTER)
        self.assertIsNone(name)
        self.assertIn("ambiguous", err.lower())
        self.assertIn("AI-3", err)
        self.assertIn("AI-4", err)

    def test_unknown(self):
        name, err = resolve_participant_target("opus", ROSTER)
        self.assertIsNone(name)
        self.assertIn("invalid target", err)

    def test_missing_slot(self):
        name, err = resolve_participant_target("AI-9", ROSTER)
        self.assertIsNone(name)
        self.assertIn("doesn't exist", err)


if __name__ == "__main__":
    unittest.main()
