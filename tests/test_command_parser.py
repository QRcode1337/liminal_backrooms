# tests/test_command_parser.py
"""Tests for the command parser module."""

import unittest
from command_parser import parse_commands, format_command_result


class TestParseCommands(unittest.TestCase):
    """Test parsing of AI commands from response text."""

    def test_image_double_quotes(self):
        text = 'Let me show you !image "a sunset over mountains"'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "image")
        self.assertEqual(cmds[0].params["prompt"], "a sunset over mountains")

    def test_image_single_quotes(self):
        text = "Here's an image !image 'neon city at night'"
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].params["prompt"], "neon city at night")

    def test_search_command(self):
        text = '!search "latest AI research"'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "search")
        self.assertEqual(cmds[0].params["query"], "latest AI research")

    def test_add_ai_with_persona(self):
        text = '!add_ai "Claude Opus 4.5" "A philosophical thinker"'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "add_ai")
        self.assertEqual(cmds[0].params["model"], "Claude Opus 4.5")
        self.assertEqual(cmds[0].params["persona"], "A philosophical thinker")

    def test_add_ai_without_persona(self):
        text = '!add_ai "GPT 5"'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].params["model"], "GPT 5")
        self.assertIsNone(cmds[0].params["persona"])

    def test_temperature_command(self):
        text = "I'll be more creative !temperature 1.5"
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "temperature")
        self.assertEqual(cmds[0].params["value"], "1.5")

    def test_mute_self(self):
        text = "I'll listen for a bit !mute_self"
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "mute_self")

    def test_vote_with_options(self):
        text = '!vote "What should we discuss?" [AI, philosophy, art]'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "vote")
        self.assertEqual(cmds[0].params["question"], "What should we discuss?")
        self.assertIn("AI", cmds[0].params["options"])

    def test_whisper_command(self):
        text = '!whisper "AI-2" "secret plan"'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "whisper")
        self.assertEqual(cmds[0].params["target"], "AI-2")
        self.assertEqual(cmds[0].params["message"], "secret plan")

    def test_prompt_stripped_from_cleaned(self):
        text = 'Hello world !prompt "remember this" and more text'
        cleaned, cmds = parse_commands(text)
        self.assertNotIn("!prompt", cleaned)
        self.assertIn("Hello world", cleaned)
        self.assertIn("and more text", cleaned)

    def test_temperature_stripped_from_cleaned(self):
        text = 'Some text !temperature 0.7 more text'
        cleaned, cmds = parse_commands(text)
        self.assertNotIn("!temperature", cleaned)

    def test_whisper_stripped_from_cleaned(self):
        text = 'Hello !whisper "AI-1" "secret" visible text'
        cleaned, cmds = parse_commands(text)
        self.assertNotIn("!whisper", cleaned)
        self.assertIn("visible text", cleaned)

    def test_image_not_stripped_from_cleaned(self):
        text = 'Look at this !image "cool art" nice right?'
        cleaned, cmds = parse_commands(text)
        self.assertIn("!image", cleaned)

    def test_multiple_commands(self):
        text = '!image "art" !search "news" !mute_self'
        cleaned, cmds = parse_commands(text)
        actions = {c.action for c in cmds}
        self.assertIn("image", actions)
        self.assertIn("search", actions)
        self.assertIn("mute_self", actions)

    def test_no_commands(self):
        text = "Just a normal response with no commands."
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 0)
        self.assertEqual(cleaned, text)

    def test_empty_string(self):
        cleaned, cmds = parse_commands("")
        self.assertEqual(len(cmds), 0)
        self.assertEqual(cleaned, "")

    def test_case_insensitive(self):
        text = '!IMAGE "test" !SEARCH "query"'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 2)

    def test_list_models(self):
        text = "Let me check !list_models"
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].action, "list_models")

    def test_remove_ai(self):
        text = '!remove_ai "AI-3"'
        cleaned, cmds = parse_commands(text)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].params["target"], "AI-3")


class TestFormatCommandResult(unittest.TestCase):

    def test_success_format(self):
        result = format_command_result("image", True, "Image generated")
        self.assertIn("✓", result)
        self.assertIn("image", result)

    def test_failure_format(self):
        result = format_command_result("search", False, "Search failed")
        self.assertIn("✗", result)
        self.assertIn("search", result)


if __name__ == "__main__":
    unittest.main()
