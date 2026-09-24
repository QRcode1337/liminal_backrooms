#!/usr/bin/env python3
"""Tests for !image backend routing (Grok Imagine + OpenAI GPT Image)."""

import unittest
from unittest.mock import Mock, patch

import shared_utils


class ChooseImageBackendsTests(unittest.TestCase):
    def test_grok_caller_tries_xai_then_openai(self):
        backends = shared_utils.choose_image_backends("xai/grok-4.20-0309-non-reasoning")
        names = [b[0] for b in backends]
        self.assertEqual(names[0], "xai")
        self.assertIn("openai", names)
        self.assertEqual(backends[0][1], "grok-imagine-image-2.0")

    def test_codex_caller_tries_openai_first(self):
        backends = shared_utils.choose_image_backends("cx/gpt-5.6-luna-medium")
        self.assertEqual(backends[0][0], "openai")
        self.assertEqual(backends[0][1], "gpt-image-2")
        self.assertIn("xai", [b[0] for b in backends])

    def test_codex_prefix_same_as_cx(self):
        backends = shared_utils.choose_image_backends("codex/gpt-5.6-sol-high")
        self.assertEqual(backends[0][0], "openai")

    def test_default_starts_with_grok(self):
        backends = shared_utils.choose_image_backends(None)
        self.assertEqual(backends[0], ("xai", "grok-imagine-image-2.0"))


class GenerateImageFromTextTests(unittest.TestCase):
    def test_xai_success_saves_b64(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "data": [{"b64_json": "aGVsbG8="}]  # "hello"
        }
        with patch.object(shared_utils.os, "getenv", side_effect=lambda k, d=None: {
            "XAI_API_KEY": "xai-test",
            "OPENAI_API_KEY": None,
            "OPENROUTER_API_KEY": None,
        }.get(k, d)):
            with patch.object(shared_utils.requests, "post", return_value=response) as post:
                with patch.object(shared_utils, "_write_image_bytes", return_value="images/out.png"):
                    result = shared_utils.generate_image_from_text(
                        "a red cube", caller_model="xai/grok-4.6"
                    )
        self.assertTrue(result["success"])
        self.assertEqual(result["model"], "grok-imagine-image-2.0")
        self.assertEqual(result["backend"], "xai")
        self.assertIn("images/generations", post.call_args.args[0])

    def test_falls_back_to_openai_when_xai_fails(self):
        xai_resp = Mock(status_code=400)
        xai_resp.text = "nope"
        oai_resp = Mock(status_code=200)
        oai_resp.json.return_value = {"data": [{"b64_json": "aGVsbG8="}]}

        def fake_post(url, **kwargs):
            if "x.ai" in url:
                return xai_resp
            return oai_resp

        with patch.object(shared_utils.os, "getenv", side_effect=lambda k, d=None: {
            "XAI_API_KEY": "xai-test",
            "OPENAI_API_KEY": "oai-test",
            "OPENROUTER_API_KEY": None,
        }.get(k, d)):
            with patch.object(shared_utils.requests, "post", side_effect=fake_post):
                with patch.object(shared_utils, "_write_image_bytes", return_value="images/out.png"):
                    result = shared_utils.generate_image_from_text(
                        "a red cube", caller_model="xai/grok-4.6"
                    )
        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "openai")
        self.assertEqual(result["model"], "gpt-image-2")


if __name__ == "__main__":
    unittest.main()
