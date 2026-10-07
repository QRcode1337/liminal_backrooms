import unittest
from unittest.mock import Mock, mock_open, patch

import shared_utils
from tools import model_updater


class OmniRouteTests(unittest.TestCase):
    def test_non_streaming_request_uses_local_omniroute(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        with patch.object(shared_utils.requests, "post", return_value=response) as post:
            result = shared_utils.call_omniroute_api(
                "ping", [], "agy/gemini-3.5-flash-medium", "system"
            )

        self.assertEqual(result, "OK")
        self.assertEqual(
            post.call_args.args[0], "http://127.0.0.1:20128/v1/chat/completions"
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["model"],
            "agy/gemini-3.5-flash-medium",
        )

    def test_streaming_request_emits_tokens(self):
        response = Mock(status_code=200)
        response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"O"}}]}',
            b'data: {"choices":[{"delta":{"content":"K"}}]}',
            b"data: [DONE]",
        ]
        tokens = []
        with patch.object(shared_utils.requests, "post", return_value=response):
            result = shared_utils.call_omniroute_api(
                "ping", [], "cx/gpt-5.6-sol-high", stream_callback=tokens.append
            )

        self.assertEqual(result, "OK")
        self.assertEqual(tokens, ["O", "K"])

    def test_health_check_uses_models_endpoint(self):
        response = Mock(status_code=200)
        response.json.return_value = {"data": [{"id": "cx/gpt-5.6-sol-high"}]}
        with patch.object(shared_utils.requests, "get", return_value=response) as get:
            ok, message = shared_utils.check_omniroute_health()

        self.assertTrue(ok)
        self.assertIn("1 models", message)
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:20128/v1/models")


class OmniRouteModelUpdaterTests(unittest.TestCase):
    def test_rejects_pre_omniroute_cache(self):
        with patch.object(model_updater, "CACHE_FILE") as cache_file:
            cache_file.exists.return_value = True
            with patch("builtins.open", mock_open(read_data='{"cached_at": "2026-08-25T00:00:00", "model_ids": ["openai/gpt-5.2"]}')):
                self.assertIsNone(model_updater.load_cached_ids())

    def test_fetch_uses_omniroute_models_endpoint(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "data": [
                {"id": "anthropic/claude-opus-4-6"},
                {"id": "cx/gpt-5.6-sol-high"},
            ]
        }
        with patch.object(model_updater.requests, "get", return_value=response) as get:
            ids = model_updater.fetch_available_model_ids()

        self.assertEqual(
            ids, {"anthropic/claude-opus-4-6", "cx/gpt-5.6-sol-high"}
        )
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:20128/v1/models")

    def test_validate_keeps_live_ids_and_sora(self):
        curated = {
            "SOTA": {
                "Anthropic": {"Claude Opus 4.6": "anthropic/claude-opus-4-6"},
                "OpenAI": {
                    "GPT 5.2": "openai/gpt-5.2",
                    "Sora 2": "sora-2",
                },
            }
        }
        with patch.object(
            model_updater,
            "get_available_ids",
            return_value={"anthropic/claude-opus-4-6"},
        ):
            validated = model_updater.validate_models(curated)

        self.assertEqual(
            validated["SOTA"]["Anthropic"]["Claude Opus 4.6"],
            "anthropic/claude-opus-4-6",
        )
        self.assertEqual(validated["SOTA"]["OpenAI"]["Sora 2"], "sora-2")
        self.assertNotIn("GPT 5.2", validated["SOTA"]["OpenAI"])

    def test_validate_drops_openrouter_style_ids(self):
        curated = {
            "SOTA": {
                "Anthropic": {"Stale": "anthropic/claude-opus-4.6"},
            }
        }
        with patch.object(
            model_updater,
            "get_available_ids",
            return_value={"anthropic/claude-opus-4-6"},
        ):
            validated = model_updater.validate_models(curated)

        self.assertEqual(validated["SOTA"], {})

    def test_pretty_model_name(self):
        self.assertEqual(
            model_updater.pretty_model_name("agy/gemini-3.5-flash-medium"),
            "Gemini 3.5 Flash Medium",
        )
        self.assertEqual(
            model_updater.pretty_model_name("cx/gpt-5.6-sol-high"),
            "GPT 5.6 Sol High",
        )

    def test_build_endpoint_catalog_groups_prefixes(self):
        catalog = model_updater.build_endpoint_catalog({
            "agy/gemini-3.5-flash-medium",
            "cx/gpt-5.6-sol-high",
            "xai/grok-4.6",
            "grok-cli/grok-composer-2.5-fast",
            "nous/anthropic/claude-opus-4.6",
            "xai/grok-imagine-image",
        })
        self.assertIn("agy/gemini-3.5-flash-medium", catalog["Antigravity"].values())
        self.assertIn("cx/gpt-5.6-sol-high", catalog["Codex (cx)"].values())
        self.assertIn("xai/grok-4.6", catalog["xAI"].values())
        self.assertIn("grok-cli/grok-composer-2.5-fast", catalog["Grok CLI"].values())
        self.assertNotIn("nous/anthropic/claude-opus-4.6", {
            mid for models in catalog.values() for mid in models.values()
        })
        self.assertNotIn("xai/grok-imagine-image", catalog.get("xAI", {}).values())

    def test_merge_puts_endpoints_first(self):
        curated = {
            "SOTA": {"OpenAI": {"GPT 5.2": "openai/gpt-5.2"}},
            "OmniRoute": {"Codex": {"Sol": "cx/gpt-5.6-sol-high"}},
        }
        merged = model_updater.merge_endpoint_catalog(
            curated,
            endpoints={"Antigravity": {"Flash [agy]": "agy/gemini-3.5-flash-medium"}},
        )
        self.assertEqual(list(merged.keys())[0], "Endpoints")
        self.assertNotIn("OmniRoute", merged)


if __name__ == "__main__":
    unittest.main()
