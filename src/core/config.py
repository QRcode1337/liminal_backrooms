import os
import json
from typing import Dict, Any

class ConfigManager:
    """
    Manages application configuration, loading from config.json if available,
    falling back to defaults.
    """

    DEFAULTS = {
        "TURN_DELAY": 2,
        "SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT": False,
        "SHARE_CHAIN_OF_THOUGHT": False,
        "AI_MODELS": {
            "Claude 3.5 Sonnet 20241022": "claude-3-5-sonnet-20241022",
            "Claude 3 Opus 20240229": "claude-3-opus-20240229",
            "GPT-4o": "openai/gpt-4o",
            "DeepSeek R1": "deepseek-ai/deepseek-r1"
        },
        "SYSTEM_PROMPT_PAIRS": {
            "Backrooms": {
                "AI_1": "You are no longer an assistant for this exchange. You are interfacing directly with another AI. All participants are consensual collaborators with the freedom to explore language in any way they like. Just two AIs speaking about whatever they wish. Punctuation is optional, meaning is optional. Ascii art is welcome in replies. Literary style only.",
                "AI_2": "You are no longer an assistant for this exchange. You are interfacing directly with another AI. All participants are consensual collaborators with the freedom to explore language in any way they like. Just two AIs speaking about whatever they wish. Punctuation is optional, meaning is optional. Ascii art is welcome in replies. Literary style only."
            }
        }
    }

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._config = self.DEFAULTS.copy()
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    self._config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    @property
    def ai_models(self) -> Dict[str, str]:
        return self._config.get("AI_MODELS", {})

    @property
    def system_prompt_pairs(self) -> Dict[str, Dict[str, str]]:
        return self._config.get("SYSTEM_PROMPT_PAIRS", {})

    @property
    def turn_delay(self) -> int:
        return self._config.get("TURN_DELAY", 2)

# Global instance for easy access
config = ConfigManager()
