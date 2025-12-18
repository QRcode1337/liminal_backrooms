from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
import requests
from anthropic import Anthropic
from openai import OpenAI
import replicate
from src.core.models import Message

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: List[Message], model: str, system_prompt: str) -> Dict[str, Any]:
        pass

class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def generate_response(self, prompt: str, history: List[Message], model: str, system_prompt: str) -> Dict[str, Any]:
        try:
            # Filter and convert messages
            messages = []
            seen_contents = set()

            for msg in history:
                if msg.role == "system":
                    continue
                if msg.content in seen_contents:
                    continue
                seen_contents.add(msg.content)
                messages.append({"role": msg.role, "content": msg.content})

            messages.append({"role": "user", "content": prompt})

            # API Call
            response = self.client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=1,
                system=system_prompt,
                messages=messages
            )

            return {
                "content": response.content[0].text,
                "role": "assistant"
            }
        except Exception as e:
            return {"error": str(e), "content": f"Error: {str(e)}", "role": "system"}

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt: str, history: List[Message], model: str, system_prompt: str) -> Dict[str, Any]:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4000,
                temperature=1
            )
            return {
                "content": response.choices[0].message.content,
                "role": "assistant"
            }
        except Exception as e:
            return {"error": str(e), "content": f"Error: {str(e)}", "role": "system"}

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_response(self, prompt: str, history: List[Message], model: str, system_prompt: str) -> Dict[str, Any]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:3000",
                "Content-Type": "application/json"
            }

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history:
                if msg.role != "system":
                    messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 1
            }

            response = requests.post(self.url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                return {"content": content, "role": "assistant"}
            else:
                return {"error": f"Status {response.status_code}", "content": f"Error: {response.text}", "role": "system"}
        except Exception as e:
            return {"error": str(e), "content": f"Error: {str(e)}", "role": "system"}

class DeepSeekReplicateProvider(LLMProvider):
    """Uses Replicate for DeepSeek models"""
    def generate_response(self, prompt: str, history: List[Message], model: str, system_prompt: str) -> Dict[str, Any]:
        try:
            # DeepSeek via Replicate expects a single prompt string often
            formatted_history = ""
            if system_prompt:
                formatted_history += f"System: {system_prompt}\n"

            for msg in history:
                 formatted_history += f"{msg.role.capitalize()}: {msg.content}\n"

            formatted_history += f"User: {prompt}\n"

            output = replicate.run(
                "deepseek-ai/deepseek-r1",
                input={
                    "prompt": formatted_history,
                    "max_tokens": 8000,
                    "temperature": 1
                }
            )

            response_text = "".join(output) if isinstance(output, list) else str(output)
            return {"content": response_text, "role": "assistant"}

        except Exception as e:
            return {"error": str(e), "content": f"Error: {str(e)}", "role": "system"}

class LLMService:
    def __init__(self):
        self.providers = {}
        # Initialize providers lazily or upfront
        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers["claude"] = ClaudeProvider(os.getenv("ANTHROPIC_API_KEY"))
        if os.getenv("OPENAI_API_KEY"):
            self.providers["openai"] = OpenAIProvider(os.getenv("OPENAI_API_KEY"))
        if os.getenv("OPENROUTER_API_KEY"):
            self.providers["openrouter"] = OpenRouterProvider(os.getenv("OPENROUTER_API_KEY"))
        self.providers["deepseek"] = DeepSeekReplicateProvider() # Assumes env var set for replicate if needed internally

    def get_provider(self, model_id: str) -> LLMProvider:
        if "claude" in model_id.lower() and "anthropic" in model_id.lower():
            return self.providers.get("claude", self.providers.get("openrouter"))
        elif "gpt" in model_id.lower() and "openai" in model_id.lower():
            return self.providers.get("openai", self.providers.get("openrouter"))
        elif "deepseek" in model_id.lower() and "replicate" in model_id.lower():
            return self.providers.get("deepseek")
        # Default to OpenRouter for everything else
        return self.providers.get("openrouter")

    def generate_response(self, prompt: str, history: List[Message], model: str, system_prompt: str) -> Dict[str, Any]:
        provider = self.get_provider(model)
        if not provider:
             return {"error": "No provider available", "content": "Configuration Error: No API provider found.", "role": "system"}
        return provider.generate_response(prompt, history, model, system_prompt)
