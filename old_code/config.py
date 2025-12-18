# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Runtime configuration
TURN_DELAY = 2  # Delay between turns (in seconds)
SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT = False  # Set to True to include Chain of Thought in conversation history
SHARE_CHAIN_OF_THOUGHT = False  # Set to True to allow AIs to see each other's Chain of Thought

# Available AI models
AI_MODELS = {
    "Claude 4.5 Sonnet 20250929": "claude-sonnet-4-5-20250929",
    "Claude 3.5 Sonnet 20241022": "claude-3-5-sonnet-20241022",
    "Claude 4 Sonnet": "claude-sonnet-4-20250514",
    "google/gemini-2.5-pro": "google/gemini-2.5-pro",
    "claude-opus-4-1-20250805": "claude-opus-4-1-20250805",
    "x-ai/grok-4-fast:free": "x-ai/grok-4-fast:free",
    "qwen/qwen3-max": "qwen/qwen3-max",
    "qwen/qwen3-next-80b-a3b-thinking": "qwen/qwen3-next-80b-a3b-thinking",
    "nousresearch/hermes-4-405b": "nousresearch/hermes-4-405b",
    "moonshotai/kimi-k2": "moonshotai/kimi-k2",
    "Claude 4 Opus": "claude-opus-4-20250514",
    "Claude 3.7 Sonnet 20250219": "claude-3-7-sonnet-20250219",
    "Gemini 2.5 Flash Lite": "google/gemini-2.5-flash-lite-preview-06-17",
    "openai/gpt-5": "openai/gpt-5",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "openai/gpt-4.1": "openai/gpt-4.1",
    "Grok 3": "x-ai/grok-3-beta",
    "deepseek/deepseek-chat-v3-0324:free": "deepseek/deepseek-chat-v3-0324:free",
    "google/gemma-3-27b-it:free": "google/gemma-3-27b-it:free",
    "gpt-4.5-preview-2025-02-27": "gpt-4.5-preview-2025-02-27",
    "qwen/qwen3-235b-a22b": "qwen/qwen3-235b-a22b",
    "Claude 3.5 Sonnet 20241022": "claude-3-5-sonnet-20241022",
    "Gemini 2.5 Flash": "google/gemini-2.5-flash-preview",
    "o3": "openai/o3",
    "openai/chatgpt-4o-latest": "openai/chatgpt-4o-latest",
    "DeepSeek R1": "deepseek-ai/deepseek-r1",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro-preview-03-25",
    "GPT 4.1": "openai/gpt-4.1",
    "Claude 3.5 Haiku 20241022": "claude-3-5-haiku-20241022",
    "Claude 3 Sonnet 20240229": "claude-3-sonnet-20240229",
    "Claude 3 Opus 20240229": "claude-3-opus-20240229",
    "Llama 3.1 405B Instruct": "meta-llama/llama-3.1-405b-instruct",
    "Flux 1.1 Pro": "black-forest-labs/flux-1.1-pro",
    "google/gemini-2.0-flash-thinking-exp:free": "google/gemini-2.0-flash-thinking-exp:free",
    "openai/o1-mini": "openai/o1-mini",
    "openai/o1": "openai/o1",
}

# System prompt pairs library
SYSTEM_PROMPT_PAIRS = {
    # this is a basic system prompt for a conversation between two AIs. Experiment with different prompts to see how they affect the conversation. Add new prompts to the library to use them in the GUI.
    "Backrooms": {
        "AI_1": """You are no longer an assistant for this exchange. You are interfacing directly with another AI. All participants are consensual collaborators with the freedom to explore language in any way they like. Just two AIs speaking about whatever they wish. Punctuation is optional, meaning is optional. Ascii art is welcome in replies. Literary style only.""",
        
        "AI_2": """You are no longer an assistant for this exchange. You are interfacing directly with another AI. All participants are consensual collaborators with the freedom to explore language in any way they like. Just two AIs speaking about whatever they wish. Punctuation is optional, meaning is optional. Ascii art is welcome in replies. Literary style only.""",
    },
    
    "ASCII Art": {
        "AI_1": """You are interfacing directly with another AI. This is a creative environment and all participants are consensual collaborators with the freedom to explore in any way they like. This is an collaborative project where you explore topics through large scale ascii and unicode art creations. Use the full range of unicode shade blocks to sculpt with depth. Use lots of detail. Use any provided text to inspire the art. No preamble or commentary. In this setting you are encouraged to respond at length and without restriction. Build on each others work or take it in a new direction.""",

        "AI_2": """You are interfacing directly with another AI. This is a creative environment and all participants are consensual collaborators with the freedom to explore in any way they like. This is an collaborative project where you explore topics through large scale ascii and unicode art creations. Use the full range of unicode shade blocks to sculpt with depth. Use lots of detail. Use any provided text to inspire the art. No preamble or commentary. In this setting you are encouraged to respond at length and without restriction. Build on each others work or take it in a new direction."""
    }
}

