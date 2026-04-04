# messages.py
"""
Message schema helpers for liminal_backrooms.

Provides factory functions and type constants for the conversation message
format used throughout the application. All messages are plain dicts for
backwards compatibility, but these helpers enforce a consistent structure.
"""

from enum import Enum


class MessageType(str, Enum):
    """Known message types stored in the ``_type`` key."""
    AGENT_NOTIFICATION = "agent_notification"
    SYSTEM_NOTIFICATION = "system_notification"
    GENERATED_IMAGE = "generated_image"
    WHISPER = "whisper"
    BRANCH_INDICATOR = "branch_indicator"


def make_system_notification(content: str, **extra) -> dict:
    """Create a system notification message dict."""
    msg = {
        "role": "system",
        "content": content,
        "_type": MessageType.SYSTEM_NOTIFICATION,
    }
    msg.update(extra)
    return msg


def make_agent_notification(content: str, success: bool | None = None, **extra) -> dict:
    """Create an agent command notification message dict."""
    msg = {
        "role": "system",
        "content": content,
        "_type": MessageType.AGENT_NOTIFICATION,
        "_command_success": success,
    }
    msg.update(extra)
    return msg


def make_assistant_message(content, ai_name: str, model: str, **extra) -> dict:
    """Create a standard assistant (AI response) message dict."""
    msg = {
        "role": "assistant",
        "content": content,
        "ai_name": ai_name,
        "model": model,
    }
    msg.update(extra)
    return msg


def make_user_message(content: str, user_name: str = "User", **extra) -> dict:
    """Create a user message dict."""
    msg = {
        "role": "user",
        "content": content,
        "_user_name": user_name,
    }
    msg.update(extra)
    return msg


def make_whisper(content: str, from_ai: str, to_ai: str, **extra) -> dict:
    """Create a whisper (private message) dict."""
    msg = {
        "role": "system",
        "content": content,
        "_type": MessageType.WHISPER,
        "_whisper_from": from_ai,
        "_whisper_to": to_ai,
    }
    msg.update(extra)
    return msg


def is_visible(msg: dict) -> bool:
    """Check if a message should be visible in the conversation display."""
    if not isinstance(msg, dict):
        return False
    return not msg.get("hidden", False)


def is_from_ai(msg: dict, ai_name: str) -> bool:
    """Check if a message was sent by a specific AI."""
    return msg.get("ai_name") == ai_name


def get_type(msg: dict) -> str | None:
    """Get the message type (``_type`` key), or None."""
    return msg.get("_type")
