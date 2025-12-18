import pytest
from src.core.models import Message, Branch
from src.core.conversation_manager import ConversationManager
from src.core.config import config
import os

def test_config_loading():
    assert config.get("TURN_DELAY") is not None
    assert "AI_MODELS" in config._config

def test_message_model():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    d = msg.to_dict()
    assert d["role"] == "user"

def test_branch_creation():
    manager = ConversationManager()
    manager.add_user_message("Hello world")

    # Test main conversation
    assert len(manager.main_conversation) == 1
    assert manager.main_conversation[0].content == "Hello world"

    # Test branching
    bid = manager.create_branch("rabbithole", "world", parent_id=None)
    assert bid in manager.branches
    assert manager.branches[bid].type == "rabbithole"
    assert len(manager.branches[bid].conversation) > 1 # History + indicator + prompt

def test_manager_switching():
    manager = ConversationManager()
    manager.add_user_message("Main")

    bid = manager.create_branch("fork", "Main")
    assert manager.active_branch_id == bid

    current_msgs = manager.get_current_conversation()
    assert current_msgs[-1].role == "user" # The prompt for fork

    manager.switch_branch("main")
    assert manager.active_branch_id is None
    assert manager.get_current_conversation()[0].content == "Main"
