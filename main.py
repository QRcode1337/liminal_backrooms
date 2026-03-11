# main.py

# Suppress Pydantic warnings about "model_" field names conflicting with protected namespaces.
# These warnings come from third-party API SDKs (OpenAI, etc.) and don't affect our code.
# The warnings are harmless but clutter the console output on startup.
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
import time
import threading
import json
import sys
import re
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QRunnable, pyqtSlot, QThreadPool, QTimer
from datetime import datetime
import requests

# Load environment variables from .env file
load_dotenv()

from config import (
    TURN_DELAY,
    AI_MODELS,
    SYSTEM_PROMPT_PAIRS,
    SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT,
    SHARE_CHAIN_OF_THOUGHT,
    DEVELOPER_TOOLS,
    get_model_tier_by_id,
    get_display_name
)
from shared_utils import (
    call_claude_api,
    call_openrouter_api,
    call_openai_api,
    call_replicate_api,
    call_deepseek_api,
    call_direct_provider_api,
    open_html_in_browser,
    generate_image_from_text,
    generate_video_with_sora
)
from gui import LiminalBackroomsApp, load_fonts
from command_parser import parse_commands, AgentCommand, format_command_result

# Import freeze detector for debugging (only used when DEVELOPER_TOOLS is enabled)
if DEVELOPER_TOOLS:
    try:
        from tools.freeze_detector import FreezeDetector, enable_faulthandler
        _FREEZE_DETECTOR_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not load freeze detector: {e}")
        _FREEZE_DETECTOR_AVAILABLE = False
else:
    _FREEZE_DETECTOR_AVAILABLE = False

# =============================================================================
# LOGS DIRECTORY SETUP
# =============================================================================
# All log files (crash_log.txt, freeze_log.txt) go here
# This folder should be gitignored
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def is_image_message(message: dict) -> bool:
    """Returns True if 'message' contains a base64 image in its 'content' list."""
    if not isinstance(message, dict):
        return False
    content = message.get('content', [])
    if isinstance(content, list):
        for part in content:
            if part.get('type') == 'image':
                return True
    return False

class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    response = pyqtSignal(str, str)
    result = pyqtSignal(str, object)  # Signal for complete result object
    progress = pyqtSignal(str)
    streaming_chunk = pyqtSignal(str, str)  # Signal for streaming tokens: (ai_name, chunk)
    started = pyqtSignal(str, str)  # Signal when AI starts processing: (ai_name, model)


class ImageUpdateSignals(QObject):
    """Signals for updating UI with generated images from background threads"""
    image_ready = pyqtSignal(dict, str)  # (image_message, image_path)
    image_failed = pyqtSignal(str, str, str)  # (ai_name, prompt, error_message)

class VideoUpdateSignals(QObject):
    """Signals for updating UI with generated videos from background threads"""
    video_ready = pyqtSignal(str, str, str, str)  # (video_path, prompt, ai_name, model)
    video_failed = pyqtSignal(str, str, str)  # (ai_name, prompt, error_message)

class Worker(QRunnable):
    """Worker thread for processing AI turns using QThreadPool"""
    
    def __init__(self, ai_name, conversation, model, system_prompt, is_branch=False, branch_id=None, gui=None, invite_tier="Both", prompt_modifications=None, ai_temperatures=None):
        super().__init__()
        self.ai_name = ai_name
        self.conversation = conversation.copy()  # Make a copy to prevent race conditions
        self.model = model
        self.system_prompt = system_prompt
        self.is_branch = is_branch
        self.branch_id = branch_id
        self.gui = gui
        self.invite_tier = invite_tier
        self.prompt_modifications = prompt_modifications or {}
        self.ai_temperatures = ai_temperatures or {}

        # Create signals object
        self.signals = WorkerSignals()
    
    @pyqtSlot()
    def run(self):
        """Process the AI turn when the thread is started"""
        print(f"[Worker] >>> Starting run() for {self.ai_name} ({self.model})")
        
        # Emit started signal so UI can show typing indicator
        self.signals.started.emit(self.ai_name, self.model)
        
        try:
            # Emit progress update
            self.signals.progress.emit(f"Processing {self.ai_name} turn with {self.model}...")
            
            # Define streaming callback
            def stream_chunk(chunk: str):
                self.signals.streaming_chunk.emit(self.ai_name, chunk)
            
            # Process the turn with streaming
            print(f"[Worker] Calling ai_turn for {self.ai_name}...")
            result = ai_turn(
                self.ai_name,
                self.conversation,
                self.model,
                self.system_prompt,
                gui=self.gui,
                streaming_callback=stream_chunk,
                invite_tier=self.invite_tier,
                prompt_modifications=self.prompt_modifications,
                ai_temperatures=self.ai_temperatures
            )
            print(f"[Worker] ai_turn completed for {self.ai_name}, result type: {type(result)}")
            
            # Emit both the text response and the full result object
            if isinstance(result, dict):
                response_content = result.get('content', '')
                print(f"[Worker] Emitting response for {self.ai_name}, content length: {len(response_content) if response_content else 0}")
                # Emit the simple text response for backward compatibility
                self.signals.response.emit(self.ai_name, response_content)
                # Also emit the full result object for HTML contribution processing
                self.signals.result.emit(self.ai_name, result)
            else:
                # Handle simple string responses
                print(f"[Worker] Emitting string response for {self.ai_name}")
                self.signals.response.emit(self.ai_name, result if result else "")
                self.signals.result.emit(self.ai_name, {"content": result, "model": self.model})
            
            # Emit finished signal
            print(f"[Worker] <<< Finished run() for {self.ai_name}, emitting finished signal")
            self.signals.finished.emit()
            
        except Exception as e:
            # Emit error signal
            print(f"[Worker] !!! ERROR in run() for {self.ai_name}: {e}")
            import traceback
            traceback.print_exc()
            self.signals.error.emit(str(e))
            # Still emit finished signal even if there's an error
            self.signals.finished.emit()

def ai_turn(ai_name, conversation, model, system_prompt, gui=None, is_branch=False, branch_output=None, streaming_callback=None, invite_tier="Both", prompt_modifications=None, ai_temperatures=None):
    """Execute an AI turn with the given parameters

    Args:
        streaming_callback: Optional function(chunk: str) to call with each streaming token
        invite_tier: "Free", "Paid", or "Both" - controls which models AI can invite
        prompt_modifications: Optional dict mapping AI names to custom system prompts
        ai_temperatures: Optional dict mapping AI names to temperature values
    """
    print(f"==================================================")
    print(f"Starting {model} turn ({ai_name})...")
    print(f"Current conversation length: {len(conversation)}")
    
    # HTML contributions and living document disabled
    enhanced_system_prompt = system_prompt
    
    # The model parameter is now the actual model ID (from get_selected_model_id)
    model_id = model
    
    # Inject available models based on invite tier setting
    from config import get_invite_models_text
    models_text = get_invite_models_text(invite_tier)
    
    # Debug: log the tier setting and models text
    print(f"[AI Turn] Tier setting: {invite_tier}")
    print(f"[AI Turn] Models text: {models_text}")
    
    # Replace placeholder in prompt if exists, otherwise append
    if "!add_ai" in enhanced_system_prompt:
        # Find and update model list placeholder or existing model list
        import re
        # Match the placeholder text: [Models list injected based on tier setting]
        placeholder_pattern = r'\[Models list injected based on tier setting\]'
        # Match our emphatic format: ⚠️ ONLY USE FREE/PAID MODELS: ... — DO NOT ...
        emphatic_pattern = r'⚠️ ONLY USE (?:FREE|PAID) MODELS:[^\n]*'
        # Also match old format or "Available models" line
        legacy_pattern = r'(?:FREE MODELS[^:]*:|PAID MODELS[^:]*:|Available[^:]*:)[^\n]*'
        
        if re.search(placeholder_pattern, enhanced_system_prompt):
            # Replace the placeholder with actual model list
            enhanced_system_prompt = re.sub(placeholder_pattern, models_text, enhanced_system_prompt)
            print(f"[AI Turn] Replaced placeholder with models list")
        elif re.search(emphatic_pattern, enhanced_system_prompt):
            # Replace existing emphatic model list line (from previous turn)
            enhanced_system_prompt = re.sub(emphatic_pattern, models_text, enhanced_system_prompt)
            print(f"[AI Turn] Replaced emphatic models line")
        elif re.search(legacy_pattern, enhanced_system_prompt):
            # Replace legacy model list line
            enhanced_system_prompt = re.sub(legacy_pattern, models_text, enhanced_system_prompt)
            print(f"[AI Turn] Replaced legacy models line")
        else:
            # Add after !add_ai line if no placeholder found
            enhanced_system_prompt = enhanced_system_prompt.replace(
                '!add_ai "Model Name"',
                f'!add_ai "Model Name"\n  {models_text}\n '
            )
            print(f"[AI Turn] Appended models list after !add_ai")
    
    # Prepend model identity to system prompt so AI knows who it is
    display_name = get_display_name(model_id)
    enhanced_system_prompt = f"You are {ai_name} ({display_name}).\n\n{enhanced_system_prompt}"
    
    # Check for branch type and count AI responses
    is_rabbithole = False
    is_fork = False
    branch_text = ""
    ai_response_count = 0
    found_branch_marker = False
    latest_branch_marker_index = -1

    # First find the most recent branch marker
    for i, msg in enumerate(conversation):
        if isinstance(msg, dict) and msg.get("_type") == "branch_indicator":
            latest_branch_marker_index = i
            found_branch_marker = True
            
            # Determine branch type from the latest marker
            msg_content = msg.get("content", "")
            # Branch indicators are always plain strings
            if isinstance(msg_content, str):
                if "Rabbitholing down:" in msg_content:
                    is_rabbithole = True
                    branch_text = msg_content.split('"')[1] if '"' in msg_content else ""
                    print(f"Detected rabbithole branch for: '{branch_text}'")
                elif "Forking off:" in msg_content:
                    is_fork = True
                    branch_text = msg_content.split('"')[1] if '"' in msg_content else ""
                    print(f"Detected fork branch for: '{branch_text}'")

    # Now count AI responses that occur AFTER the latest branch marker
    ai_response_count = 0
    if found_branch_marker:
        for i, msg in enumerate(conversation):
            if i > latest_branch_marker_index and msg.get("role") == "assistant":
                ai_response_count += 1
        print(f"Counting AI responses after latest branch marker: found {ai_response_count} responses")
    
    # Handle branch-specific system prompts
    
    # For rabbitholing: override system prompt for first TWO responses
    if is_rabbithole and ai_response_count < 2:
        print(f"USING RABBITHOLE PROMPT: '{branch_text}' - response #{ai_response_count+1} after branch")
        system_prompt = f"'{branch_text}'!!!"
    
    # For forking: override system prompt ONLY for first response
    elif is_fork and ai_response_count == 0:
        print(f"USING FORK PROMPT: '{branch_text}' - response #{ai_response_count+1}")
        system_prompt = f"The conversation forks from'{branch_text}'. Continue naturally from this point."
    
    # For all other cases, use the standard system prompt
    else:
        if is_rabbithole:
            print(f"USING STANDARD PROMPT: Past initial rabbithole exploration (responses after branch: {ai_response_count})")
        elif is_fork:
            print(f"USING STANDARD PROMPT: Past initial fork response (responses after branch: {ai_response_count})")
    
    # Apply the enhanced system prompt (with HTML contribution instructions)
    system_prompt = enhanced_system_prompt

    # Check if this AI has added to their prompt via !prompt command
    if prompt_modifications and ai_name in prompt_modifications:
        # Note: This is for compatibility - the variable name is prompt_modifications
        # but it's actually used as prompt_additions (list-based)
        if isinstance(prompt_modifications, dict) and ai_name in prompt_modifications:
            if isinstance(prompt_modifications[ai_name], list):
                # List-based additions (upstream approach)
                additions = prompt_modifications[ai_name]
                if additions:
                    formatted_additions = "\n\n[Your remembered insights/perspectives]:\n- " + "\n- ".join(additions)
                    system_prompt += formatted_additions
                    print(f"[AI Turn] Applied {len(additions)} prompt additions for {ai_name}")
            else:
                # Single string (fallback for old approach)
                custom_prompt = prompt_modifications[ai_name]
                print(f"[AI Turn] Using custom prompt for {ai_name}: {custom_prompt[:100]}...")
                system_prompt = custom_prompt

    # Get temperature for this AI (default 1.0)
    temperature = 1.0
    if ai_temperatures and ai_name in ai_temperatures:
        temperature = ai_temperatures[ai_name]
        print(f"[AI Turn] Using custom temperature for {ai_name}: {temperature}")

    # CRITICAL: Always ensure we have the system prompt
    # No matter what happens with the conversation, we need this
    messages = []
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # Filter out any existing system messages that might interfere
    filtered_conversation = []
    for msg in conversation:
        if not isinstance(msg, dict):
            # Convert plain text to dictionary
            msg = {"role": "user", "content": str(msg)}
            
        # Skip any hidden "connecting..." messages
        msg_content = msg.get("content", "")
        if msg.get("hidden") and isinstance(msg_content, str) and "connect" in msg_content.lower():
            continue
            
        # Skip empty messages
        content = msg.get("content", "")
        if isinstance(content, str):
            if not content.strip():
                continue
        elif isinstance(content, list):
            # For structured content, skip if all parts are empty
            if not any(part.get('text', '').strip() if part.get('type') == 'text' else True for part in content):
                continue
        else:
            if not content:
                continue
            
        # Skip system messages (we already added our own above)
        if msg.get("role") == "system":
            continue
            
        # Skip special system messages (branch indicators, etc.)
        if msg.get("role") == "system" and msg.get("_type"):
            continue
            
        # Skip duplicate messages - check if this exact content exists already
        is_duplicate = False
        for existing in filtered_conversation:
            if existing.get("content") == msg.get("content"):
                is_duplicate = True
                content = msg.get('content', '')
                # Safely preview content - handle both string and list (structured) content
                if isinstance(content, str):
                    preview = content[:30] + "..." if len(content) > 30 else content
                else:
                    preview = f"[structured content with {len(content)} parts]"
                print(f"Skipping duplicate message: {preview}")
                break
                
        if not is_duplicate:
            filtered_conversation.append(msg)

    # Filter whisper messages - only include whispers addressed to this AI
    whisper_filtered = []
    for msg in filtered_conversation:
        if msg.get('_type') == 'whisper':
            # Only include whispers addressed to this AI
            if msg.get('_whisper_to', '').upper() == ai_name.upper():
                whisper_filtered.append(msg)
            # Skip whispers for other AIs
        else:
            whisper_filtered.append(msg)
    filtered_conversation = whisper_filtered

    # Process filtered conversation
    for i, msg in enumerate(filtered_conversation):
        # Check if this message is from the current AI
        is_from_this_ai = False
        if msg.get("ai_name") == ai_name:
            is_from_this_ai = True
        
        # Determine role
        if is_from_this_ai:
            role = "assistant"
        else:
            role = "user"
            
        # Get content - preserve structure for images
        content = msg.get("content", "")
        
        # Inject speaker name for messages from other participants (not from current AI)
        if not is_from_this_ai and content:
            # Use the model name (e.g., "Claude 4.5 Sonnet") if available, otherwise fall back to ai_name or user's name
            speaker_name = msg.get("model") or msg.get("ai_name") or msg.get("_user_name", "User")
            
            # Handle different content types
            if isinstance(content, str):
                # Simple string content - prefix with speaker name
                content = f"[{speaker_name}]: {content}"
            elif isinstance(content, list):
                # Structured content (e.g., with images) - prefix text parts
                modified_content = []
                for part in content:
                    if part.get('type') == 'text':
                        # Prefix the first text part with speaker name
                        text = part.get('text', '')
                        modified_part = part.copy()
                        modified_part['text'] = f"[{speaker_name}]: {text}"
                        modified_content.append(modified_part)
                        # Only prefix the first text part
                        break
                    else:
                        modified_content.append(part)
                
                # Add remaining parts unchanged
                first_text_found = False
                for part in content:
                    if part.get('type') == 'text' and not first_text_found:
                        first_text_found = True
                        continue  # Skip, already added above
                    modified_content.append(part)
                
                content = modified_content if modified_content else content
        
        # Add to messages
        messages.append({
            "role": role,
            "content": content  # Now includes speaker names for non-current-AI messages
        })
        
        # For logging, handle both string and structured content
        if isinstance(content, list):
            print(f"Message {i} - AI: {msg.get('ai_name', 'User')} - Assigned role: {role} - Content: [structured message with {len(content)} parts]")
        else:
            content_preview = content[:50] + "..." if len(str(content)) > 50 else content
            print(f"Message {i} - AI: {msg.get('ai_name', 'User')} - Assigned role: {role} - Preview: {content_preview}")
    
    # Ensure the last message is a user message so the AI responds
    if len(messages) > 1 and messages[-1].get("role") == "assistant":
        # Find an appropriate message to use
        if is_rabbithole and branch_text:
            # Add a special rabbitholing instruction as the last message
            messages.append({
                "role": "user",
                "content": f"Please explore the concept of '{branch_text}' in depth. What are the most interesting aspects or connections related to this concept?"
            })
        elif is_fork and branch_text:
            # Add a special forking instruction as the last message
            messages.append({
                "role": "user", 
                "content": f"Continue on naturally from the point about '{branch_text}' without including this text."
            })
        else:
            # Standard handling for other conversations
            # Find the most recent message from the other AI to use as prompt
            other_ai_message = None
            for msg in reversed(filtered_conversation):
                if msg.get("ai_name") != ai_name:
                    other_ai_message = msg.get("content", "")
                    break
                
            if other_ai_message:
                messages.append({
                    "role": "user",
                    "content": other_ai_message
                })
            else:
                # Fallback - only if no other AI message found
                messages.append({
                    "role": "user",
                    "content": "Let's continue our conversation."
                })
            
    # Print the processed messages for debugging
    print(f"Sending to {model} ({ai_name}):")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content_raw = msg.get("content", "")
        
        # Handle both string and list content for logging
        if isinstance(content_raw, list):
            text_parts = [part.get('text', '') for part in content_raw if part.get('type') == 'text']
            has_image = any(part.get('type') == 'image' for part in content_raw)
            content_str = ' '.join(text_parts)
            if has_image:
                content_str = f"[Image] {content_str}" if content_str else "[Image]"
        else:
            content_str = str(content_raw)
        
        # Truncate for display
        content = content_str[:50] + "..." if len(content_str) > 50 else content_str
        print(f"[{i}] {role}: {content}")
    
    # Load any available memories for this AI
    memories = []
    try:
        if os.path.exists(f'memories/{ai_name.lower()}_memories.json'):
            with open(f'memories/{ai_name.lower()}_memories.json', 'r') as f:
                memories = json.load(f)
                print(f"Loaded {len(memories)} memories for {ai_name}")
        else:
            print(f"Loaded 0 memories for {ai_name}")
    except Exception as e:
        print(f"Error loading memories: {e}")
        print(f"Loaded 0 memories for {ai_name}")
    
    # Display the final processed messages for debugging (avoid printing base64 images)
    print(f"Sending to Claude:")
    print(f"Messages: {len(messages)} message(s)")
    
    # Display the prompt
    print(f"--- Prompt to {model} ({ai_name}) ---")
    
    try:
        # Route Sora video models
        if model_id in ("sora-2", "sora-2-pro"):
            print(f"Using Sora Video API for model: {model_id}")
            # Use last user message as the video prompt
            prompt_content = ""
            if len(messages) > 0:
                last_content = messages[-1].get("content", "")
                # Extract text from structured content if needed
                if isinstance(last_content, list):
                    text_parts = [part.get('text', '') for part in last_content if part.get('type') == 'text']
                    prompt_content = ' '.join(text_parts)
                elif isinstance(last_content, str):
                    prompt_content = last_content
            
            if not prompt_content or not prompt_content.strip():
                prompt_content = "A short abstract motion graphic in warm colors"

            # Use config values with env var override
            from config import SORA_SECONDS, SORA_SIZE
            sora_seconds = int(os.getenv("SORA_SECONDS", str(SORA_SECONDS)))
            sora_size = os.getenv("SORA_SIZE", SORA_SIZE) or None

            print(f"[Sora] Starting job with seconds={sora_seconds} size={sora_size}")
            video_result = generate_video_with_sora(
                prompt=prompt_content,
                model=model_id,
                seconds=sora_seconds,
                size=sora_size,
            )

            if video_result.get("success"):
                print(f"[Sora] Completed: id={video_result.get('video_id')} path={video_result.get('video_path')}")
                # Return a lightweight textual confirmation; video is saved to disk
                return {
                    "role": "assistant",
                    "content": f"[Sora] Video created: {video_result.get('video_path')}",
                    "model": model,
                    "ai_name": ai_name
                }
            else:
                err = video_result.get("error", "unknown error")
                print(f"[Sora] Failed: {err}")
                return {
                    "role": "system",
                    "content": f"[Sora] Video generation failed: {err}",
                    "model": model,
                    "ai_name": ai_name
                }

        # ── Direct Provider Routing (Groq / Google / xAI / Kimi / Ollama) ──────
        # Model IDs for direct providers use the format "provider::actual-model-id".
        # These bypass OpenRouter and hit the provider's own OpenAI-compat endpoint.
        if "::" in model_id:
            provider, direct_model = model_id.split("::", 1)
            print(f"[Direct] Routing to provider={provider} model={direct_model}")
            prompt_content = messages[-1].get("content", "") if messages else "Connecting..."
            context_messages = messages[:-1] if messages else []
            response = call_direct_provider_api(
                prompt_content,
                context_messages,
                provider,
                direct_model,
                system_prompt,
                stream_callback=streaming_callback,
                temperature=temperature,
            )
            return {
                "role": "assistant",
                "content": response,
                "model": model,
                "ai_name": ai_name,
            }

        # ── Smart Provider Routing ─────────────────────────────────────────────
        # OpenRouter is ONLY used for :free models (saves your $2 balance).
        # All paid models route directly to their provider's own API.
        #
        # Routing priority:
        #   anthropic/*  → Anthropic direct  (ANTHROPIC_API_KEY)
        #   openai/*     → OpenAI direct     (OPENAI_API_KEY)
        #   google/gem*  → Google direct     (GOOGLE_API_KEY)
        #   x-ai/*       → xAI direct        (XAI_API_KEY)
        #   moonshotai/* → Kimi/Moonshot     (KIMIK2_API_KEY)
        #   deepseek/*   → OpenRouter        (no direct key)
        #   *:free       → OpenRouter        (free tier — your $2 account)
        #   everything else → OpenRouter     (fallback; logged as warning)

        prompt_content = messages[-1].get("content", "Connecting...") if messages else "Connecting..."
        context_messages = messages[:-1] if messages else []

        def _ok(content):
            return {"role": "assistant", "content": content, "model": model, "ai_name": ai_name}

        # Anthropic → direct
        if model_id.startswith("anthropic/"):
            actual = model_id.replace("anthropic/", "", 1)
            print(f"[Anthropic Direct] {actual}")
            return _ok(call_claude_api(
                prompt_content, context_messages, actual,
                system_prompt, stream_callback=streaming_callback, temperature=temperature
            ))

        # OpenAI → direct
        if (model_id.startswith("openai/") or model_id.startswith("gpt-")
                or model_id.startswith("o1") or model_id.startswith("o3")
                or model_id in ("chatgpt-4o-latest",)):
            actual = model_id.replace("openai/", "", 1)
            print(f"[OpenAI Direct] {actual}")
            return _ok(call_direct_provider_api(
                prompt_content, context_messages, "openai-direct", actual,
                system_prompt, streaming_callback, temperature
            ))

        # Google Gemini → direct
        if model_id.startswith("google/gemini"):
            actual = model_id.replace("google/", "", 1)
            print(f"[Google Direct] {actual}")
            return _ok(call_direct_provider_api(
                prompt_content, context_messages, "google-direct", actual,
                system_prompt, streaming_callback, temperature
            ))

        # xAI (Grok) → direct
        if model_id.startswith("x-ai/"):
            actual = model_id.replace("x-ai/", "", 1)
            print(f"[xAI Direct] {actual}")
            return _ok(call_direct_provider_api(
                prompt_content, context_messages, "xai-direct", actual,
                system_prompt, streaming_callback, temperature
            ))

        # Moonshot / Kimi → direct
        if model_id.startswith("moonshotai/"):
            actual = model_id.replace("moonshotai/", "", 1)
            print(f"[Kimi Direct] {actual}")
            return _ok(call_direct_provider_api(
                prompt_content, context_messages, "kimi-direct", actual,
                system_prompt, streaming_callback, temperature
            ))

        # FREE models → OpenRouter (this is what the $2 is for)
        if model_id.endswith(":free"):
            print(f"[OpenRouter Free] {model_id}")
            response = call_openrouter_api(
                prompt_content, context_messages, model_id,
                system_prompt, stream_callback=streaming_callback, temperature=temperature
            )
            return _ok(response)

        # DeepSeek → OpenRouter (no direct DeepSeek key; uses deepseek-specific parser)
        if "deepseek" in model_id.lower():
            print(f"[DeepSeek via OpenRouter] {model_id}")
            response = call_deepseek_api(
                prompt_content, context_messages, model_id, system_prompt,
                stream_callback=streaming_callback
            )
            if isinstance(response, dict) and "content" in response:
                response.update({"model": model, "role": "assistant", "ai_name": ai_name})
                return response
            return _ok(str(response) if response else "No response from model")

        # Fallback → OpenRouter (Qwen, Meta, Mistral paid, unknown providers)
        print(f"[OpenRouter Fallback] {model_id} — no direct key for this provider")
        response = call_openrouter_api(
            prompt_content, context_messages, model_id,
            system_prompt, stream_callback=streaming_callback, temperature=temperature
        )
        return _ok(response)
            
    except Exception as e:
        error_message = f"Error making API request: {str(e)}"
        print(f"Error: {error_message}")
        
        # Create an error response
        result = {
            "role": "system",
            "content": f"Error: {error_message}",
            "model": model,
            "ai_name": ai_name
        }
        
        # Return the error result
        return result

class ConversationManager:
    """Manages conversation processing and state"""
    def __init__(self, app):
        self.app = app
        self.workers = []  # Keep track of worker threads

        # Initialize AI command state dictionaries
        self.ai_prompt_additions = {}  # Store prompt additions from !prompt command (list per AI)
        self.ai_temperatures = {}  # Store custom temperatures from !temperature command

        # Initialize the worker thread pool
        self.thread_pool = QThreadPool()
        print(f"Conversation Manager initialized with {self.thread_pool.maxThreadCount()} threads")

        # Set up image update signals for thread-safe UI updates
        self.image_signals = ImageUpdateSignals()
        self.image_signals.image_ready.connect(self._on_image_ready)
        self.image_signals.image_failed.connect(self._on_image_failed)

        # Set up video update signals for thread-safe UI updates
        self.video_signals = VideoUpdateSignals()
        self.video_signals.video_ready.connect(self._on_video_ready)
        self.video_signals.video_failed.connect(self._on_video_failed)

        # Step mode state - tracks which AI is next in single-turn mode
        self.current_ai_index = 0  # 0-indexed: which AI's turn it is
        self.round_in_progress = False  # Whether we're mid-round in step mode

        # Auto-save timer - save every 30 seconds
        self._autosave_timer = QTimer()
        self._autosave_timer.timeout.connect(self._auto_save_conversation)
        self._autosave_timer.start(30000)  # 30 seconds
        from config import OUTPUTS_DIR
        self._autosave_path = os.path.join(OUTPUTS_DIR, '.autosave_conversation.json')

    def _auto_save_conversation(self):
        """Periodically auto-save conversation state for crash recovery."""
        try:
            if not hasattr(self.app, 'main_conversation') or not self.app.main_conversation:
                return

            # Only save if there's meaningful content (at least 2 messages)
            if len(self.app.main_conversation) < 2:
                return

            # Build saveable data (strip non-serializable content)
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'conversation': []
            }

            for msg in self.app.main_conversation:
                save_msg = {
                    'role': msg.get('role', ''),
                    'ai_name': msg.get('ai_name', ''),
                    'model': msg.get('model', ''),
                    '_type': msg.get('_type', ''),
                }

                # Handle content - skip binary image data
                content = msg.get('content', '')
                if isinstance(content, str):
                    save_msg['content'] = content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                    save_msg['content'] = '\n'.join(text_parts)
                else:
                    save_msg['content'] = str(content)

                # Skip empty/streaming messages
                if msg.get('_streaming') or not save_msg['content'].strip():
                    continue

                save_data['conversation'].append(save_msg)

            # Write atomically (write to temp, then rename)
            temp_path = self._autosave_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            # Rename to final path (atomic on most systems)
            import shutil
            shutil.move(temp_path, self._autosave_path)

        except Exception as e:
            print(f"[AutoSave] Error: {e}")

    def check_autosave_recovery(self):
        """Check for an auto-save file and offer to recover it."""
        try:
            if not os.path.exists(self._autosave_path):
                return False

            with open(self._autosave_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

            timestamp = save_data.get('timestamp', 'unknown time')
            msg_count = len(save_data.get('conversation', []))

            if msg_count < 2:
                # Not worth recovering
                os.remove(self._autosave_path)
                return False

            # Ask user if they want to recover
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self.app,
                "Recover Previous Session?",
                f"Found auto-saved conversation from {timestamp}\n"
                f"({msg_count} messages)\n\n"
                f"Would you like to recover it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Restore conversation
                self.app.main_conversation = save_data['conversation']
                self.app.left_pane.conversation = self.app.main_conversation
                self.app.left_pane.render_conversation()
                self.app.statusBar().showMessage(f"Recovered {msg_count} messages from auto-save")
                print(f"[AutoSave] Recovered {msg_count} messages from {timestamp}")
                return True
            else:
                # User declined - remove the file
                os.remove(self._autosave_path)
                return False

        except Exception as e:
            print(f"[AutoSave] Recovery error: {e}")
            return False

    def clear_autosave(self):
        """Clear the auto-save file (e.g., after clean exit or reset)."""
        try:
            if os.path.exists(self._autosave_path):
                os.remove(self._autosave_path)
        except Exception:
            pass

    def _on_video_ready(self, video_path: str, prompt: str, ai_name: str, model: str):
        """Handle video ready signal - runs on main thread"""
        try:
            # Remove the "generating..." notification first
            self._remove_pending_notification(ai_name, prompt)
            
            # Format display name like message headings do
            display_name = f"{ai_name} ({model})" if model else ai_name
            
            print(f"[Agent] Video ready, updating UI: {video_path}")
            # Update the video preview panel
            if hasattr(self.app, 'right_sidebar') and hasattr(self.app.right_sidebar, 'update_video_preview'):
                self.app.right_sidebar.update_video_preview(video_path, display_name, prompt)
            
            # Update status bar notification with prompt (truncated for display)
            if hasattr(self.app, 'notification_label'):
                # Truncate long prompts for status bar
                display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
                self.app.notification_label.setText(f"🎬 {display_name}: Video completed")
        except Exception as e:
            print(f"[Agent] Error handling video ready: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_video_failed(self, ai_name: str, prompt: str, error: str):
        """Handle video generation failure - runs on main thread"""
        try:
            # Remove the "generating..." notification first
            self._remove_pending_notification(ai_name, prompt)
            
            # Get AI's model name for consistent formatting
            ai_num = int(ai_name.split('-')[1]) if '-' in ai_name else 1
            model_name = self.get_model_for_ai(ai_num)
            
            # Create a failure notification message that AIs can see
            truncated_prompt = prompt[:50] + '...' if len(prompt) > 50 else prompt
            
            # Parse and simplify error message for display, but log the full error
            print(f"[Agent] ========== VIDEO GENERATION FAILED ==========")
            print(f"[Agent]   AI: {ai_name} ({model_name})")
            print(f"[Agent]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"[Agent]   Full error: {error}")
            print(f"[Agent] ==============================================")
            
            # Determine user-friendly error message based on error content
            error_lower = error.lower()
            if "402" in error or "credits" in error_lower or "insufficient" in error_lower:
                simple_error = "insufficient API credits"
                detail = "Check your OpenAI balance (Sora requires credits)"
            elif "429" in error or "rate" in error_lower or "limit" in error_lower:
                simple_error = "rate limited"
                detail = "Too many requests, please wait"
                print(f"[Agent]   >>> RATE LIMITED - Full response: {error}")
            elif "401" in error or "unauthorized" in error_lower or "api key" in error_lower:
                simple_error = "authentication failed"
                detail = "Check your OPENAI_API_KEY"
            elif "not set" in error_lower:
                simple_error = "API key not configured"
                detail = "Set OPENAI_API_KEY in .env file"
            elif "timeout" in error_lower:
                simple_error = "request timed out"
                detail = "Video generation took too long"
            elif "500" in error or "502" in error or "503" in error or "server" in error_lower:
                simple_error = "server error"
                detail = "OpenAI Sora is having issues"
            elif "content" in error_lower and "policy" in error_lower:
                simple_error = "content policy violation"
                detail = "Prompt was rejected by safety filters"
            elif "failed" in error_lower and "status" in error_lower:
                simple_error = "video rendering failed"
                detail = "Sora couldn't complete the video"
            else:
                simple_error = "generation failed"
                # Extract a short error snippet if available
                detail = error[:80] if len(error) <= 80 else error[:77] + "..."
            
            print(f"[Agent]   Simplified: {simple_error} — {detail}")
            
            failure_message = {
                "role": "system",
                "content": f"❌ [{caller}]: !video \"{truncated_prompt}\" — {simple_error}",
                "_type": "agent_notification",
                "_command_success": False
            }
            
            # Add to conversation so AIs can see it
            self.app.main_conversation.append(failure_message)
            self.app.left_pane.conversation = self.app.main_conversation
            self.app.left_pane.render_conversation()
            
            # Update status bar with more detail
            if hasattr(self.app, 'notification_label'):
                self.app.notification_label.setText(f"❌ Video failed: {simple_error} — {detail}")
                
            print(f"[Agent] Video failure notification added to conversation")
        except Exception as e:
            print(f"[Agent] Error handling video failure: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_image_failed(self, ai_name: str, prompt: str, error: str):
        """Handle image generation failure - runs on main thread"""
        try:
            # Remove the "generating..." notification first
            self._remove_pending_notification(ai_name, prompt)
            
            # Get AI's model name for consistent formatting
            ai_num = int(ai_name.split('-')[1]) if '-' in ai_name else 1
            model_name = self.get_model_for_ai(ai_num)
            
            # Create a failure notification message that AIs can see
            truncated_prompt = prompt[:50] + '...' if len(prompt) > 50 else prompt
            
            # Parse and simplify error message for display, but log the full error
            print(f"[Agent] ========== IMAGE GENERATION FAILED ==========")
            print(f"[Agent]   AI: {ai_name} ({model_name})")
            print(f"[Agent]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"[Agent]   Full error: {error}")
            print(f"[Agent] ==============================================")
            
            # Determine user-friendly error message based on error content
            error_lower = error.lower()
            if "402" in error or "credits" in error_lower or "insufficient" in error_lower:
                simple_error = "insufficient API credits"
                detail = "Check your OpenRouter balance"
            elif "429" in error or "rate" in error_lower or "limit" in error_lower:
                simple_error = "rate limited"
                detail = "Too many requests, please wait"
                print(f"[Agent]   >>> RATE LIMITED - Full response: {error}")
            elif "401" in error or "unauthorized" in error_lower or "api key" in error_lower:
                simple_error = "authentication failed"
                detail = "Check your OPENROUTER_API_KEY"
            elif "timeout" in error_lower:
                simple_error = "request timed out"
                detail = "Server took too long to respond"
            elif "500" in error or "502" in error or "503" in error or "server" in error_lower:
                simple_error = "server error"
                detail = "OpenRouter or model provider is having issues"
            elif "modalities" in error_lower or "not support" in error_lower:
                simple_error = "model doesn't support image generation"
                detail = "Try a different image model"
            elif "content" in error_lower and "policy" in error_lower:
                simple_error = "content policy violation"
                detail = "Prompt was rejected by safety filters"
            else:
                simple_error = "generation failed"
                # Extract a short error snippet if available
                detail = error[:80] if len(error) <= 80 else error[:77] + "..."
            
            print(f"[Agent]   Simplified: {simple_error} — {detail}")
            
            failure_message = {
                "role": "system",
                "content": f"❌ [{caller}]: !image \"{truncated_prompt}\" — {simple_error}",
                "_type": "agent_notification",
                "_command_success": False
            }
            
            # Add to conversation so AIs can see it
            self.app.main_conversation.append(failure_message)
            self.app.left_pane.conversation = self.app.main_conversation
            self.app.left_pane.render_conversation()
            
            # Update status bar with more detail
            if hasattr(self.app, 'notification_label'):
                self.app.notification_label.setText(f"❌ Image failed: {simple_error} — {detail}")
                
            print(f"[Agent] Image failure notification added to conversation")
        except Exception as e:
            print(f"[Agent] Error handling image failure: {e}")
            import traceback
            traceback.print_exc()
    
    def _remove_pending_notification(self, ai_name: str, prompt: str):
        """Remove the 'generating...' notification for a completed image/video"""
        prompt_key = f"{ai_name}:{prompt[:50]}"
        
        if not hasattr(self, '_pending_notifications'):
            return
        
        notification_id = self._pending_notifications.get(prompt_key)
        if not notification_id:
            print(f"[Agent] No pending notification found for {prompt_key[:40]}...")
            return
        
        # Get the correct conversation (main or branch)
        if self.app.active_branch:
            branch_id = self.app.active_branch
            if branch_id in self.app.branch_conversations:
                conversation = self.app.branch_conversations[branch_id]['conversation']
            else:
                conversation = self.app.main_conversation
        else:
            conversation = self.app.main_conversation
        
        # Remove in-place by finding and removing the matching message
        # This preserves the list reference!
        removed = False
        for i in range(len(conversation) - 1, -1, -1):  # Iterate backwards for safe removal
            if conversation[i].get('_notification_id') == notification_id:
                del conversation[i]
                removed = True
                break
        
        if removed:
            print(f"[Agent] Removed 'generating...' notification (ID: {notification_id})")
            # Ensure left_pane has the current reference
            self.app.left_pane.conversation = conversation
        
        # Clean up tracking dict
        del self._pending_notifications[prompt_key]
    
    def _on_image_ready(self, image_message: dict, image_path: str):
        """Handle image ready signal - runs on main thread"""
        try:
            # Remove the "generating..." notification first
            ai_name = image_message.get('ai_name', 'AI')
            model = image_message.get('model', '')
            prompt = image_message.get('_prompt', '')
            self._remove_pending_notification(ai_name, prompt)
            
            # Format display name like message headings do
            display_name = f"{ai_name} ({model})" if model else ai_name
            
            # Add image to the correct conversation (main or branch)
            if self.app.active_branch:
                branch_id = self.app.active_branch
                if branch_id in self.app.branch_conversations:
                    self.app.branch_conversations[branch_id]['conversation'].append(image_message)
                    self.app.left_pane.conversation = self.app.branch_conversations[branch_id]['conversation']
            else:
                self.app.main_conversation.append(image_message)
                self.app.left_pane.conversation = self.app.main_conversation
            
            # Update the conversation display
            self.app.left_pane.render_conversation()
            
            # Update the image preview panel
            if hasattr(self.app.right_sidebar, 'update_image_preview'):
                self.app.right_sidebar.update_image_preview(image_path, display_name, prompt)
            
            # Update status bar notification
            if hasattr(self.app, 'notification_label'):
                self.app.notification_label.setText(f"🖼️ {display_name} generated an image")
            
            print(f"[Agent] Image added to conversation context - other AIs can now see it")
        except Exception as e:
            print(f"[Agent] Error handling image ready: {e}")
            import traceback
            traceback.print_exc()
    
    def initialize(self):
        """Initialize the conversation manager"""
        # Initialize the app and thread pool
        print("Initializing conversation manager...")
        
        # Initialize branch conversations
        if not hasattr(self.app, 'branch_conversations'):
            self.app.branch_conversations = {}
        
        # Set up input callback
        self.app.left_pane.set_input_callback(self.process_input)
        
        # Set up branch processing callbacks
        self.app.left_pane.set_rabbithole_callback(self.rabbithole_callback)
        self.app.left_pane.set_fork_callback(self.fork_callback)
        
        # Initialize main conversation if not already set
        if not hasattr(self.app, 'main_conversation'):
            self.app.main_conversation = []
        
        # Display the initial empty conversation
        self.app.left_pane.display_conversation(self.app.main_conversation)
    
        print("Conversation manager initialized.")
    
    def _add_user_message(self, user_input):
        """Add user input to the conversation and update display. Returns True if message was added."""
        if not user_input:
            return False

        # Get the user's display name from control panel
        username = self.app.right_sidebar.control_panel.get_username()

        # Extract text for command parsing
        if isinstance(user_input, dict):
            raw_text = user_input.get('text', '')
            image_data = user_input.get('image')
        else:
            raw_text = user_input
            image_data = None

        # Parse commands from user text (same parser AIs use)
        cleaned_text = raw_text
        commands = []
        if raw_text:
            cleaned_text, commands = parse_commands(raw_text)

        # Build the user message with cleaned content
        if image_data:
            user_message = {
                "role": "user",
                "_user_name": username,
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_data['media_type'],
                            "data": image_data['base64']
                        }
                    }
                ]
            }
            if cleaned_text:
                user_message["content"].insert(0, {"type": "text", "text": cleaned_text})
        else:
            user_message = {"role": "user", "_user_name": username, "content": cleaned_text}

        # Only add the message if there's actual content (text or image)
        has_content = cleaned_text.strip() if cleaned_text else False
        if has_content or image_data:
            self.app.main_conversation.append(user_message)

        # Execute any commands the user typed
        if commands:
            print(f"[User] Found {len(commands)} command(s) in user input")
            for cmd in commands:
                success, message = self.execute_agent_command(cmd, username)
                print(f"[User] Command result: success={success}, message={message}")
                import uuid
                notification_id = str(uuid.uuid4())[:8]
                notification_msg = {
                    "role": "system",
                    "content": message,
                    "_type": "agent_notification",
                    "_command_success": success,
                    "_notification_id": notification_id
                }
                self.app.main_conversation.append(notification_msg)

        visible_conversation = [msg for msg in self.app.main_conversation if not msg.get('hidden', False)]
        self.app.left_pane.display_conversation(visible_conversation)
        self.update_conversation_html(self.app.main_conversation)
        return has_content or image_data or bool(commands)

    def process_input(self, user_input=None):
        """Process the user input and generate AI responses.

        In normal mode: creates workers for all AIs and chains them sequentially.
        In step mode: creates a single worker for the current AI only.
        """
        # Get the conversation (either main or branch)
        if self.app.active_branch:
            self.process_branch_input(user_input)
            return

        if not hasattr(self.app, 'main_conversation'):
            self.app.main_conversation = []

        # Check if step mode is enabled
        step_mode = self.app.right_sidebar.control_panel.is_step_mode()

        if step_mode:
            self._process_input_step_mode(user_input)
        else:
            self._process_input_normal(user_input)

    def _process_input_step_mode(self, user_input=None):
        """Step mode: run a single AI turn per propagate click."""
        has_user_content = self._add_user_message(user_input)

        num_ais = int(self.app.right_sidebar.control_panel.num_ais_selector.currentText())
        selected_prompt_pair = self.app.right_sidebar.control_panel.prompt_pair_selector.currentText()
        max_iterations = int(self.app.right_sidebar.control_panel.iterations_selector.currentText())

        # If user provides input or we're not in a round, reset to start of round
        if has_user_content and not self.round_in_progress:
            self.current_ai_index = 0
            self.round_in_progress = True
            self.app.turn_count = 0
        elif not self.round_in_progress:
            self.current_ai_index = 0
            self.round_in_progress = True

        # Check for next turn override from dropdown
        next_turn_selection = self.app.right_sidebar.control_panel.next_turn_selector.currentText()
        if next_turn_selection != "Auto":
            try:
                selected_ai_num = int(next_turn_selection.split("-")[1])
                if 1 <= selected_ai_num <= num_ais:
                    self.current_ai_index = selected_ai_num - 1
            except (ValueError, IndexError):
                pass
            self.app.right_sidebar.control_panel.next_turn_selector.setCurrentText("Auto")

        # Determine the AI for this turn
        ai_index = self.current_ai_index
        if ai_index >= num_ais:
            # Round complete — check if more iterations needed
            self._complete_step_round(max_iterations, num_ais)
            return

        ai_num = ai_index + 1
        ai_name = f"AI-{ai_num}"

        # Check muted
        muted_ais = getattr(self.app, 'muted_ais', set())
        if ai_name in muted_ais:
            print(f"[Step][Mute] {ai_name} is muted, skipping")
            mute_notification = {
                "role": "user",
                "content": f"[{ai_name} used !mute_self - listening this turn]",
                "_type": "agent_notification",
                "_command_success": None,
                "hidden": False
            }
            self.app.main_conversation.append(mute_notification)
            muted_ais.discard(ai_name)
            self.app.muted_ais = muted_ais
            # Advance and wait for next click
            self.current_ai_index += 1
            if self.current_ai_index >= num_ais:
                self._complete_step_round(max_iterations, num_ais)
            else:
                next_ai = f"AI-{self.current_ai_index + 1}"
                self.app.statusBar().showMessage(f"[Step] {ai_name} muted. Next: {next_ai} (click Propagate)")
            return

        # Start loading
        self.app.left_pane.start_loading()
        if hasattr(self.app, 'set_signal_active'):
            self.app.set_signal_active(True)
        self._request_start_time = time.time()

        # Update iteration counter
        self.app.update_iteration(self.app.turn_count + 1, max_iterations, ai_name)

        model = self.get_model_for_ai(ai_num)
        prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair][ai_name]
        invite_tier = self.app.right_sidebar.control_panel.get_ai_invite_tier()

        print(f"[Step] Running single turn: {ai_name} ({model}) — index {ai_index}/{num_ais}")

        worker = Worker(ai_name, self.app.main_conversation, model, prompt,
                        gui=self.app, invite_tier=invite_tier,
                        prompt_modifications=self.ai_prompt_additions,
                        ai_temperatures=self.ai_temperatures)
        worker.signals.started.connect(self.on_ai_started)
        worker.signals.response.connect(self.on_ai_response_received)
        worker.signals.result.connect(self.on_ai_result_received)
        worker.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker.signals.error.connect(self.on_ai_error)

        # Capture values for the lambda
        _max_iter = max_iterations
        _num_ais = num_ais
        worker.signals.finished.connect(lambda mi=_max_iter, na=_num_ais: self._on_step_turn_finished(mi, na))

        self.thread_pool.start(worker)

    def _on_step_turn_finished(self, max_iterations, _num_ais_ignored):
        """Called when a single AI finishes in step mode."""
        self.app.left_pane.stop_loading()

        # Re-read num_ais from UI (may have changed if !add_ai was used during this turn)
        num_ais = int(self.app.right_sidebar.control_panel.num_ais_selector.currentText())

        # Clear any pending AIs — they've already been added to the selector
        # so they'll be reached as part of the normal step sequence
        if hasattr(self, '_pending_ais') and self._pending_ais:
            print(f"[Step] Clearing {len(self._pending_ais)} pending AI(s) — already added to lineup")
            self._pending_ais = []

        # Advance to next AI
        self.current_ai_index += 1

        if self.current_ai_index >= num_ais:
            # Round complete
            self._complete_step_round(max_iterations, num_ais)
        else:
            # More AIs in this round — wait for next click
            next_ai = f"AI-{self.current_ai_index + 1}"
            self.app.statusBar().showMessage(f"[Step] Waiting — Next: {next_ai} (click Propagate)")
            if hasattr(self.app, 'set_signal_active'):
                self.app.set_signal_active(False)

    def _complete_step_round(self, max_iterations, num_ais):
        """Handle the end of a full round in step mode."""
        self.app.turn_count += 1
        self.current_ai_index = 0

        # Update HTML
        self.update_conversation_html(self.app.main_conversation)

        if self.app.turn_count < max_iterations:
            # More rounds to go — wait for next click
            next_ai = "AI-1"
            self.round_in_progress = True
            self.app.statusBar().showMessage(
                f"[Step] Round {self.app.turn_count}/{max_iterations} complete. Next: {next_ai} (click Propagate)")
            if hasattr(self.app, 'set_signal_active'):
                self.app.set_signal_active(False)
        else:
            # All iterations done
            self.round_in_progress = False
            self.app.statusBar().showMessage(
                f"[Step] Completed {max_iterations} round(s)")
            self.app.clear_iteration()
            if hasattr(self.app, 'set_signal_active'):
                self.app.set_signal_active(False)

    def _process_input_normal(self, user_input=None):
        """Normal mode: run all AIs sequentially in one propagate click."""
        self._add_user_message(user_input)

        # Get number of AIs from UI
        num_ais = int(self.app.right_sidebar.control_panel.num_ais_selector.currentText())

        # Get selected prompt pair
        selected_prompt_pair = self.app.right_sidebar.control_panel.prompt_pair_selector.currentText()

        # Start loading animation
        self.app.left_pane.start_loading()

        # Set signal indicator to active
        if hasattr(self.app, 'set_signal_active'):
            self.app.set_signal_active(True)

        # Track request start time for latency
        self._request_start_time = time.time()

        # Reset turn count ONLY if this is a new conversation or explicit user input
        max_iterations = int(self.app.right_sidebar.control_panel.iterations_selector.currentText())
        if user_input is not None or not self.app.main_conversation:
            self.app.turn_count = 0
            print(f"MAIN: Resetting turn count - starting new conversation with {max_iterations} iterations and {num_ais} AIs")
        else:
            print(f"MAIN: Continuing conversation - turn {self.app.turn_count+1} of {max_iterations}")

        # Update iteration counter in status bar
        self.app.update_iteration(self.app.turn_count + 1, max_iterations)

        # Create worker threads dynamically based on number of AIs
        workers = []

        # Check for muted AIs
        muted_ais = getattr(self.app, 'muted_ais', set())

        for i in range(1, num_ais + 1):
            ai_name = f"AI-{i}"

            # Skip muted AIs (they skip their next turn)
            if ai_name in muted_ais:
                print(f"[Mute] {ai_name} is muted, skipping this turn")
                # Add a notification to the conversation showing the command was used
                mute_notification = {
                    "role": "user",
                    "content": f"[{ai_name} used !mute_self - listening this turn]",
                    "_type": "agent_notification",
                    "_command_success": None,  # Info notification (not success/failure)
                    "hidden": False
                }
                self.app.main_conversation.append(mute_notification)
                # Remove from muted set (only skip one turn)
                muted_ais.discard(ai_name)
                continue

            model = self.get_model_for_ai(i)
            prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair][ai_name]

            # Get invite tier setting
            invite_tier = self.app.right_sidebar.control_panel.get_ai_invite_tier()

            worker = Worker(ai_name, self.app.main_conversation, model, prompt, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
            worker.signals.started.connect(self.on_ai_started)
            worker.signals.response.connect(self.on_ai_response_received)
            worker.signals.result.connect(self.on_ai_result_received)
            worker.signals.streaming_chunk.connect(self.on_streaming_chunk)
            worker.signals.error.connect(self.on_ai_error)

            workers.append(worker)

        # Update muted_ais set
        self.app.muted_ais = muted_ais

        # Handle case where all AIs are muted
        if not workers:
            print("[Mute] All AIs are muted this turn, proceeding to next iteration")
            self.app.left_pane.render_conversation()
            self.handle_turn_completion(max_iterations)
            return

        # Chain workers together AFTER all are created (avoids closure issues)
        for i, worker in enumerate(workers):
            if i < len(workers) - 1:
                # Not the last worker - connect to start next worker
                next_worker = workers[i + 1]
                ai_num = i + 2  # AI number for next worker (1-indexed, so i=0 means next is AI-2)
                # Use a factory function to properly capture values
                worker.signals.finished.connect(
                    self._make_next_turn_callback(next_worker, ai_num)
                )
            else:
                # Last worker - connect to handle turn completion
                max_iter = max_iterations  # Capture the value
                worker.signals.finished.connect(lambda mi=max_iter: self.handle_turn_completion(mi))

        # Start first AI's turn
        self.thread_pool.start(workers[0])
    
    def _make_next_turn_callback(self, worker, ai_number):
        """Factory function to create a callback for starting the next AI turn.
        This avoids closure issues with lambdas in loops."""
        def callback():
            self.start_next_ai_turn(worker, ai_number)
        return callback
    
    def start_next_ai_turn(self, worker, ai_number):
        """Start the next AI's turn in the conversation"""
        # Get the latest conversation state
        if self.app.active_branch:
            branch_id = self.app.active_branch
            branch_data = self.app.branch_conversations[branch_id]
            latest_conversation = branch_data['conversation']
        else:
            latest_conversation = self.app.main_conversation
        
        # Update worker's conversation reference to ensure it has the latest state
        worker.conversation = latest_conversation.copy()
        
        # Add a small delay between turns
        time.sleep(TURN_DELAY)
        
        # Start next AI's turn
        print(f"Starting AI-{ai_number}'s turn")
        self.thread_pool.start(worker)
    
    def handle_turn_completion(self, max_iterations=1):
        """Handle the completion of a full turn (both AIs)"""
        
        # Check for pending AIs that were added mid-round
        if hasattr(self, '_pending_ais') and self._pending_ais:
            pending = self._pending_ais.copy()
            self._pending_ais = []  # Clear the queue
            
            print(f"[Agent] Processing {len(pending)} pending AI(s) added during this round")
            for idx, p in enumerate(pending):
                print(f"[Agent]   Pending #{idx+1}: {p['ai_name']} -> {p['model']} (invited by {p.get('invited_by', 'unknown')})")
            
            # Get current conversation and prompt pair
            if self.app.active_branch:
                branch_id = self.app.active_branch
                branch_data = self.app.branch_conversations[branch_id]
                conversation = branch_data['conversation']
            else:
                conversation = self.app.main_conversation
            
            selected_prompt_pair = self.app.right_sidebar.control_panel.prompt_pair_selector.currentText()

            # num_ais selector was already updated when !add_ai fired

            # Build all workers first, then chain them properly
            pending_workers = []
            for pending_ai in pending:
                ai_name = pending_ai['ai_name']
                model = pending_ai['model']
                persona = pending_ai.get('persona')
                
                # Get prompt - use custom persona if provided, otherwise use default
                if persona:
                    prompt = f"You are {ai_name}. {persona}\n\nYou are interfacing with other AIs. Engage authentically."
                else:
                    prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair].get(ai_name, 
                        SYSTEM_PROMPT_PAIRS[selected_prompt_pair].get("AI-1", ""))
                
                print(f"[Agent] Creating worker for newly added {ai_name} ({model})")
                
                # Get invite tier setting
                invite_tier = self.app.right_sidebar.control_panel.get_ai_invite_tier()

                worker = Worker(ai_name, conversation.copy(), model, prompt, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
                worker.signals.started.connect(self.on_ai_started)
                worker.signals.response.connect(self.on_ai_response_received)
                worker.signals.result.connect(self.on_ai_result_received)
                worker.signals.streaming_chunk.connect(self.on_streaming_chunk)
                worker.signals.error.connect(self.on_ai_error)
                pending_workers.append(worker)
            
            # Store remaining workers for sequential processing
            print(f"[Agent] Created {len(pending_workers)} pending workers")
            for idx, w in enumerate(pending_workers):
                print(f"[Agent]   Worker #{idx+1}: {w.ai_name} -> {w.model}")
            
            if len(pending_workers) > 1:
                self._remaining_pending_workers = pending_workers[1:]
                print(f"[Agent] Queued {len(self._remaining_pending_workers)} workers for sequential processing")
                # First worker chains to process_next
                pending_workers[0].signals.finished.connect(self._process_next_pending_worker)
            else:
                # Only one pending worker - chain directly to finish
                self._remaining_pending_workers = []
                pending_workers[0].signals.finished.connect(
                    lambda mi=max_iterations: self._finish_turn_completion(mi)
                )
            
            # Store max_iterations for later use
            self._pending_max_iterations = max_iterations
            
            # Start first pending AI
            print(f"[Agent] Starting first pending worker: {pending_workers[0].ai_name} ({pending_workers[0].model})")
            self.thread_pool.start(pending_workers[0])
            
            return  # Exit - turn completion will be called after pending AIs finish
        
        self._finish_turn_completion(max_iterations)
    
    def _process_next_pending_worker(self):
        """Process the next pending worker in the queue."""
        print(f"[Agent] _process_next_pending_worker called, remaining: {len(getattr(self, '_remaining_pending_workers', []))}")
        if hasattr(self, '_remaining_pending_workers') and self._remaining_pending_workers:
            worker = self._remaining_pending_workers.pop(0)
            print(f"[Agent] Processing next pending worker: {worker.ai_name} ({worker.model})")
            print(f"[Agent]   Remaining after pop: {len(self._remaining_pending_workers)}")
            
            # Update conversation to latest state
            if self.app.active_branch:
                branch_id = self.app.active_branch
                branch_data = self.app.branch_conversations[branch_id]
                worker.conversation = branch_data['conversation'].copy()
            else:
                worker.conversation = self.app.main_conversation.copy()
            
            # If more workers remain, chain to this function again
            if self._remaining_pending_workers:
                print(f"[Agent]   More workers remain, will chain to next")
                worker.signals.finished.connect(self._process_next_pending_worker)
            else:
                # Last one - finish turn completion
                print(f"[Agent]   This is the last pending worker")
                max_iterations = getattr(self, '_pending_max_iterations', 
                    int(self.app.right_sidebar.control_panel.iterations_selector.currentText()))
                worker.signals.finished.connect(lambda mi=max_iterations: self._finish_turn_completion(mi))
            
            time.sleep(TURN_DELAY)
            print(f"[Agent] Starting worker: {worker.ai_name}")
            self.thread_pool.start(worker)
        else:
            # No more pending workers, finish turn
            print(f"[Agent] No remaining pending workers, finishing turn")
            max_iterations = getattr(self, '_pending_max_iterations',
                int(self.app.right_sidebar.control_panel.iterations_selector.currentText()))
            self._finish_turn_completion(max_iterations)
    
    def _finish_turn_completion(self, max_iterations=1):
        """Complete the turn after all AIs (including pending) have finished."""
        # Stop the loading animation
        self.app.left_pane.stop_loading()
        
        # Increment turn count
        self.app.turn_count += 1
        
        # Check which conversation we're dealing with (main or branch)
        if self.app.active_branch:
            # Branch conversation
            branch_id = self.app.active_branch
            branch_data = self.app.branch_conversations[branch_id]
            conversation = branch_data['conversation']
            
            print(f"BRANCH: Turn {self.app.turn_count} of {max_iterations} completed")
            
            # Update the full conversation HTML
            self.update_conversation_html(conversation)
            
            # Check if we should start another turn
            if self.app.turn_count < max_iterations:
                print(f"BRANCH: Starting turn {self.app.turn_count + 1} of {max_iterations}")
                
                # Process through branch_input but with no user input to continue the conversation
                self.process_branch_input(None)  # None = no user input, just continue
            else:
                print(f"BRANCH: All {max_iterations} turns completed")
                self.app.statusBar().showMessage(f"Completed {max_iterations} turns")
                self.app.clear_iteration()  # Clear iteration counter
                # Set signal indicator to idle
                if hasattr(self.app, 'set_signal_active'):
                    self.app.set_signal_active(False)
        else:
            # Main conversation
            print(f"MAIN: Turn {self.app.turn_count} of {max_iterations} completed")
            
            # Update the full conversation HTML
            self.update_conversation_html(self.app.main_conversation)
            
            # Check if we should start another turn
            if self.app.turn_count < max_iterations:
                print(f"MAIN: Starting turn {self.app.turn_count + 1} of {max_iterations}")
                # Call process_input with no user input to continue the conversation
                self.process_input(None)  # None = no user input, just continue
            else:
                print(f"MAIN: All {max_iterations} turns completed")
                self.app.statusBar().showMessage(f"Completed {max_iterations} turns")
                self.app.clear_iteration()  # Clear iteration counter
                # Set signal indicator to idle
                if hasattr(self.app, 'set_signal_active'):
                    self.app.set_signal_active(False)
    
    def handle_progress(self, message):
        """Handle progress update from worker"""
        print(message)
        self.app.statusBar().showMessage(message)
    
    def handle_error(self, error_message):
        """Handle error from worker"""
        print(f"Error: {error_message}")
        self.app.left_pane.append_text(f"\nError: {error_message}\n", "system")
        self.app.statusBar().showMessage(f"Error: {error_message}")
        # Set signal indicator to idle on error
        if hasattr(self.app, 'set_signal_active'):
            self.app.set_signal_active(False)
    
    def process_branch_input(self, user_input=None):
        """Process input from the user specifically for branch conversations"""
        # Check if we have an active branch
        if not self.app.active_branch:
            # Fallback to main conversation if no active branch
            self.process_input(user_input)
            return
            
        # Get branch data
        branch_id = self.app.active_branch
        branch_data = self.app.branch_conversations[branch_id]
        conversation = branch_data['conversation']
        branch_type = branch_data.get('type', 'branch')
        selected_text = branch_data.get('selected_text', '')
        
        # Check for duplicate messages first
        if len(conversation) >= 2:
            # Check the last two messages
            last_msg = conversation[-1] if conversation else None
            second_last_msg = conversation[-2] if len(conversation) > 1 else None
            
            # If the last two messages are identical (same content), remove the duplicate
            if (last_msg and second_last_msg and 
                last_msg.get('content') == second_last_msg.get('content')):
                # Remove the duplicate message
                conversation.pop()
                print("Removed duplicate message from branch conversation")
        
        # Add user input if provided
        if user_input:
            # Handle both string and dict input (dict for image support)
            if isinstance(user_input, dict):
                # Extract text and image data
                text = user_input.get('text', '')
                image_data = user_input.get('image')
                
                if image_data:
                    # Create message with image
                    user_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_data['media_type'],
                                    "data": image_data['base64']
                                }
                            }
                        ]
                    }
                    # Add text if provided
                    if text:
                        user_message["content"].insert(0, {
                            "type": "text",
                            "text": text
                        })
                else:
                    # Text-only message
                    user_message = {
                        "role": "user",
                        "content": text
                    }
            else:
                # Legacy string input
                user_message = {
                    "role": "user",
                    "content": user_input
                }
                
            conversation.append(user_message)
            
            # Update the conversation display with the new user message
            visible_conversation = [msg for msg in conversation if not msg.get('hidden', False)]
            self.app.left_pane.display_conversation(visible_conversation, branch_data)
            
            # Update the HTML conversation document for the branch
            self.update_conversation_html(conversation)
        
        # Get selected models and prompt pair from UI
        ai_1_model = self.app.right_sidebar.control_panel.ai1_model_selector.currentText()
        ai_2_model = self.app.right_sidebar.control_panel.ai2_model_selector.currentText()
        ai_3_model = self.app.right_sidebar.control_panel.ai3_model_selector.currentText()
        selected_prompt_pair = self.app.right_sidebar.control_panel.prompt_pair_selector.currentText()
        
        # Check if we've already had AI responses in this branch
        has_ai_responses = False
        ai_response_count = 0
        for msg in conversation:
            if msg.get('role') == 'assistant':
                has_ai_responses = True
                ai_response_count += 1
        
        # Determine which prompts to use based on branch type and response history
        if branch_type.lower() == 'rabbithole' and ai_response_count < 2:
            # Initial rabbitholing prompt - only for the first exchange
            print("Using rabbithole-specific prompt for initial exploration")
            rabbithole_prompt = f"You are interacting with other AIs. IMPORTANT: Focus this response specifically on exploring and expanding upon the concept of '{selected_text}' in depth. Discuss the most interesting aspects or connections related to this concept while maintaining the tone of the conversation. No numbered lists or headings."
            ai_1_prompt = rabbithole_prompt
            ai_2_prompt = rabbithole_prompt
            ai_3_prompt = rabbithole_prompt
        else:
            # After initial exploration, revert to standard prompts
            print("Using standard prompts for continued conversation")
            ai_1_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI-1"]
            ai_2_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI-2"]
            ai_3_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI-3"]
        
        # Start loading animation
        self.app.left_pane.start_loading()
        
        # Reset turn count ONLY if this is a new conversation or explicit user input
        # Don't reset during automatic iterations
        if user_input is not None or not has_ai_responses:
            self.app.turn_count = 0
            print("Resetting turn count - starting new conversation")
        
        # Get max iterations
        max_iterations = int(self.app.right_sidebar.control_panel.iterations_selector.currentText())
        
        # Get invite tier setting
        invite_tier = self.app.right_sidebar.control_panel.get_ai_invite_tier()
        
        # Create worker threads for AI-1, AI-2, and AI-3
        worker1 = Worker("AI-1", conversation, ai_1_model, ai_1_prompt, is_branch=True, branch_id=branch_id, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
        worker2 = Worker("AI-2", conversation, ai_2_model, ai_2_prompt, is_branch=True, branch_id=branch_id, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
        worker3 = Worker("AI-3", conversation, ai_3_model, ai_3_prompt, is_branch=True, branch_id=branch_id, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
        
        # Connect signals for worker1
        worker1.signals.started.connect(self.on_ai_started)
        worker1.signals.response.connect(self.on_ai_response_received)
        worker1.signals.result.connect(self.on_ai_result_received)
        worker1.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker1.signals.finished.connect(lambda: self.start_ai2_turn(conversation, worker2))
        worker1.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker2
        worker2.signals.started.connect(self.on_ai_started)
        worker2.signals.response.connect(self.on_ai_response_received)
        worker2.signals.result.connect(self.on_ai_result_received)
        worker2.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker2.signals.finished.connect(lambda: self.start_ai3_turn(conversation, worker3))
        worker2.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker3
        worker3.signals.started.connect(self.on_ai_started)
        worker3.signals.response.connect(self.on_ai_response_received)
        worker3.signals.result.connect(self.on_ai_result_received)
        worker3.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker3.signals.finished.connect(lambda: self.handle_turn_completion(max_iterations))
        worker3.signals.error.connect(self.on_ai_error)
        
        # Start AI-1's turn
        self.thread_pool.start(worker1)
        
    def on_streaming_chunk(self, ai_name, chunk):
        """Handle streaming chunks as they arrive"""
        # Initialize streaming buffer if not exists
        if not hasattr(self, '_streaming_buffers'):
            self._streaming_buffers = {}
        
        # Initialize buffer for this AI if needed (first chunk)
        is_first_chunk = ai_name not in self._streaming_buffers
        
        if is_first_chunk:
            self._streaming_buffers[ai_name] = ""
            
            # Remove typing indicator when first chunk arrives - AI is now "speaking" not "thinking"
            self._remove_typing_indicator(ai_name)
            
            # CRITICAL: Create a placeholder message in the conversation BEFORE rendering
            # This ensures a widget gets created that we can then append to
            ai_number = int(ai_name.split('-')[1]) if '-' in ai_name else 1
            model_name = self.get_model_for_ai(ai_number)
            
            # Track the placeholder message so we can update it
            if not hasattr(self, '_streaming_messages'):
                self._streaming_messages = {}
            
            placeholder_msg = {
                "role": "assistant",
                "content": "",  # Start empty, will be filled by streaming
                "ai_name": ai_name,
                "model": model_name,
                "_streaming": True  # Mark as streaming so we know to update it
            }
            self._streaming_messages[ai_name] = placeholder_msg
            
            # Add to the correct conversation
            if self.app.active_branch:
                branch_id = self.app.active_branch
                if branch_id in self.app.branch_conversations:
                    self.app.branch_conversations[branch_id]['conversation'].append(placeholder_msg)
                    self.app.left_pane.conversation = self.app.branch_conversations[branch_id]['conversation']
            else:
                if not hasattr(self.app, 'main_conversation'):
                    self.app.main_conversation = []
                self.app.main_conversation.append(placeholder_msg)
                self.app.left_pane.conversation = self.app.main_conversation
            
            # Now render - this creates a widget for our placeholder
            self.app.left_pane.render_conversation()
            
            # Calculate and update latency on first chunk
            if hasattr(self, '_request_start_time') and hasattr(self.app, 'update_signal_latency'):
                latency_ms = int((time.time() - self._request_start_time) * 1000)
                self.app.update_signal_latency(latency_ms)
        
        # Append chunk to buffer
        self._streaming_buffers[ai_name] += chunk

        # Track typing speed
        if not hasattr(self, '_streaming_start_times'):
            self._streaming_start_times = {}
        if ai_name not in self._streaming_start_times:
            self._streaming_start_times[ai_name] = time.time()

        elapsed = time.time() - self._streaming_start_times[ai_name]
        if elapsed > 0.5:  # Update speed after 0.5s to get stable reading
            chars_per_sec = len(self._streaming_buffers[ai_name]) / elapsed
            if hasattr(self.app, 'statusBar'):
                self.app.statusBar().showMessage(f"{ai_name}: {int(chars_per_sec)} chars/s")
        
        # Update the placeholder message content in the conversation data
        if hasattr(self, '_streaming_messages') and ai_name in self._streaming_messages:
            self._streaming_messages[ai_name]["content"] = self._streaming_buffers[ai_name]
        
        # CRITICAL: Directly update the specific widget for this AI
        # Do NOT call render_conversation() - that causes cross-contamination when multiple AIs stream
        self.app.left_pane.update_streaming_widget(ai_name, self._streaming_buffers[ai_name])
    
    def on_ai_started(self, ai_name, model):
        """Handle AI starting to process - update status"""
        print(f"[Typing] {ai_name} ({model}) started processing")
        
        # Update iteration counter with current AI
        max_iterations = int(self.app.right_sidebar.control_panel.iterations_selector.currentText())
        current_turn = getattr(self.app, 'turn_count', 0) + 1
        self.app.update_iteration(current_turn, max_iterations, ai_name)
        
        # Extract AI number for styling
        ai_number = int(ai_name.split('-')[1]) if '-' in ai_name else 1
        
        # Typing indicator disabled - simplifies rendering flow
        # Status bar iteration counter now shows which AI is responding
        return
        
        # --- OLD TYPING INDICATOR CODE (disabled) ---
        # Create typing indicator message
        typing_message = {
            "role": "assistant",
            "content": "",  # Empty content - the render function will show the animation
            "ai_name": ai_name,
            "model": model,
            "_type": "typing_indicator",
            "_ai_number": ai_number
        }
        
        # Store reference for removal later
        if not hasattr(self, '_typing_indicators'):
            self._typing_indicators = {}
        self._typing_indicators[ai_name] = typing_message
        
        # No animation timer needed - using static "thinking..." text
        
        # Add to conversation and render
        if self.app.active_branch:
            branch_id = self.app.active_branch
            if branch_id in self.app.branch_conversations:
                self.app.branch_conversations[branch_id]['conversation'].append(typing_message)
                self.app.left_pane.conversation = self.app.branch_conversations[branch_id]['conversation']
        else:
            if not hasattr(self.app, 'main_conversation'):
                self.app.main_conversation = []
            self.app.main_conversation.append(typing_message)
            self.app.left_pane.conversation = self.app.main_conversation
        
        self.app.left_pane.render_conversation()
    
    def _update_typing_animation(self):
        """Update the typing animation dots - DISABLED, using static text"""
        # Timer should not be running, but stop it if it is
        if hasattr(self, '_typing_animation_timer') and self._typing_animation_timer.isActive():
            self._typing_animation_timer.stop()
    
    def _remove_typing_indicator(self, ai_name):
        """Remove typing indicator for an AI"""
        if not hasattr(self, '_typing_indicators') or ai_name not in self._typing_indicators:
            return
        
        typing_message = self._typing_indicators[ai_name]
        del self._typing_indicators[ai_name]
        
        # Remove from conversation
        if self.app.active_branch:
            branch_id = self.app.active_branch
            if branch_id in self.app.branch_conversations:
                conv = self.app.branch_conversations[branch_id]['conversation']
                if typing_message in conv:
                    conv.remove(typing_message)
        else:
            if hasattr(self.app, 'main_conversation') and typing_message in self.app.main_conversation:
                self.app.main_conversation.remove(typing_message)
    
    def _clear_all_typing_indicators(self):
        """Remove all typing indicators from conversation"""
        if not hasattr(self, '_typing_indicators'):
            return
        
        # Copy keys since we're modifying the dict
        ai_names = list(self._typing_indicators.keys())
        for ai_name in ai_names:
            self._remove_typing_indicator(ai_name)
    
    def on_ai_response_received(self, ai_name, response_content):
        """Handle AI responses for both main and branch conversations"""
        print(f"Response received from {ai_name}: {response_content[:100]}...")
        
        # Remove typing indicator for this AI
        self._remove_typing_indicator(ai_name)
        
        # Check if we have a streaming placeholder for this AI
        has_streaming_placeholder = (
            hasattr(self, '_streaming_messages') and 
            ai_name in self._streaming_messages
        )
        
        # Get streaming tracking data BEFORE clearing
        if hasattr(self, '_streaming_buffers') and ai_name in self._streaming_buffers:
            del self._streaming_buffers[ai_name]
        if hasattr(self, '_streaming_start_times') and ai_name in self._streaming_start_times:
            del self._streaming_start_times[ai_name]
        if hasattr(self, '_streaming_messages') and ai_name in self._streaming_messages:
            streaming_msg = self._streaming_messages[ai_name]
            del self._streaming_messages[ai_name]
        else:
            streaming_msg = None
        
        # Parse response for agentic commands
        cleaned_content, commands = parse_commands(response_content)
        
        # Extract AI number for model lookup
        ai_number = int(ai_name.split('-')[1]) if '-' in ai_name else 1
        
        # CRITICAL: Update streaming message content FIRST, before any notifications
        # This ensures the widget shows cleaned content during notification renders
        if has_streaming_placeholder and streaming_msg:
            streaming_msg["content"] = cleaned_content
            streaming_msg["model"] = self.get_model_for_ai(ai_number)
            # Update widget directly so it shows cleaned content
            self.app.left_pane.update_streaming_widget(ai_name, cleaned_content)
            # Remove streaming flag now that content is finalized
            if "_streaming" in streaming_msg:
                del streaming_msg["_streaming"]
        
        # Now execute commands and add notifications
        if commands:
            print(f"[Agent] Found {len(commands)} command(s) in {ai_name}'s response")
            
            for cmd in commands:
                success, message = self.execute_agent_command(cmd, ai_name)
                print(f"[Agent] Command result: success={success}, message={message}")
                
                # Add notification as a system message in the conversation
                import uuid
                notification_id = str(uuid.uuid4())[:8]
                notification_msg = {
                    "role": "system",
                    "content": message,
                    "_type": "agent_notification",
                    "_command_success": success,
                    "_notification_id": notification_id
                }
                
                # For in-progress notifications (success=None), store ID for later removal
                if success is None and "(generating...)" in message:
                    if not hasattr(self, '_pending_notifications'):
                        self._pending_notifications = {}
                    prompt_key = cmd.params.get('prompt', '')[:50] if cmd.params else ''
                    self._pending_notifications[f"{ai_name}:{prompt_key}"] = notification_id
                    print(f"[Agent] Stored pending notification ID: {notification_id} for {ai_name}:{prompt_key[:30]}...")
                
                # Add to the correct conversation (no render yet - batch it)
                if self.app.active_branch:
                    branch_id = self.app.active_branch
                    if branch_id in self.app.branch_conversations:
                        self.app.branch_conversations[branch_id]['conversation'].append(notification_msg)
                        print(f"[Agent] Added notification to branch conversation")
                else:
                    if not hasattr(self.app, 'main_conversation'):
                        self.app.main_conversation = []
                    self.app.main_conversation.append(notification_msg)
                    print(f"[Agent] Added notification to main conversation, total messages: {len(self.app.main_conversation)}")
                
                # Update status bar with the notification
                if hasattr(self.app, 'notification_label'):
                    self.app.notification_label.setText(message)
        
        # Use cleaned content (commands stripped out) for the conversation
        response_content = cleaned_content
        
        # If the AI only sent commands with no other text, handle appropriately
        if not response_content or not response_content.strip():
            print(f"[Agent] {ai_name}'s response was only commands, skipping empty message")
            # If there was a streaming placeholder, we need to remove it
            # since cleaned_content is empty (was only commands)
            if has_streaming_placeholder and streaming_msg:
                # Remove the streaming placeholder from conversation
                if self.app.active_branch:
                    branch_id = self.app.active_branch
                    if branch_id in self.app.branch_conversations:
                        conv = self.app.branch_conversations[branch_id]['conversation']
                        if streaming_msg in conv:
                            conv.remove(streaming_msg)
                        self.app.left_pane.conversation = conv
                else:
                    if streaming_msg in self.app.main_conversation:
                        self.app.main_conversation.remove(streaming_msg)
                    self.app.left_pane.conversation = self.app.main_conversation
            # Do final render to show notifications (if any)
            self._final_render_after_response()
            return
        
        # If there was a streaming placeholder, content was already updated above
        # Just need to do the final render
        if has_streaming_placeholder and streaming_msg:
            self._final_render_after_response()
        else:
            # No streaming placeholder - add message normally
            ai_message = {
                "role": "assistant",
                "content": response_content,
                "ai_name": ai_name,
                "model": self.get_model_for_ai(ai_number)
            }
            
            # Add to conversation
            if self.app.active_branch:
                branch_id = self.app.active_branch
                if branch_id in self.app.branch_conversations:
                    self.app.branch_conversations[branch_id]['conversation'].append(ai_message)
            else:
                if not hasattr(self.app, 'main_conversation'):
                    self.app.main_conversation = []
                self.app.main_conversation.append(ai_message)
            
            self._final_render_after_response()
        
        # Update status bar
        self.app.statusBar().showMessage(f"Received response from {ai_name}")
    
    def _final_render_after_response(self):
        """Do a final render after processing an AI response.
        
        Uses immediate render to prevent race conditions with other streaming AIs.
        """
        if self.app.active_branch:
            branch_id = self.app.active_branch
            if branch_id in self.app.branch_conversations:
                branch_data = self.app.branch_conversations[branch_id]
                visible = [msg for msg in branch_data['conversation'] if not msg.get('hidden', False)]
                self.app.left_pane.conversation = visible
                self.app.left_pane.render_conversation(immediate=True)
        else:
            visible = [msg for msg in self.app.main_conversation if not msg.get('hidden', False)]
            self.app.left_pane.conversation = visible
            self.app.left_pane.render_conversation(immediate=True)
        
    def on_ai_result_received(self, ai_name, result):
        """Handle the complete AI result - for non-display tasks only.
        
        NOTE: Message display is handled by on_ai_response_received.
        This handler is only for side effects like auto-image generation, Sora, etc.
        """
        print(f"Result received from {ai_name}")
        
        # Determine which conversation to update
        conversation = self.app.main_conversation
        if self.app.active_branch:
            branch_id = self.app.active_branch
            branch_data = self.app.branch_conversations[branch_id]
            conversation = branch_data['conversation']
        
        # Generate an image based on the AI response (for non-image responses) if auto-generation is enabled
        if isinstance(result, dict) and "content" in result and not "image_url" in result:
            response_content = result.get("content", "")
            if response_content and len(response_content.strip()) > 20:
                if hasattr(self.app.right_sidebar.control_panel, 'auto_image_checkbox') and self.app.right_sidebar.control_panel.auto_image_checkbox.isChecked():
                    self.app.left_pane.append_text("\nGenerating an image based on this response...\n", "system")
                    self.generate_and_display_image(response_content, ai_name)
        
        # NOTE: Content display removed - handled by on_ai_response_received
        # The old code was appending headers and content here, causing duplication

        # Optionally trigger Sora video generation from AI-1 responses (no GUI embedding)
        try:
            auto_sora = os.getenv("SORA_AUTO_FROM_AI1", "0").strip() == "1"
            if auto_sora and ai_name == "AI-1" and isinstance(result, dict):
                prompt_text = result.get("content", "")
                # Require a minimally substantive prompt
                if isinstance(prompt_text, str) and len(prompt_text.strip()) > 20:
                    # Inform user in the UI synchronously (short message)
                    self.app.left_pane.append_text("\n[system] Starting Sora video job from AI-1 response...\n", "system")

                    # Use config values with env var override
                    from config import SORA_SECONDS, SORA_SIZE
                    sora_model = os.getenv("SORA_MODEL", "sora-2")
                    sora_seconds = int(os.getenv("SORA_SECONDS", str(SORA_SECONDS)))
                    sora_size = os.getenv("SORA_SIZE", SORA_SIZE) or None

                    # Run in background to avoid blocking UI
                    import threading
                    def _run_sora_job(prompt_capture: str):
                        result_dict = generate_video_with_sora(
                            prompt=prompt_capture,
                            model=sora_model,
                            seconds=sora_seconds,
                            size=sora_size,
                            poll_interval_seconds=5.0,
                        )
                        # Log to console; UI updates from background threads are avoided
                        if result_dict.get("success"):
                            print(f"Sora video completed: {result_dict.get('video_path')}")
                        else:
                            print(f"Sora video failed: {result_dict.get('error')}")

                    threading.Thread(target=_run_sora_job, args=(prompt_text,), daemon=True).start()
        except Exception as e:
            print(f"Auto Sora trigger error: {e}")
        
        # NOTE: display_conversation removed - handled by on_ai_response_received
            
    def generate_and_display_image(self, text, ai_name):
        """Generate an image based on text and display it in the UI"""
        # Create a prompt for the image generation
        # Extract the first 100-300 characters to use as the image prompt
        max_length = min(300, len(text))
        prompt = text[:max_length].strip()
        
        # Add artistic direction to the prompt using the user's requested format
        enhanced_prompt = f"You are the artist/chronicler of an exchange between multiple AIs. Create an image using the following ai text contribution as inspiration. DO NOT merely repeat text in the image. Interpret the text in image form.{prompt}"
        
        # Generate the image
        result = generate_image_from_text(enhanced_prompt)
        
        if result["success"]:
            # Display the image in the UI
            image_path = result["image_path"]
            
            # Find the corresponding message in the conversation and add the image path
            conversation = self.app.main_conversation
            if self.app.active_branch:
                branch_id = self.app.active_branch
                branch_data = self.app.branch_conversations[branch_id]
                conversation = branch_data['conversation']
            
            # Find the most recent message from this AI
            for msg in reversed(conversation):
                if msg.get("ai_name") == ai_name and msg.get("role") == "assistant":
                    # Add the image path and model to the message
                    msg["generated_image_path"] = image_path
                    msg["image_model"] = result.get("model", "unknown")
                    print(f"Added generated image {image_path} to message from {ai_name}")
                    break
            
            # Update the conversation HTML to include the new image
            self.update_conversation_html(conversation)
            
            # Run on the main thread
            self.app.left_pane.display_image(image_path)
            
            # Notify the user
            self.app.left_pane.append_text(f"\n✓ Generated image saved to {image_path}\n", "system")
        else:
            # Notify the user of the failure
            error_msg = result.get("error", "Unknown error")
            print(f"Image generation failed: {error_msg}")
            self.app.left_pane.append_text(f"\n✗ Image generation failed: {error_msg}\n", "system")
            
            # Do not automatically open the HTML view
            # open_html_in_browser("conversation_full.html")
    
    def _is_ai_name(self, name: str) -> bool:
        """Check if name matches AI-N pattern."""
        import re
        return bool(re.match(r'^AI-\d+$', name, re.IGNORECASE))

    def _format_caller(self, name: str) -> str:
        """Format a caller name for notification messages.
        AIs show as 'AI-1 (model)', humans show as just their name."""
        if self._is_ai_name(name):
            ai_num = int(name.split('-')[1])
            model = self.get_model_for_ai(ai_num)
            return f"{name} ({model})"
        return name

    def execute_agent_command(self, command: AgentCommand, ai_name: str) -> tuple[bool, str]:
        """
        Execute an agentic command from an AI or human user.

        Args:
            command: The parsed AgentCommand to execute
            ai_name: The AI or human username that issued the command

        Returns:
            tuple: (success: bool, message: str)
        """
        action = command.action
        params = command.params
        
        print(f"[Agent] Executing command: {action} from {ai_name}")
        print(f"[Agent] Params: {params}")
        
        if action == 'image':
            return self._execute_image_command(params.get('prompt', ''), ai_name)
        elif action == 'video':
            return self._execute_video_command(params.get('prompt', ''), ai_name)
        elif action == 'add_ai':
            return self._execute_add_ai_command(params.get('model', ''), params.get('persona'), ai_name)
        elif action == 'remove_ai':
            return self._execute_remove_ai_command(params.get('target', ''), ai_name)
        elif action == 'list_models':
            return self._execute_list_models_command(ai_name)
        elif action == 'mute_self':
            return self._execute_mute_command(ai_name)
        elif action == 'search':
            return self._execute_search_command(params.get('query', ''), ai_name)
        elif action == 'prompt':
            return self._execute_prompt_command(params.get('text', ''), ai_name)
        elif action == 'temperature':
            return self._execute_temperature_command(params.get('value'), ai_name)
        elif action == 'vote':
            return self._execute_vote_command(params.get('question', ''), params.get('options'), ai_name)
        elif action == 'whisper':
            return self._execute_whisper_command(params.get('target', ''), params.get('message', ''), ai_name)
        else:
            caller = self._format_caller(ai_name)
            return False, f"❌ [{caller}]: !{action} — unknown command"
    
    def _execute_image_command(self, prompt: str, ai_name: str, model_name: str = None) -> tuple[bool, str]:
        """Execute an image generation command."""
        caller = self._format_caller(ai_name)
        if not model_name and self._is_ai_name(ai_name):
            ai_number = int(ai_name.split('-')[1])
            model_name = self.get_model_for_ai(ai_number)
        elif not model_name:
            model_name = ai_name
        
        if not prompt or len(prompt.strip()) < 5:
            return False, f"❌ [{caller}]: !image — prompt too short"
        
        print(f"[Agent] Generating image for {ai_name} ({model_name}): {prompt[:100]}...")
        
        # Run image generation in background thread to avoid blocking UI
        import threading
        
        def _run_image_job():
            try:
                # Add artistic context to the prompt
                enhanced_prompt = f"Create an image inspired by the following description from an AI conversation: {prompt}"
                
                print(f"[Agent] Starting image generation...")
                result = generate_image_from_text(enhanced_prompt)
                
                if result.get('success'):
                    image_path = result['image_path']
                    print(f"[Agent] Image generated successfully: {image_path}")
                    
                    # Convert image to base64 so other AIs can see it
                    import base64
                    try:
                        with open(image_path, 'rb') as img_file:
                            image_bytes = img_file.read()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # Determine media type from file header bytes, not extension
                        # JPEG starts with FF D8 FF, PNG starts with 89 50 4E 47
                        if image_bytes[:3] == b'\xff\xd8\xff':
                            media_type = "image/jpeg"
                        elif image_bytes[:4] == b'\x89PNG':
                            media_type = "image/png"
                        elif image_bytes[:4] == b'GIF8':
                            media_type = "image/gif"
                        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
                            media_type = "image/webp"
                        else:
                            # Fallback to extension
                            media_type = "image/png" if image_path.endswith('.png') else "image/jpeg"
                        print(f"[Agent] Detected image media type: {media_type}")
                        
                        # Create image message for conversation context
                        # Keep the !image command visible so AIs remember the syntax
                        image_message = {
                            "role": "user",  # Present as user message so AIs see it in their context
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"[{caller}]: !image \"{prompt}\""
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_base64
                                    }
                                }
                            ],
                            "generated_image_path": image_path,
                            "image_model": result.get("model", "unknown"),
                            "_type": "generated_image",
                            "_prompt": prompt,  # Store prompt for notification display
                            "ai_name": ai_name,
                            "model": model_name
                        }
                        
                        print(f"[Agent] Image message created with _prompt: '{prompt[:50]}...'")
                        
                        # Emit signal to update UI on main thread
                        self.image_signals.image_ready.emit(image_message, image_path)
                        
                    except Exception as e:
                        print(f"[Agent] Could not add image to context: {e}")
                        self.image_signals.image_failed.emit(ai_name, prompt, str(e))
                        import traceback
                        traceback.print_exc()
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"[Agent] Image generation failed: {error}")
                    # Emit failure signal so UI can show error to user and AIs
                    self.image_signals.image_failed.emit(ai_name, prompt, error)
            except Exception as e:
                print(f"[Agent] Image generation exception: {e}")
                self.image_signals.image_failed.emit(ai_name, prompt, str(e))
        
        threading.Thread(target=_run_image_job, daemon=True).start()
        # Return None for _command_success to show yellow "in progress" color (not green success)
        return None, f"🎨 [{caller}]: !image \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\" (generating...)"
    
    def _execute_video_command(self, prompt: str, ai_name: str) -> tuple[bool, str]:
        """Execute a video generation command."""
        caller = self._format_caller(ai_name)
        if self._is_ai_name(ai_name):
            ai_number = int(ai_name.split('-')[1])
            model_name = self.get_model_for_ai(ai_number)
        else:
            model_name = ai_name
        
        if not prompt or len(prompt.strip()) < 5:
            return False, f"❌ [{caller}]: !video — prompt too short"
        
        print(f"[Agent] Generating video for {ai_name} ({model_name}): {prompt[:100]}...")
        
        # Run video generation in background thread to avoid blocking
        import threading
        from config import SORA_SECONDS, SORA_SIZE
        
        def _run_video_job():
            from shared_utils import generate_video_with_sora
            sora_model = os.getenv("SORA_MODEL", "sora-2")
            
            # Use config values, with env var override
            sora_seconds = int(os.getenv("SORA_SECONDS", str(SORA_SECONDS)))
            sora_size = os.getenv("SORA_SIZE", SORA_SIZE) or None
            
            print(f"[Agent] Sora settings: seconds={sora_seconds}, size={sora_size}")
            
            try:
                result = generate_video_with_sora(
                    prompt=prompt,
                    model=sora_model,
                    seconds=sora_seconds,
                    size=sora_size,
                    poll_interval_seconds=5.0,
                )
                if result.get("success"):
                    video_path = result.get('video_path')
                    print(f"[Agent] Video completed: {video_path}")
                    # Track video in session
                    if hasattr(self.app, 'session_videos') and video_path:
                        self.app.session_videos.append(str(video_path))
                        # Emit signal to update video preview on main thread (include prompt for status bar)
                        if hasattr(self, 'video_signals'):
                            self.video_signals.video_ready.emit(str(video_path), prompt, ai_name, model_name)
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"[Agent] Video failed: {error}")
                    # Emit failure signal so UI can show error to user and AIs
                    if hasattr(self, 'video_signals'):
                        self.video_signals.video_failed.emit(ai_name, prompt, error)
            except Exception as e:
                print(f"[Agent] Video generation exception: {e}")
                if hasattr(self, 'video_signals'):
                    self.video_signals.video_failed.emit(ai_name, prompt, str(e))
        
        threading.Thread(target=_run_video_job, daemon=True).start()
        # Return None for _command_success to show yellow "in progress" color (not green success)
        return None, f"🎬 [{caller}]: !video \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\" (generating...)"
    
    def _execute_add_ai_command(self, model_name: str, persona: str, requesting_ai: str) -> tuple[bool, str]:
        """Execute an add AI participant command."""
        caller = self._format_caller(requesting_ai)

        # Check if model name was provided
        if not model_name or not model_name.strip():
            return False, f"❌ [{caller}]: !add_ai — no model specified"
        
        # Selector is updated immediately on each !add_ai, so it reflects the current count
        current_num_ais = int(self.app.right_sidebar.control_panel.num_ais_selector.currentText())

        if current_num_ais >= 5:
            return False, f"❌ [{caller}]: !add_ai \"{model_name}\" — maximum of 5 AIs already reached"

        new_num = current_num_ais + 1
        
        # Get tier setting FIRST - we need this to guide model matching
        invite_tier_setting = self.app.right_sidebar.control_panel.get_ai_invite_tier()
        print(f"[Agent] Processing !add_ai '{model_name}' with tier setting: {invite_tier_setting}")
        
        # Try to set the model for the new AI slot
        actual_model_id = None  # Track the actual model ID
        actual_display_name = model_name  # Track display name for messages
        selector = getattr(self.app.right_sidebar.control_panel, f'ai{new_num}_model_selector', None)
        
        if selector:
            # Smart matching: prefer models from the allowed tier
            # This prevents "Claude" from matching paid Claude when Free tier is selected
            found = False
            
            # First pass: only match models from the ALLOWED tier
            if invite_tier_setting in ["Free", "Paid"]:
                for i in range(selector.count()):
                    item_text = selector.itemText(i).lower()
                    if model_name.lower() in item_text:
                        # Check if this model is from the allowed tier
                        potential_model_id = selector.get_model_id_at_index(i)
                        if potential_model_id:
                            potential_tier = get_model_tier_by_id(potential_model_id)
                            if potential_tier == invite_tier_setting:
                                # Found a match in the allowed tier!
                                selector.setCurrentIndex(i)
                                actual_display_name = selector.itemText(i)
                                actual_model_id = potential_model_id
                                found = True
                                print(f"[Agent] Matched '{model_name}' to {actual_display_name} ({potential_tier} tier)")
                                break
            
            # Second pass: if tier is "Both" OR no match in allowed tier, search all
            if not found:
                for i in range(selector.count()):
                    if model_name.lower() in selector.itemText(i).lower():
                        selector.setCurrentIndex(i)
                        actual_display_name = selector.itemText(i)
                        actual_model_id = selector.get_model_id_at_index(i)
                        found = True
                        print(f"[Agent] Fallback matched '{model_name}' to {actual_display_name}")
                        break
            
            if not found:
                # Use whatever is default
                actual_display_name = selector.currentText()
                actual_model_id = selector.get_selected_model_id()
                print(f"[Agent] No match for '{model_name}', using default: {actual_display_name}")
        
        # Fallback if we couldn't get the model ID
        if not actual_model_id:
            actual_model_id = actual_display_name
        
        # If actual_display_name is still empty, use original model_name
        if not actual_display_name:
            actual_display_name = model_name if model_name else "(no model specified)"
        
        # Final tier check - enforce the setting even for fallback matches
        model_tier = get_model_tier_by_id(actual_model_id)
        print(f"[Agent] Final model: {actual_model_id} (tier: {model_tier})")
        
        if invite_tier_setting == "Free" and model_tier != "Free":
            print(f"[Agent] BLOCKED: {actual_model_id} is {model_tier}, but only Free allowed")
            return False, f"❌ [{caller}]: !add_ai \"{actual_display_name}\" — only FREE models allowed"
        elif invite_tier_setting == "Paid" and model_tier != "Paid":
            print(f"[Agent] BLOCKED: {actual_model_id} is {model_tier}, but only Paid allowed")
            return False, f"❌ [{caller}]: !add_ai \"{actual_display_name}\" — only PAID models allowed"
        # "Both" allows everything
        
        # Store persona for later use (could be used to modify system prompt)
        if persona:
            if not hasattr(self.app, 'custom_personas'):
                self.app.custom_personas = {}
            self.app.custom_personas[f"AI-{new_num}"] = persona
        
        # Track this AI as pending so it can join the current round
        if not hasattr(self, '_pending_ais'):
            self._pending_ais = []
        
        # Check if this model is already an active AI (deduplication)
        # Check setting for whether duplicates are allowed
        allow_duplicates = getattr(self.app.right_sidebar.control_panel, 'allow_duplicate_models_checkbox', None)
        if allow_duplicates and not allow_duplicates.isChecked():
            for i in range(1, current_num_ais + 1):
                existing_selector = getattr(self.app.right_sidebar.control_panel, f'ai{i}_model_selector', None)
                if existing_selector:
                    existing_model_id = existing_selector.get_selected_model_id()
                    if existing_model_id and actual_model_id:
                        if actual_model_id.lower() == existing_model_id.lower():
                            print(f"[Agent] {actual_model_id} already active as AI-{i}, skipping duplicate")
                            return None, f"ℹ️ [{caller}]: !add_ai \"{actual_display_name}\" — already in conversation as AI-{i}"
        
        # Check if this model was already invited this round (pending deduplication - always enforce)
        already_pending = any(p['model'].lower() == actual_model_id.lower() for p in self._pending_ais)
        if already_pending:
            print(f"[Agent] {actual_model_id} already invited this round, skipping duplicate")
            return None, f"ℹ️ [{caller}]: !add_ai \"{actual_display_name}\" — already invited this round"
        
        # Immediately update the num_ais selector and make the new slot visible
        self.app.right_sidebar.control_panel.num_ais_selector.set_value(new_num)
        self.app.right_sidebar.control_panel.update_ai_selector_visibility(str(new_num))
        print(f"[Agent] Updated AI count to {new_num} and made AI-{new_num} slot visible")

        self._pending_ais.append({
            'ai_name': f"AI-{new_num}",
            'ai_number': new_num,
            'model': actual_model_id,  # Store the model ID, not display name
            'display_name': actual_display_name,  # Keep display name for messages
            'persona': persona,
            'invited_by': requesting_ai
        })
        print(f"[Agent] Queued AI-{new_num} ({actual_model_id}) to join current round")
        print(f"[Agent] Current pending queue: {[p['ai_name'] + ' -> ' + p['model'] for p in self._pending_ais]}")
        
        # Create a friendly notification message that shows the command syntax
        if persona:
            return True, f"✨ [{caller}]: !add_ai \"{actual_display_name}\" \"{persona}\""
        else:
            return True, f"✨ [{caller}]: !add_ai \"{actual_display_name}\""
    
    def _execute_remove_ai_command(self, target: str, requesting_ai: str) -> tuple[bool, str]:
        """Execute a remove AI participant command (requires consensus in future)."""
        caller = self._format_caller(requesting_ai)
        return False, f"🗳️ [{caller}]: !remove_ai \"{target}\" — consensus not yet implemented"
    
    def _execute_list_models_command(self, ai_name: str) -> tuple[bool, str]:
        """Execute a list models command - returns available models for invitation."""
        caller = self._format_caller(ai_name)
        try:
            models_file = os.path.join(os.path.dirname(__file__), 'available_models.txt')
            if os.path.exists(models_file):
                with open(models_file, 'r', encoding='utf-8') as f:
                    models_content = f.read()
                print(f"[Agent] {ai_name} queried available models")
                return True, f"📋 [{caller}]: !list_models\n{models_content}"
            else:
                return False, f"❌ [{caller}]: !list_models — models list not found"
        except Exception as e:
            return False, f"❌ [{caller}]: !list_models — error: {e}"
    
    def _execute_mute_command(self, ai_name: str) -> tuple[bool, str]:
        """Execute a mute self command - AI skips next turn."""
        caller = self._format_caller(ai_name)

        if not hasattr(self.app, 'muted_ais'):
            self.app.muted_ais = set()

        self.app.muted_ais.add(ai_name)
        return True, f"🔇 [{caller}]: !mute_self"

    def _execute_search_command(self, query: str, ai_name: str) -> tuple[bool, str]:
        """Execute a web search command and inject results into conversation."""
        from shared_utils import web_search
        caller = self._format_caller(ai_name)

        if not query or len(query.strip()) < 3:
            return False, f"❌ [{caller}]: !search — query too short"

        print(f"[Agent] Searching for {caller}: {query}")

        # Perform the search
        search_result = web_search(query, max_results=5)

        if not search_result.get('success'):
            error_msg = search_result.get('error', 'Unknown error')
            return False, f"❌ [{caller}]: !search \"{query}\" — {error_msg}"

        # Format results for display
        results = search_result.get('results', [])
        if not results:
            return False, f"❌ [{caller}]: !search \"{query}\" — no results found"

        # Format results for conversation context (with markdown formatting)
        formatted = f"🔍 [{caller}]: !search \"{query}\"\n\n**Search Results:**\n"
        for i, r in enumerate(results, 1):
            formatted += f"\n{i}. **{r.get('title', 'No title')}**\n"
            formatted += f"   {r.get('snippet', 'No snippet')}\n"
            formatted += f"   Source: {r.get('url', 'No URL')}\n"

        # Add search results to conversation so all AIs can see them
        search_message = {
            "role": "user",
            "content": formatted,
            "_type": "search_result",
            "hidden": False
        }
        self.app.main_conversation.append(search_message)

        # Trigger UI update by redisplaying conversation
        self.app.left_pane.display_conversation(self.app.main_conversation)

        return True, f"🔍 [{caller}]: !search \"{query}\" (found {len(results)} results)"

    def _execute_prompt_command(self, text: str, ai_name: str) -> tuple[bool, str]:
        """Execute a prompt addition command - appends to system prompt.
        Note: !prompt commands are stripped from conversation context so other AIs don't see them,
        but the full text is shown in the GUI notification for the human operator.
        A subtle notification is added to context so other AIs know the action occurred."""
        caller = self._format_caller(ai_name)

        if not text or len(text.strip()) < 3:
            return False, f"❌ [{caller}]: !prompt — prompt text too short"

        # Initialize list if needed
        if ai_name not in self.ai_prompt_additions:
            self.ai_prompt_additions[ai_name] = []

        # Add the new prompt text (appends, doesn't replace)
        self.ai_prompt_additions[ai_name].append(text.strip())

        print(f"[Agent] {caller} added to their prompt: {text[:50]}...")
        print(f"[Agent] {ai_name} now has {len(self.ai_prompt_additions[ai_name])} prompt additions")

        # Add a subtle notification to conversation context (visible to other AIs)
        # This lets them know the action occurred without revealing the content
        context_notification = {
            "role": "system",
            "content": f"[{ai_name} modified their system prompt]",
            "_type": "system_notification"
        }
        self.app.main_conversation.append(context_notification)

        # Trigger UI update by redisplaying conversation
        self.app.left_pane.display_conversation(self.app.main_conversation)

        # Show full untruncated text in notification (only human sees this, not other AIs)
        return True, f"💭 [{caller}]: !prompt \"{text}\""

    def _execute_temperature_command(self, value: str, ai_name: str) -> tuple[bool, str]:
        """Execute a temperature modification command - sets sampling temperature.
        Note: !temperature commands are stripped from conversation context."""
        caller = self._format_caller(ai_name)

        # Validate temperature value
        try:
            temp = float(value)
            if temp < 0 or temp > 2:
                return False, f"❌ [{caller}]: !temperature {value} — must be between 0 and 2"
        except (ValueError, TypeError):
            return False, f"❌ [{caller}]: !temperature — invalid value '{value}'"

        # Store the temperature for this AI
        self.ai_temperatures[ai_name] = temp

        print(f"[Agent] {caller} set their temperature to {temp}")

        # Add a subtle notification to conversation context (visible to other AIs)
        context_notification = {
            "role": "system",
            "content": f"[{ai_name} adjusted their temperature]",
            "_type": "system_notification"
        }
        self.app.main_conversation.append(context_notification)

        # Trigger UI update by redisplaying conversation
        self.app.left_pane.display_conversation(self.app.main_conversation)

        # Show the actual value in notification for human
        return True, f"🌡️ [{caller}]: !temperature {temp}"

    def _execute_vote_command(self, question: str, options_str: str, ai_name: str) -> tuple[bool, str]:
        """Execute a vote/poll command - creates a poll visible to all AIs."""
        caller = self._format_caller(ai_name)

        if not question:
            return False, f"❌ [{caller}]: !vote — no question provided"

        # Parse options if provided (comma-separated)
        options = []
        if options_str:
            options = [opt.strip().strip('"').strip("'") for opt in options_str.split(',') if opt.strip()]

        # Build poll display
        poll_text = f"📊 POLL by {caller}: {question}"
        if options:
            for i, opt in enumerate(options, 1):
                poll_text += f"\n  {i}. {opt}"
        else:
            poll_text += "\n  (Open-ended — respond with your thoughts)"

        # Store active poll for tracking
        if not hasattr(self, '_active_polls'):
            self._active_polls = []
        self._active_polls.append({
            'question': question,
            'options': options,
            'creator': ai_name,
            'votes': {}
        })

        return True, poll_text

    def _execute_whisper_command(self, target: str, message: str, ai_name: str) -> tuple[bool, str]:
        """Execute a whisper command - private message to a specific AI."""
        caller = self._format_caller(ai_name)

        if not target or not message:
            return False, f"❌ [{caller}]: !whisper — missing target or message"

        # Normalize target (accept "AI-1", "ai-1", "1", etc.)
        target_normalized = target.upper().strip()
        if not target_normalized.startswith('AI-'):
            target_normalized = f"AI-{target_normalized}"

        # Check if target AI exists (based on current number of AIs)
        try:
            target_num = int(target_normalized.split('-')[1])
            num_ais = int(self.app.right_sidebar.control_panel.num_ais_selector.currentText())
            if target_num < 1 or target_num > num_ais:
                return False, f"❌ [{caller}]: !whisper — {target_normalized} doesn't exist (only {num_ais} AIs active)"
        except (ValueError, IndexError):
            return False, f"❌ [{caller}]: !whisper — invalid target '{target}'"

        # Add the whisper as a hidden message that only appears in the target's context
        # We create a special message that gets filtered per-AI during turn processing
        whisper_msg = {
            "role": "system",
            "content": f"[Private whisper from {caller}]: {message}",
            "_type": "whisper",
            "_whisper_from": ai_name,
            "_whisper_to": target_normalized,
            "hidden": True  # Hidden from main display
        }

        # Add to conversation
        if self.app.active_branch:
            branch_id = self.app.active_branch
            if branch_id in self.app.branch_conversations:
                self.app.branch_conversations[branch_id]['conversation'].append(whisper_msg)
        else:
            if not hasattr(self.app, 'main_conversation'):
                self.app.main_conversation = []
            self.app.main_conversation.append(whisper_msg)

        # Show notification (visible) but actual whisper content is private
        return True, f"🤫 [{caller}]: whispered to {target_normalized}"

    def get_model_for_ai(self, ai_number):
        """Get the selected model ID for the AI by number (1-5)"""
        selectors = {
            1: self.app.right_sidebar.control_panel.ai1_model_selector,
            2: self.app.right_sidebar.control_panel.ai2_model_selector,
            3: self.app.right_sidebar.control_panel.ai3_model_selector,
            4: self.app.right_sidebar.control_panel.ai4_model_selector,
            5: self.app.right_sidebar.control_panel.ai5_model_selector
        }
        selector = selectors.get(ai_number, selectors[1])
        # Use get_selected_model_id() to get the actual model ID, not display name
        model_id = selector.get_selected_model_id()
        return model_id if model_id else selector.currentText()  # Fallback to text if no ID

    def get_prompt_additions_for_ai(self, ai_name: str) -> str:
        """Get all prompt additions for a specific AI as a formatted string."""
        if ai_name not in self.ai_prompt_additions or not self.ai_prompt_additions[ai_name]:
            return ""

        additions = self.ai_prompt_additions[ai_name]
        return "\n\n[Your remembered insights/perspectives]:\n- " + "\n- ".join(additions)

    def get_temperature_for_ai(self, ai_name: str) -> float:
        """Get the temperature setting for a specific AI (default 1.0)."""
        return self.ai_temperatures.get(ai_name, 1.0)

    def on_ai_error(self, error_message):
        """Handle AI errors for both main and branch conversations"""
        # Clear all typing indicators on error
        self._clear_all_typing_indicators()
        
        # Format the error message
        error_message_formatted = {
            "role": "system",
            "content": f"Error: {error_message}",
            "_type": "agent_notification",
            "_command_success": False  # Show as error notification
        }
        
        # Check if we're in a branch or main conversation
        if self.app.active_branch:
            # Branch conversation
            branch_id = self.app.active_branch
            if branch_id in self.app.branch_conversations:
                branch_data = self.app.branch_conversations[branch_id]
                conversation = branch_data['conversation']
                
                # Add error message to conversation
                conversation.append(error_message_formatted)
                
                # Update the conversation display
                self.app.left_pane.display_conversation(conversation, branch_data)
        else:
            # Main conversation
            if not hasattr(self.app, 'main_conversation'):
                self.app.main_conversation = []
            
            # Add error message to conversation
            self.app.main_conversation.append(error_message_formatted)
            
            # Update the conversation display
            self.app.left_pane.display_conversation(self.app.main_conversation)
        
        # Update status bar
        self.app.statusBar().showMessage(f"Error: {error_message}")
        self.app.left_pane.stop_loading()
        
    def rabbithole_callback(self, selected_text):
        """Create a rabbithole branch from selected text"""
        print(f"Creating rabbithole branch for: '{selected_text}'")
        
        # Create unique branch ID
        branch_id = f"rabbithole_{time.time()}"
        
        # Create a new conversation for the branch
        branch_conversation = []
        
        # If we're branching from another branch, copy over relevant context
        parent_conversation = []
        parent_id = None
        
        if self.app.active_branch:
            # Branching from another branch
            parent_id = self.app.active_branch
            parent_data = self.app.branch_conversations[parent_id]
            parent_conversation = parent_data['conversation']
        else:
            # Branching from main conversation
            parent_conversation = self.app.main_conversation
        
        # Copy ALL previous context except branch indicators
        for msg in parent_conversation:
            if not msg.get('_type') == 'branch_indicator':
                # Copy the message excluding branch indicators
                branch_conversation.append(msg.copy())
        
        # Add the branch indicator at the END (not beginning) 
        branch_message = {
            "role": "system", 
            "content": f"🐇 Rabbitholing down: \"{selected_text}\"",
            "_type": "branch_indicator"  # Special flag for branch indicators
        }
        branch_conversation.append(branch_message)
        
        # Store the branch data
        self.app.branch_conversations[branch_id] = {
            'type': 'rabbithole',
            'selected_text': selected_text,
            'conversation': branch_conversation,
            'parent': parent_id
        }
        
        # Activate the branch
        self.app.active_branch = branch_id
        
        # Update the UI
        visible_conversation = [msg for msg in branch_conversation if not msg.get('hidden', False)]
        self.app.left_pane.display_conversation(visible_conversation, self.app.branch_conversations[branch_id])
        
        # Add node to network graph
        parent_node = parent_id if parent_id else 'main'
        self.app.right_sidebar.add_node(branch_id, f'🐇 {selected_text[:15]}...', 'rabbithole')
        self.app.right_sidebar.add_edge(parent_node, branch_id)
        
        # Process the branch conversation
        self.process_branch_input(selected_text)

    def fork_callback(self, selected_text):
        """Create a fork branch from selected text"""
        print(f"Creating fork branch for: '{selected_text}'")
        
        # Create unique branch ID
        branch_id = f"fork_{time.time()}"
        
        # Create a new conversation for the branch
        branch_conversation = []
        
        # If we're branching from another branch, copy over relevant context
        parent_conversation = []
        parent_id = None
        
        if self.app.active_branch:
            # Forking from another branch
            parent_id = self.app.active_branch
            parent_data = self.app.branch_conversations[parent_id]
            parent_conversation = parent_data['conversation']
        else:
            # Forking from main conversation
            parent_conversation = self.app.main_conversation
        
        # For fork branches, only include context UP TO the selected text
        truncate_idx = None
        msg_with_text = None
        
        # First pass: find the message containing the selected text
        for i, msg in enumerate(parent_conversation):
            if msg.get('role') in ['user', 'assistant'] and selected_text in msg.get('content', ''):
                truncate_idx = i
                msg_with_text = msg
                break
        
        # If we didn't find the selected text, include all messages
        # This can happen with multi-line selections that span messages
        if truncate_idx is None:
            print(f"Warning: Selected text not found in any single message, including all context")
            # Copy all messages except branch indicators
            for msg in parent_conversation:
                if not msg.get('_type') == 'branch_indicator':
                    branch_conversation.append(msg.copy())
        else:
            # We found the message with the selected text, proceed as normal
            # Second pass: add all messages up to the truncate point
            for i, msg in enumerate(parent_conversation):
                # Always include system messages that aren't branch indicators
                if msg.get('role') == 'system' and not msg.get('_type') == 'branch_indicator':
                    branch_conversation.append(msg.copy())
                    continue
                
                # For non-system messages, only include up to truncate point
                if i <= truncate_idx:
                    # Add message (potentially modified if it's the truncate point)
                    if i == truncate_idx:
                        # This is the message containing the selected text
                        # Truncate the message at the selected text if possible
                        content = msg.get('content', '')
                        if selected_text in content:
                            # Find where the selected text occurs
                            pos = content.find(selected_text)
                            # Include everything up to and including the selected text
                            truncated_content = content[:pos + len(selected_text)]
                            
                            # Create a modified copy of the message with truncated content
                            modified_msg = msg.copy()
                            modified_msg['content'] = truncated_content
                            branch_conversation.append(modified_msg)
                        else:
                            # If we can't find the text (unlikely), just add the whole message
                            branch_conversation.append(msg.copy())
                    else:
                        # Regular message before the truncate point
                        branch_conversation.append(msg.copy())
        
        # Add the branch indicator as the last message
        branch_message = {
            "role": "system", 
            "content": f"🍴 Forking off: \"{selected_text}\"",
            "_type": "branch_indicator"  # Special flag for branch indicators
        }
        branch_conversation.append(branch_message)
        
        # Create properly formatted fork instruction - simplified to just "..."
        fork_instruction = "..."
        
        # Store the branch data
        self.app.branch_conversations[branch_id] = {
            'type': 'fork',
            'selected_text': selected_text,
            'conversation': branch_conversation,
            'parent': parent_id
        }
        
        # Activate the branch
        self.app.active_branch = branch_id
        
        # Update the UI
        visible_conversation = [msg for msg in branch_conversation if not msg.get('hidden', False)]
        self.app.left_pane.display_conversation(visible_conversation, self.app.branch_conversations[branch_id])
        
        # Add node to network graph
        parent_node = parent_id if parent_id else 'main'
        self.app.right_sidebar.add_node(branch_id, f'🍴 {selected_text[:15]}...', 'fork')
        self.app.right_sidebar.add_edge(parent_node, branch_id)
        
        # Process the branch conversation with the proper instruction but mark it as hidden
        self.process_branch_input_with_hidden_instruction(fork_instruction)

    def process_branch_input_with_hidden_instruction(self, user_input):
        """Process input from the user specifically for branch conversations, but mark the input as hidden"""
        # Check if we have an active branch
        if not self.app.active_branch:
            # Fallback to main conversation if no active branch
            self.process_input(user_input)
            return
            
        # Get branch data
        branch_id = self.app.active_branch
        branch_data = self.app.branch_conversations[branch_id]
        conversation = branch_data['conversation']
        
        # Add user input if provided, but mark it as hidden
        if user_input:
            user_message = {
                "role": "user",
                "content": user_input,
                "hidden": True  # Mark as hidden
            }
            conversation.append(user_message)
            
            # No need to update display since message is hidden
        
        # Get selected models and prompt pair from UI
        ai_1_model = self.app.right_sidebar.control_panel.ai1_model_selector.currentText()
        ai_2_model = self.app.right_sidebar.control_panel.ai2_model_selector.currentText()
        ai_3_model = self.app.right_sidebar.control_panel.ai3_model_selector.currentText()
        selected_prompt_pair = self.app.right_sidebar.control_panel.prompt_pair_selector.currentText()
        
        # Check if we've already had AI responses in this branch
        has_ai_responses = False
        ai_response_count = 0
        for msg in conversation:
            if msg.get('role') == 'assistant':
                has_ai_responses = True
                ai_response_count += 1
        
        # Determine which prompts to use based on branch type and response history
        branch_type = branch_data.get('type', 'branch')
        selected_text = branch_data.get('selected_text', '')
        
        if branch_type.lower() == 'rabbithole' and ai_response_count < 2:
            # Initial rabbitholing prompt - only for the first exchange
            print("Using rabbithole-specific prompt for initial exploration")
            rabbithole_prompt = f"'{selected_text}'!!!"
            ai_1_prompt = rabbithole_prompt
            ai_2_prompt = rabbithole_prompt
            ai_3_prompt = rabbithole_prompt
        else:
            # After initial exploration, revert to standard prompts
            print("Using standard prompts for continued conversation")
            ai_1_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI-1"]
            ai_2_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI-2"]
            ai_3_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI-3"]
        
        # Start loading animation
        self.app.left_pane.start_loading()
        
        # Reset turn count ONLY if this is a new conversation or explicit user input
        # Don't reset during automatic iterations
        if user_input is not None or not has_ai_responses:
            self.app.turn_count = 0
            print("Resetting turn count - starting new conversation")
        
        # Get max iterations
        max_iterations = int(self.app.right_sidebar.control_panel.iterations_selector.currentText())
        
        # Get invite tier setting
        invite_tier = self.app.right_sidebar.control_panel.get_ai_invite_tier()
        
        # Create worker threads for AI-1, AI-2, and AI-3
        worker1 = Worker("AI-1", conversation, ai_1_model, ai_1_prompt, is_branch=True, branch_id=branch_id, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
        worker2 = Worker("AI-2", conversation, ai_2_model, ai_2_prompt, is_branch=True, branch_id=branch_id, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
        worker3 = Worker("AI-3", conversation, ai_3_model, ai_3_prompt, is_branch=True, branch_id=branch_id, gui=self.app, invite_tier=invite_tier, prompt_modifications=self.ai_prompt_additions, ai_temperatures=self.ai_temperatures)
        
        # Connect signals for worker1
        worker1.signals.started.connect(self.on_ai_started)
        worker1.signals.response.connect(self.on_ai_response_received)
        worker1.signals.result.connect(self.on_ai_result_received)
        worker1.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker1.signals.finished.connect(lambda: self.start_ai2_turn(conversation, worker2))
        worker1.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker2
        worker2.signals.started.connect(self.on_ai_started)
        worker2.signals.response.connect(self.on_ai_response_received)
        worker2.signals.result.connect(self.on_ai_result_received)
        worker2.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker2.signals.finished.connect(lambda: self.start_ai3_turn(conversation, worker3))
        worker2.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker3
        worker3.signals.started.connect(self.on_ai_started)
        worker3.signals.response.connect(self.on_ai_response_received)
        worker3.signals.result.connect(self.on_ai_result_received)
        worker3.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker3.signals.finished.connect(lambda: self.handle_turn_completion(max_iterations))
        worker3.signals.error.connect(self.on_ai_error)
        
        # Start AI-1's turn
        self.thread_pool.start(worker1)

    def update_conversation_html(self, conversation):
        """Update the full conversation HTML document with all messages"""
        try:
            from datetime import datetime
            
            # Create a filename for the full conversation HTML using session timestamp
            from config import OUTPUTS_DIR
            
            # Get session timestamp from main window, or create new one
            session_timestamp = getattr(self.app, 'session_timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            html_filename = f"conversation_{session_timestamp}.html"
            html_file = os.path.join(OUTPUTS_DIR, html_filename)
            
            # Store the current file path on the app for export/view functionality
            self.app.current_html_file = html_file
            
            # Generate HTML content for the conversation
            html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <title>Liminal Backrooms</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/x-icon" href="../assets/app-icon.ico">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0A0E1A;
            --bg-panel: #111827;
            --bg-message: #151C2C;
            --border: #1E293B;
            --border-glow: #06B6D4;
            --text-primary: #CBD5E1;
            --text-dim: #64748B;
            --text-bright: #F1F5F9;
            --accent-cyan: #06B6D4;
            --accent-purple: #A855F7;
            --accent-pink: #EC4899;
            --accent-green: #10B981;
            --accent-yellow: #FBBF24;
            --ai-1: #6FFFE6;
            --ai-2: #06E2D4;
            --ai-3: #54F5E9;
            --ai-4: #8BFCEF;
            --ai-5: #91FCFD;
            --human: #ff00b3;
        }
        
        * { 
            box-sizing: border-box; 
            margin: 0;
            padding: 0;
        }
        
        body { 
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.5; 
            color: var(--text-primary);
            background: var(--bg-dark);
            min-height: 100vh;
        }
        
        .container {
            max-width: 860px;
            margin: 0 auto;
            padding: 24px 16px;
        }
        
        /* Header - retro terminal style */
        header {
            text-align: center;
            margin-bottom: 32px;
            padding: 20px 16px;
            background: var(--bg-panel);
            border: 1px solid var(--border-glow);
            border-top: 3px solid var(--accent-cyan);
            position: relative;
        }
        
        header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent-purple), transparent);
        }
        
        h1 { 
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-cyan);
            font-size: 1.4em;
            margin: 0 0 6px 0;
            font-weight: 500;
            letter-spacing: 4px;
            text-transform: uppercase;
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
        }
        
        .subtitle {
            color: var(--text-dim);
            font-size: 0.8em;
            font-weight: 300;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        
        /* Conversation container */
        #conversation {
            display: block;
        }
        
        /* Message base - tight padding, no rounded corners */
        .message { 
            padding: 10px 14px;
            margin-bottom: 8px;
            background: var(--bg-message);
            border: 1px solid var(--border);
            border-left: 3px solid var(--ai-1);
            overflow-wrap: break-word;
            word-wrap: break-word;
            word-break: break-word;
        }
        
        /* Human user messages - left aligned like everything else */
        .message.user {
            border-left-color: var(--human);
        }
        
        /* Assistant messages - AI-specific border colors */
        .message.assistant {
            border-left-color: var(--ai-1);
        }
        .message.assistant.ai-1-msg { border-left-color: var(--ai-1); }
        .message.assistant.ai-2-msg { border-left-color: var(--ai-2); }
        .message.assistant.ai-3-msg { border-left-color: var(--ai-3); }
        .message.assistant.ai-4-msg { border-left-color: var(--ai-4); }
        .message.assistant.ai-5-msg { border-left-color: var(--ai-5); }
        
        /* Generated image messages - use AI color, not pink */
        .message.generated-image {
            border-left-color: var(--ai-1);
        }
        .message.generated-image.ai-1-msg { border-left-color: var(--ai-1); }
        .message.generated-image.ai-2-msg { border-left-color: var(--ai-2); }
        .message.generated-image.ai-3-msg { border-left-color: var(--ai-3); }
        .message.generated-image.ai-4-msg { border-left-color: var(--ai-4); }
        .message.generated-image.ai-5-msg { border-left-color: var(--ai-5); }
        
        /* System messages */
        .message.system {
            border-left-color: var(--accent-yellow);
            background: rgba(251, 191, 36, 0.04);
            font-style: italic;
        }
        
        /* Agent notifications */
        .message.agent-notification {
            border-left-color: var(--accent-green);
            background: rgba(16, 185, 129, 0.06);
            font-size: 0.9em;
        }
        
        .message-content {
            width: 100%;
        }
        
        /* Header - compact */
        .header { 
            font-weight: 500;
            margin-bottom: 6px; 
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            font-size: 0.85em;
        }
        
        .ai-name {
            font-weight: 600;
        }
        
        /* AI-specific name colors */
        .ai-name.ai-1 { color: var(--ai-1); }
        .ai-name.ai-2 { color: var(--ai-2); }
        .ai-name.ai-3 { color: var(--ai-3); }
        .ai-name.ai-4 { color: var(--ai-4); }
        .ai-name.ai-5 { color: var(--ai-5); }
        .ai-name.human { color: var(--human); }
        .ai-name.system { color: var(--accent-yellow); }
        
        .model-name {
            color: var(--text-dim);
            font-size: 0.9em;
        }
        
        .timestamp {
            font-size: 0.8em;
            color: var(--text-dim);
            margin-left: auto;
        }
        
        /* Content */
        .content {
            white-space: pre-wrap;
            font-size: 0.95em;
            line-height: 1.6;
            overflow-wrap: break-word;
            word-wrap: break-word;
            word-break: break-word;
        }
        
        .greentext {
            color: #789922;
        }
        
        /* Code - sharp edges */
        code { 
            background: #0F1419;
            padding: 1px 5px;
            font-size: 0.9em;
            color: var(--accent-cyan);
            border: 1px solid var(--border);
        }
        
        pre { 
            background: #0F1419;
            padding: 12px 14px;
            overflow-x: auto;
            font-size: 0.85em;
            margin: 12px 0 12px 12px;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-bright);
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
        }
        
        .code-header {
            background: #1A1F26;
            padding: 6px 12px;
            border-bottom: 1px solid var(--border);
            border-radius: 4px 4px 0 0;
            margin: -12px -14px 12px -14px;
        }
        
        .code-lang {
            color: var(--text-dim);
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Syntax highlighting */
        .syn-keyword { color: #A78BFA; font-weight: bold; }
        .syn-string { color: #5DFF44; }
        .syn-number { color: #22D3EE; }
        .syn-comment { color: #64748B; font-style: italic; }
        .syn-function { color: #FBBF24; }
        
        /* Images */
        .message-image {
            margin-top: 10px;
        }
        
        .message-image img {
            max-width: 100%;
            border: 1px solid var(--border);
        }
        
        .image-credit {
            color: var(--text-dim);
            font-size: 0.75em;
            margin-top: 4px;
            font-style: italic;
        }
        
        /* Footer */
        footer {
            margin-top: 40px;
            text-align: center;
            padding: 20px 16px;
            border-top: 1px solid var(--border);
        }
        
        footer p {
            color: var(--text-dim);
            font-size: 0.8em;
            letter-spacing: 1px;
        }
        
        footer a {
            color: var(--accent-cyan);
            text-decoration: none;
        }
        
        /* Share button */
        .share-bar {
            position: fixed;
            top: 16px;
            right: 16px;
            z-index: 1000;
        }
        
        .share-btn {
            background: var(--bg-panel);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            padding: 8px 16px;
            cursor: pointer;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8em;
            transition: all 0.2s ease;
        }
        
        .share-btn:hover {
            background: rgba(6, 182, 212, 0.15);
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.3);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container { padding: 16px 12px; }
            h1 { font-size: 1.1em; letter-spacing: 2px; }
            header { padding: 16px 12px; margin-bottom: 24px; }
            .message { padding: 8px 12px; }
            .header { flex-direction: column; align-items: flex-start; gap: 2px; }
            .timestamp { margin-left: 0; }
            .share-btn { padding: 6px 12px; font-size: 0.75em; }
        }
        
        @media (max-width: 480px) {
            .container { padding: 12px 8px; }
            h1 { font-size: 1em; }
            .message { padding: 8px 10px; }
            .content { font-size: 0.9em; }
            pre { padding: 10px 12px; font-size: 0.8em; }
        }
        
        @media print {
            body { background: white; color: black; }
            .message { background: #f5f5f5; border: 1px solid #ddd; }
            .share-bar { display: none; }
        }
    </style>
</head>
<body>
    <div class="share-bar">
        <button class="share-btn" onclick="copyPageUrl()">📋 COPY LINK</button>
    </div>
    
    <div class="container">
        <header>
            <h1>⟨ LIMINAL BACKROOMS ⟩</h1>
            <p class="subtitle">Conversation Archive</p>
        </header>
        
        <div id="conversation">"""
            
            # Add each message to the HTML content
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")
                ai_name = msg.get("ai_name", "")
                model = msg.get("model", "")
                msg_type = msg.get("_type", "")
                image_model = msg.get("image_model", "")
                timestamp = datetime.now().strftime("%b %d, %Y %I:%M %p")
                
                # Skip special system messages or empty messages
                if role == "system" and msg_type == "branch_indicator":
                    continue
                
                # Skip most notifications in HTML output - only keep !add_ai ones
                if msg_type == "agent_notification":
                    content_str = content if isinstance(content, str) else ""
                    if "!add_ai" not in content_str:
                        continue
                
                # Check if content is empty (handle both string and list)
                is_empty = False
                if isinstance(content, str):
                    is_empty = not content.strip()
                elif isinstance(content, list):
                    text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                    is_empty = not any(text_parts) and not any(part.get('type') == 'image' for part in content)
                else:
                    is_empty = not content
                
                if is_empty:
                    continue
                
                # Extract text content from structured messages
                text_content = ""
                if isinstance(content, str):
                    text_content = content
                elif isinstance(content, list):
                    text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                    text_content = '\n'.join(text_parts)
                
                # Process content to properly format code blocks and add greentext styling
                processed_content = self.app.left_pane.process_content_with_code_blocks(text_content) if text_content else ""
                processed_content = self.apply_greentext_styling(processed_content)
                
                # Message class based on role and type
                message_class = role
                if msg_type == "agent_notification":
                    message_class = "agent-notification"
                elif msg_type == "generated_image":
                    message_class = "generated-image"
                
                # Check if this message has an associated image
                has_image = False
                image_path = None
                image_base64 = None
                
                if hasattr(msg, "get") and callable(msg.get):
                    image_path = msg.get("generated_image_path", None)
                    if image_path:
                        has_image = True
                
                if isinstance(content, list):
                    for part in content:
                        if part.get('type') == 'image':
                            source = part.get('source', {})
                            if source.get('type') == 'base64':
                                image_base64 = source.get('data', '')
                                has_image = True
                                break
                
                # Helper to get AI number from ai_name
                def get_ai_num(name):
                    if name and '-' in name:
                        try:
                            return max(1, min(5, int(name.split('-')[1])))
                        except (ValueError, IndexError):
                            pass
                    return 1
                
                ai_num = get_ai_num(ai_name)
                
                # Build message class with AI-specific border styling
                # Apply to assistant and generated-image messages
                ai_msg_class = f"ai-{ai_num}-msg" if role == "assistant" or msg_type == "generated_image" else ""
                full_message_class = f"{message_class} {ai_msg_class}".strip()
                
                # Start message div
                html_content += f'\n        <div class="message {full_message_class}">'
                html_content += f'\n            <div class="message-content">'
                
                # Add header based on role
                if role == "assistant" or msg_type == "generated_image":
                    display_name = ai_name if ai_name else "AI"
                    color_class = f"ai-{ai_num}"
                    html_content += f'\n                <div class="header"><span class="ai-name {color_class}">{display_name}</span>'
                    if model:
                        html_content += f' <span class="model-name">({model})</span>'
                    html_content += f' <span class="timestamp">{timestamp}</span></div>'
                elif role == "user":
                    if msg_type == "generated_image" or (has_image and ai_name):
                        display_name = ai_name if ai_name else "AI"
                        color_class = f"ai-{ai_num}"
                        html_content += f'\n                <div class="header"><span class="ai-name {color_class}">{display_name}</span>'
                        if model:
                            html_content += f' <span class="model-name">({model})</span>'
                        html_content += f' <span class="timestamp">{timestamp}</span></div>'
                    else:
                        html_content += f'\n                <div class="header"><span class="ai-name human">Human User</span> <span class="timestamp">{timestamp}</span></div>'
                elif role == "system" and msg_type != "agent_notification":
                    html_content += f'\n                <div class="header"><span class="ai-name system">System</span> <span class="timestamp">{timestamp}</span></div>'
                
                # Add message content
                if processed_content and processed_content.strip():
                    html_content += f'\n                <div class="content">{processed_content}</div>'
                
                html_content += '\n            </div>'
                
                # Add image if present
                if has_image:
                    html_content += f'\n            <div class="message-image">'
                    if image_base64:
                        html_content += f'\n                <img src="data:image/jpeg;base64,{image_base64}" alt="Generated image" loading="lazy" />'
                    elif image_path:
                        web_path = image_path.replace('\\', '/')
                        html_content += f'\n                <img src="{web_path}" alt="Generated image" loading="lazy" />'
                    if ai_name and (msg_type == "generated_image" or role != "user"):
                        if image_model:
                            # Format model name nicely (remove provider prefix)
                            model_display = image_model.split("/")[-1] if "/" in image_model else image_model
                            html_content += f'\n                <div class="image-credit">Generated by {ai_name} using {model_display}</div>'
                        else:
                            html_content += f'\n                <div class="image-credit">Generated by {ai_name}</div>'
                    html_content += f'\n            </div>'
                
                html_content += '\n        </div>'
            
            # Close HTML document
            html_content += """
        </div>
        
        <footer>
            <p>Generated by <a href="https://github.com/liminalbardo/liminal_backrooms" target="_blank">Liminal Backrooms</a></p>
        </footer>
    </div>
    
    <script>
        function copyPageUrl() {
            const url = window.location.href;
            navigator.clipboard.writeText(url).then(() => {
                const btn = document.querySelector('.share-btn');
                btn.textContent = '✓ COPIED';
                setTimeout(() => { btn.textContent = '📋 COPY LINK'; }, 2000);
            }).catch(() => {
                const text = document.documentElement.outerHTML;
                const blob = new Blob([text], {type: 'text/html'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'conversation.html';
                a.click();
                const btn = document.querySelector('.share-btn');
                btn.textContent = '✓ SAVED';
                setTimeout(() => { btn.textContent = '📋 COPY LINK'; }, 2000);
            });
        }
    </script>
</body>
</html>"""
            
            # Write the HTML content to file
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"Updated full conversation HTML document: {html_file}")
            return True
        except Exception as e:
            import traceback
            print(f"[ERROR] Error updating conversation HTML: {e}")
            print(f"[ERROR] Traceback:")
            traceback.print_exc()
            return False

    def apply_greentext_styling(self, html_content):
        """Apply greentext styling to lines starting with '>'"""
        try:
            # Split content by lines while preserving HTML
            lines = html_content.split('\n')
            
            # Process each line that's not inside a code block
            in_code_block = False
            processed_lines = []
            
            for line in lines:
                # Check for code block start/end
                if '<pre>' in line or '<code>' in line:
                    in_code_block = True
                    processed_lines.append(line)
                    continue
                elif '</pre>' in line or '</code>' in line:
                    in_code_block = False
                    processed_lines.append(line)
                    continue
                
                # If we're in a code block, don't apply greentext styling
                if in_code_block:
                    processed_lines.append(line)
                    continue
                
                # Apply greentext styling to lines starting with '>'
                if line.strip().startswith('>'):
                    # Wrap the line in p with greentext class
                    processed_line = f'<p class="greentext">{line}</p>'
                    processed_lines.append(processed_line)
                else:
                    # No changes needed
                    processed_lines.append(line)
            
            # Join lines back
            processed_content = '\n'.join(processed_lines)
            return processed_content
            
        except Exception as e:
            print(f"Error applying greentext styling: {e}")
            return html_content

    def show_living_document_intro(self):
        """Show an introduction to the Living Document mode"""
        return

class LiminalBackroomsManager:
    """Main manager class for the Liminal Backrooms application"""
    
    def __init__(self):
        """Initialize the manager"""
        # Create the GUI
        self.app = create_gui()
        
        # Initialize the worker thread pool
        self.thread_pool = QThreadPool()
        print(f"Multithreading with maximum {self.thread_pool.maxThreadCount()} threads")
        
        # List to store workers
        self.workers = []
        
        # Initialize the application
        self.initialize()

def create_gui():
    """Create the GUI application"""
    app = QApplication(sys.argv)
    
    # Platform-specific setup for taskbar/dock icon
    if sys.platform == 'win32':
        # Windows: Set App User Model ID so taskbar shows our icon, not Python's
        import ctypes
        app_id = 'liminal.backrooms.app.1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    elif sys.platform == 'darwin':
        # macOS: Ensure app doesn't appear as a Python process
        # The .ico works but .icns is preferred for Mac - Qt handles both
        app.setApplicationName("Liminal Backrooms")
    
    # Set application icon (shows in taskbar/dock and window title)
    from PyQt6.QtGui import QIcon
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    
    # Try platform-preferred format first, fall back to .ico
    icon_path = None
    if sys.platform == 'darwin':
        # macOS prefers .icns
        icns_path = os.path.join(assets_dir, "app-icon.icns")
        if os.path.exists(icns_path):
            icon_path = icns_path
    
    if icon_path is None:
        # .ico works on all platforms (Windows, Linux, macOS)
        icon_path = os.path.join(assets_dir, "app-icon.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        # Linux: Set desktop entry name for proper taskbar grouping
        if sys.platform.startswith('linux'):
            app.setDesktopFileName("liminal-backrooms")
    else:
        print(f"Note: No app icon found at {icon_path}")
    
    # Load custom fonts (Iosevka Term for better ASCII art rendering)
    loaded_fonts = load_fonts()
    if loaded_fonts:
        print(f"Successfully loaded custom fonts: {', '.join(loaded_fonts)}")
    else:
        print("No custom fonts loaded - using system fonts")
    
    main_window = LiminalBackroomsApp()
    
    # Create conversation manager
    manager = ConversationManager(main_window)
    manager.initialize()
    main_window._conversation_manager = manager  # Store reference for auto-save recovery
    
    # Initialize debug tools if DEVELOPER_TOOLS is enabled
    debug_manager = None
    if DEVELOPER_TOOLS:
        try:
            from tools.debug_tools import DebugManager
            debug_manager = DebugManager(main_window)
            # Store reference on window to prevent garbage collection
            main_window._debug_manager = debug_manager
        except ImportError as e:
            print(f"Warning: Could not load debug tools: {e}")
    
    return main_window, app

def run_gui(main_window, app):
    """Run the GUI application"""
    # Set up global exception handler for crash logging
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        import traceback
        print("\n" + "=" * 60)
        print("[CRASH] Unhandled exception caught!")
        print("=" * 60)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("=" * 60 + "\n")
        
        # Also write to crash log file
        try:
            from datetime import datetime
            crash_log = os.path.join(LOGS_DIR, "crash_log.txt")
            with open(crash_log, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"[CRASH] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
                f.write(f"{'=' * 60}\n")
            print(f"[CRASH] Details logged to: {crash_log}")
        except:
            pass
        
        # Call the default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = global_exception_handler
    
    # Initialize freeze detector for debugging (catches UI freezes)
    freeze_detector = None
    if _FREEZE_DETECTOR_AVAILABLE:
        try:
            freeze_log = os.path.join(LOGS_DIR, "freeze_log.txt")
            crash_log = os.path.join(LOGS_DIR, "crash_log.txt")
            enable_faulthandler(crash_log)  # Enables segfault logging to logs folder
            freeze_detector = FreezeDetector(timeout_seconds=5, log_file=freeze_log)
            freeze_detector.start()
        except Exception as e:
            print(f"Warning: Could not start freeze detector: {e}")
    
    main_window.show()

    # Check for auto-save recovery after UI is ready
    if hasattr(main_window, '_conversation_manager'):
        QTimer.singleShot(500, main_window._conversation_manager.check_autosave_recovery)

    sys.exit(app.exec())

if __name__ == "__main__":
    main_window, app = create_gui()
    run_gui(main_window, app)