import os
import json
import logging
import re
from src.core.config import AI_MODELS
from src.services.llm_service import (
    call_claude_api,
    call_openrouter_api,
    call_openai_api,
    call_replicate_api,
    call_deepseek_api,
)
from src.services.media_service import generate_video_with_sora
from src.core.config import SORA_SECONDS, SORA_SIZE

def ai_turn(ai_name, conversation, model, system_prompt, gui=None, is_branch=False, branch_output=None, streaming_callback=None):
    """Execute an AI turn with the given parameters

    Args:
        streaming_callback: Optional function(chunk: str) to call with each streaming token
    """
    print(f"==================================================")
    print(f"Starting {model} turn ({ai_name})...")
    print(f"Current conversation length: {len(conversation)}")

    # HTML contributions and living document disabled
    enhanced_system_prompt = system_prompt

    # Get the actual model ID from the display name
    model_id = AI_MODELS.get(model, model)

    # Prepend model identity to system prompt so AI knows who it is
    enhanced_system_prompt = f"You are {ai_name} ({model}).\n\n{enhanced_system_prompt}"

    # Apply any self-added prompt additions for this AI
    # Also get custom temperature setting
    ai_temperature = 1.0  # Default
    if gui and hasattr(gui, 'conversation_manager') and gui.conversation_manager:
        prompt_additions = gui.conversation_manager.get_prompt_additions_for_ai(ai_name)
        if prompt_additions:
            enhanced_system_prompt += prompt_additions
            print(f"[Prompt] Applied prompt additions for {ai_name}")

        # Get custom temperature if set
        ai_temperature = gui.conversation_manager.get_temperature_for_ai(ai_name)
        if ai_temperature != 1.0:
            print(f"[Temperature] Using custom temperature {ai_temperature} for {ai_name}")

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
            # Use the model name (e.g., "Claude 4.5 Sonnet") if available, otherwise fall back to ai_name or "User"
            speaker_name = msg.get("model") or msg.get("ai_name", "User")

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

        # Route Claude models through OpenRouter instead of direct Anthropic API
        # This avoids issues with image handling differences between the APIs
        # Set to False to use OpenRouter for Claude (recommended for image support)
        USE_DIRECT_ANTHROPIC_API = False

        if USE_DIRECT_ANTHROPIC_API and ("claude" in model_id.lower() or model_id in ["anthropic/claude-3-opus-20240229", "anthropic/claude-3-sonnet-20240229", "anthropic/claude-3-haiku-20240307"]):
            print(f"Using Claude API for model: {model_id}")

            # CRITICAL: Make sure there are no duplicates in the messages and system prompt is included
            final_messages = []
            seen_contents = set()

            for msg in messages:
                # Skip empty messages - handle both string and list content
                content = msg.get("content", "")
                is_empty = False
                if isinstance(content, list):
                    # For structured content, check if all parts are empty
                    text_parts = [part.get('text', '').strip() for part in content if part.get('type') == 'text']
                    has_image = any(part.get('type') == 'image' for part in content)
                    is_empty = not text_parts and not has_image
                elif isinstance(content, str):
                    is_empty = not content
                else:
                    is_empty = not content

                if is_empty:
                    continue

                # Handle system message separately
                if msg.get("role") == "system":
                    continue

                # Check for duplicates by content - create hashable representation
                content = msg.get("content", "")

                # Create a hashable content_hash for duplicate detection
                if isinstance(content, list):
                    # For structured messages, use text parts for hash
                    text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                    content_hash = ''.join(text_parts)
                elif isinstance(content, str):
                    content_hash = content
                else:
                    content_hash = str(content) if content else ""

                if content_hash and content_hash in seen_contents:
                    print(f"Skipping duplicate message in AI turn: {content_hash[:30]}...")
                    continue

                if content_hash:
                    seen_contents.add(content_hash)
                final_messages.append(msg)

            # Ensure we have at least one message
            if not final_messages:
                print("Warning: No messages left after filtering. Adding a default message.")
                final_messages.append({"role": "user", "content": "Connecting..."})

            # Get the prompt content safely
            prompt_content = ""
            if len(final_messages) > 0:
                prompt_content = final_messages[-1].get("content", "")
                # Use all messages except the last one as context
                context_messages = final_messages[:-1]
            else:
                context_messages = []
                prompt_content = "Connecting..."  # Default fallback

            # Call Claude API with filtered messages (with streaming if callback provided)
            response = call_claude_api(prompt_content, context_messages, model_id, system_prompt, stream_callback=streaming_callback, temperature=ai_temperature)

            return {
                "role": "assistant",
                "content": response,
                "model": model,
                "ai_name": ai_name
            }

        # Check for DeepSeek models to use Replicate via DeepSeek API function
        if "deepseek" in model.lower():
            print(f"Using Replicate API for DeepSeek model: {model_id}")

            # Ensure we have at least one message for the prompt
            if len(messages) > 0:
                prompt_content = messages[-1].get("content", "")
                context_messages = messages[:-1]
            else:
                prompt_content = "Connecting..."
                context_messages = []

            response = call_deepseek_api(prompt_content, context_messages, model_id, system_prompt)

            # Ensure response has the required format for the Worker class
            if isinstance(response, dict) and 'content' in response:
                # Add model info to the response
                response['model'] = model
                response['role'] = 'assistant'
                response['ai_name'] = ai_name

                # Check for HTML contribution
                if "html_contribution" in response:
                    html_contribution = response["html_contribution"]

                    # Don't update HTML document here - we'll do it in on_ai_result_received
                    # Just add indicator to the conversation part
                    response["content"] += "\n\n..."
                    if "display" in response:
                        response["display"] += "\n\n..."

                return response
            else:
                # Create a formatted response if not already in the right format
                return {
                    "role": "assistant",
                    "content": str(response) if response else "No response from model",
                    "model": model,
                    "ai_name": ai_name,
                    "display": str(response) if response else "No response from model"
                }

        # Use OpenRouter for all other models
        else:
            print(f"Using OpenRouter API for model: {model_id}")

            try:
                # Ensure we have valid messages
                if len(messages) > 0:
                    prompt_content = messages[-1].get("content", "")
                    context_messages = messages[:-1]
                else:
                    prompt_content = "Connecting..."
                    context_messages = []

                # Call OpenRouter API with streaming support
                response = call_openrouter_api(prompt_content, context_messages, model_id, system_prompt, stream_callback=streaming_callback, temperature=ai_temperature)

                # Avoid printing full response which could be large
                response_preview = str(response)[:200] + "..." if response and len(str(response)) > 200 else response
                print(f"Raw {model} Response: {response_preview}")

                result = {
                    "role": "assistant",
                    "content": response,
                    "model": model,
                    "ai_name": ai_name
                }

                return result
            except Exception as e:
                error_message = f"Error making API request: {str(e)}"
                print(f"Error: {error_message}")
                print(f"Error type: {type(e)}")

                # Create an error response
                result = {
                    "role": "system",
                    "content": f"Error: {error_message}",
                    "model": model,
                    "ai_name": ai_name
                }

                # Return the error result
                return result

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
