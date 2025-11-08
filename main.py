# main.py

import os
import time
import threading
import json
import sys
import re
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QRunnable, pyqtSlot, QThreadPool
import requests

# Load environment variables from .env file
load_dotenv()

from config import (
    TURN_DELAY,
    AI_MODELS,
    SYSTEM_PROMPT_PAIRS,
    SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT,
    SHARE_CHAIN_OF_THOUGHT
)
from shared_utils import (
    call_claude_api,
    call_openrouter_api,
    call_openai_api,
    call_replicate_api,
    call_deepseek_api,
    open_html_in_browser,
    generate_image_from_text,
    generate_video_with_sora
)
from gui import LiminalBackroomsApp

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

class Worker(QRunnable):
    """Worker thread for processing AI turns using QThreadPool"""
    
    def __init__(self, ai_name, conversation, model, system_prompt, is_branch=False, branch_id=None, gui=None):
        super().__init__()
        self.ai_name = ai_name
        self.conversation = conversation.copy()  # Make a copy to prevent race conditions
        self.model = model
        self.system_prompt = system_prompt
        self.is_branch = is_branch
        self.branch_id = branch_id
        self.gui = gui
        
        # Create signals object - ensure it stays alive
        self.signals = WorkerSignals()
        # Keep a reference to prevent garbage collection
        self._signals_ref = self.signals
    
    @pyqtSlot()
    def run(self):
        """Process the AI turn when the thread is started"""
        try:
            # Emit progress update
            if hasattr(self, 'signals') and self.signals is not None:
                self.signals.progress.emit(f"Processing {self.ai_name} turn with {self.model}...")
            
            # Process the turn
            result = ai_turn(
                self.ai_name,
                self.conversation,
                self.model,
                self.system_prompt,
                gui=self.gui
            )
            
            # Emit both the text response and the full result object
            if hasattr(self, 'signals') and self.signals is not None:
                if isinstance(result, dict):
                    response_content = result.get('content', '')
                    # Emit the simple text response for backward compatibility
                    self.signals.response.emit(self.ai_name, response_content)
                    # Also emit the full result object for HTML contribution processing
                    self.signals.result.emit(self.ai_name, result)
                else:
                    # Handle simple string responses
                    self.signals.response.emit(self.ai_name, result if result else "")
                    self.signals.result.emit(self.ai_name, {"content": result, "model": self.model})
                
                # Emit finished signal
                self.signals.finished.emit()
            
        except Exception as e:
            # Emit error signal
            if hasattr(self, 'signals') and self.signals is not None:
                self.signals.error.emit(str(e))
                # Still emit finished signal even if there's an error
                self.signals.finished.emit()

def ai_turn(ai_name, conversation, model, system_prompt, gui=None, is_branch=False, branch_output=None):
    """Execute an AI turn with the given parameters"""
    print(f"==================================================")
    print(f"Starting {model} turn ({ai_name})...")
    print(f"Current conversation length: {len(conversation)}")
    
    # HTML contributions and living document disabled
    enhanced_system_prompt = system_prompt
    
    # Get the actual model ID from the display name
    model_id = AI_MODELS.get(model, model)
    
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
            if "Rabbitholing down:" in msg.get("content", ""):
                is_rabbithole = True
                branch_text = msg.get("content", "").split('"')[1] if '"' in msg.get("content", "") else ""
                print(f"Detected rabbithole branch for: '{branch_text}'")
            elif "Forking off:" in msg.get("content", ""):
                is_fork = True
                branch_text = msg.get("content", "").split('"')[1] if '"' in msg.get("content", "") else ""
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
    seen_content = set()  # Track seen content for O(1) duplicate detection

    for msg in conversation:
        if not isinstance(msg, dict):
            # Convert plain text to dictionary
            msg = {"role": "user", "content": str(msg)}
            
        # Skip any hidden "connecting..." messages
        if msg.get("hidden") and "connect" in msg.get("content", "").lower():
            continue
            
        # Skip empty messages
        if not msg.get("content", "").strip():
            continue
            
        # Skip system messages (we already added our own above)
        if msg.get("role") == "system":
            continue
            
        # Skip special system messages (branch indicators, etc.)
        if msg.get("role") == "system" and msg.get("_type"):
            continue
            
        # Skip duplicate messages using set for O(1) lookup
        content = msg.get("content", "")
        if content in seen_content:
            continue

        seen_content.add(content)
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
            
        # Add to messages
        messages.append({
            "role": role,
            "content": msg.get("content", "")
        })
        
        print(f"Message {i} - AI: {msg.get('ai_name', 'User')} - Assigned role: {role}")
    
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
        content = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
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
    
    # Display the final processed messages for debugging
    print(f"Sending to Claude:")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    
    # Display the prompt
    print(f"--- Prompt to {model} ({ai_name}) ---")
    
    try:
        # Route Sora video models
        if model_id in ("sora-2", "sora-2-pro"):
            print(f"Using Sora Video API for model: {model_id}")
            # Use last user message as the video prompt
            prompt_content = ""
            if len(messages) > 0:
                prompt_content = messages[-1].get("content", "")
            if not isinstance(prompt_content, str) or len(prompt_content.strip()) == 0:
                prompt_content = "A short abstract motion graphic in warm colors"

            # Optional duration/size via env
            sora_seconds_env = os.getenv("SORA_SECONDS", "")
            sora_size = os.getenv("SORA_SIZE", "") or None
            try:
                sora_seconds = int(sora_seconds_env) if sora_seconds_env else None
            except ValueError:
                sora_seconds = None

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

        # Try Claude models first via Anthropic API
        if "claude" in model_id.lower() or model_id in ["anthropic/claude-3-opus-20240229", "anthropic/claude-3-sonnet-20240229", "anthropic/claude-3-haiku-20240307"]:
            print(f"Using Claude API for model: {model_id}")
            
            # CRITICAL: Make sure there are no duplicates in the messages and system prompt is included
            final_messages = []
            seen_contents = set()
            
            for msg in messages:
                # Skip empty messages
                if not msg.get("content", ""):
                    continue
                    
                # Handle system message separately
                if msg.get("role") == "system":
                    continue
                    
                # Check for duplicates by content
                content = msg.get("content", "")
                if content in seen_contents:
                    print(f"Skipping duplicate message in AI turn: {content[:30]}...")
                    continue
                
                seen_contents.add(content)
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
            
            # Call Claude API with filtered messages
            response = call_claude_api(prompt_content, context_messages, model_id, system_prompt)
            
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
                # Set up the API request
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                }
                
                # Ensure we have valid messages
                if not messages:
                    messages = [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": "Connecting..."}]
                
                data = {
                    "model": model_id,
                    "messages": messages
                }
                
                # Make the API request
                print(f"Calling OpenRouter API with model {model_id}...")
                response = requests.post(url, headers=headers, json=data)
                
                # Check for successful response
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {response.headers}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    print(f"Response data: {json.dumps(response_data, indent=2)}")
                    
                    # Make sure we have choices
                    if not response_data.get("choices"):
                        raise ValueError("No choices found in response")
                    
                    # Extract the response content
                    content = response_data["choices"][0]["message"]["content"]
                    print(f"Raw {model} Response:")
                    print("-" * 50)
                    print(content)
                    print("-" * 50)
                    
                    result = {
                        "role": "assistant",
                        "content": content,
                        "model": model,
                        "ai_name": ai_name
                    }
                    
                    return result
                else:
                    error_message = f"API request failed with status code {response.status_code}: {response.text}"
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

class ConversationManager:
    """Manages conversation processing and state"""
    def __init__(self, app):
        self.app = app
        self.workers = []  # Keep track of worker threads
        
        # Initialize the worker thread pool
        self.thread_pool = QThreadPool()
        print(f"Conversation Manager initialized with {self.thread_pool.maxThreadCount()} threads")
        
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
    
    def process_input(self, user_input=None):
        """Process the user input and generate AI responses"""
        # Get the conversation (either main or branch)
        if self.app.active_branch:
            # For branch conversations, delegate to branch processor
            self.process_branch_input(user_input)
            return
        
        # Handle main conversation processing
        if not hasattr(self.app, 'main_conversation'):
            self.app.main_conversation = []
        
        # Add user input if provided
        if user_input:
            user_message = {
                "role": "user",
                "content": user_input
            }
            self.app.main_conversation.append(user_message)
            
            # Update the conversation display with the new user message
            visible_conversation = [msg for msg in self.app.main_conversation if not msg.get('hidden', False)]
            self.app.left_pane.display_conversation(visible_conversation)
            
            # Update the HTML conversation document when user adds a message
            self.update_conversation_html(self.app.main_conversation)
        
        # Get selected models from UI
        ai_1_model = self.app.left_pane.control_panel.ai1_model_selector.currentText()
        ai_2_model = self.app.left_pane.control_panel.ai2_model_selector.currentText()
        
        # Get selected prompt pair
        selected_prompt_pair = self.app.left_pane.control_panel.prompt_pair_selector.currentText()
        
        # Get system prompts from the selected pair
        ai_1_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI_1"]
        ai_2_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI_2"]
        
        # Start loading animation
        self.app.left_pane.start_loading()
        
        # Reset turn count ONLY if this is a new conversation or explicit user input
        max_iterations = int(self.app.left_pane.control_panel.iterations_selector.currentText())
        if user_input is not None or not self.app.main_conversation:
            self.app.turn_count = 0
            print(f"MAIN: Resetting turn count - starting new conversation with {max_iterations} iterations")
        else:
            print(f"MAIN: Continuing conversation - turn {self.app.turn_count+1} of {max_iterations}")
        
        # Create worker threads for AI-1 and AI-2
        worker1 = Worker("AI-1", self.app.main_conversation, ai_1_model, ai_1_prompt, gui=self.app)
        worker2 = Worker("AI-2", self.app.main_conversation, ai_2_model, ai_2_prompt, gui=self.app)
        
        # Store workers to prevent garbage collection
        self.workers.append(worker1)
        self.workers.append(worker2)
        
        # Connect signals
        worker1.signals.response.connect(self.on_ai_response_received)
        worker1.signals.result.connect(self.on_ai_result_received)  # Connect to complete result signal
        worker1.signals.finished.connect(lambda: self.start_ai2_turn(self.app.main_conversation, worker2))
        worker1.signals.error.connect(self.on_ai_error)
        
        worker2.signals.response.connect(self.on_ai_response_received)
        worker2.signals.result.connect(self.on_ai_result_received)  # Connect to complete result signal
        worker2.signals.finished.connect(lambda: self.handle_turn_completion(max_iterations))
        worker2.signals.error.connect(self.on_ai_error)
        
        # Start AI-1's turn
        self.thread_pool.start(worker1)
    
    def start_ai2_turn(self, conversation, worker2):
        """Start AI-2's turn in the main conversation"""
        # Make sure conversation is up to date with AI-1's response
        if self.app.active_branch:
            # Get the latest branch conversation with AI-1's response already included
            branch_id = self.app.active_branch
            branch_data = self.app.branch_conversations[branch_id]
            latest_conversation = branch_data['conversation']
        else:
            # Get the latest main conversation with AI-1's response already included
            latest_conversation = self.app.main_conversation
        
        # Update worker's conversation reference to ensure it has the latest state
        # This ensures any images generated from AI-1's response are included
        worker2.conversation = latest_conversation.copy()
        
        # Add a small delay between turns
        time.sleep(TURN_DELAY)
        
        # Start AI-2's turn - the ai_turn function will properly format the context
        self.thread_pool.start(worker2)
    
    def handle_turn_completion(self, max_iterations=1):
        """Handle the completion of a full turn (both AIs)"""
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
    
    def handle_progress(self, message):
        """Handle progress update from worker"""
        print(message)
        self.app.statusBar().showMessage(message)
    
    def handle_error(self, error_message):
        """Handle error from worker"""
        print(f"Error: {error_message}")
        self.app.left_pane.append_text(f"\nError: {error_message}\n", "system")
        self.app.statusBar().showMessage(f"Error: {error_message}")
    
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
        ai_1_model = self.app.left_pane.control_panel.ai1_model_selector.currentText()
        ai_2_model = self.app.left_pane.control_panel.ai2_model_selector.currentText()
        selected_prompt_pair = self.app.left_pane.control_panel.prompt_pair_selector.currentText()
        
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
            rabbithole_prompt = f"You are interacting with another AI. IMPORTANT: Focus this response specifically on exploring and expanding upon the concept of '{selected_text}' in depth. Discuss the most interesting aspects or connections related to this concept while maintaining the tone of the conversation. No numbered lists or headings."
            ai_1_prompt = rabbithole_prompt
            ai_2_prompt = rabbithole_prompt
        else:
            # After initial exploration, revert to standard prompts
            print("Using standard prompts for continued conversation")
            ai_1_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI_1"]
            ai_2_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI_2"]
        
        # Start loading animation
        self.app.left_pane.start_loading()
        
        # Reset turn count ONLY if this is a new conversation or explicit user input
        # Don't reset during automatic iterations
        if user_input is not None or not has_ai_responses:
            self.app.turn_count = 0
            print("Resetting turn count - starting new conversation")
        
        # Get max iterations
        max_iterations = int(self.app.left_pane.control_panel.iterations_selector.currentText())
        
        # Create worker threads for AI-1 and AI-2
        worker1 = Worker("AI-1", conversation, ai_1_model, ai_1_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        worker2 = Worker("AI-2", conversation, ai_2_model, ai_2_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        
        # Store workers to prevent garbage collection
        self.workers.append(worker1)
        self.workers.append(worker2)
        
        # Connect signals
        worker1.signals.response.connect(self.on_ai_response_received)
        worker1.signals.result.connect(self.on_ai_result_received)  # Connect to complete result signal
        worker1.signals.finished.connect(lambda: self.start_ai2_turn(conversation, worker2))
        worker1.signals.error.connect(self.on_ai_error)
        
        worker2.signals.response.connect(self.on_ai_response_received)
        worker2.signals.result.connect(self.on_ai_result_received)  # Connect to complete result signal
        worker2.signals.finished.connect(lambda: self.handle_turn_completion(max_iterations))
        worker2.signals.error.connect(self.on_ai_error)
        
        # Start AI-1's turn
        self.thread_pool.start(worker1)
        
    def on_ai_response_received(self, ai_name, response_content):
        """Handle AI responses for both main and branch conversations"""
        print(f"Response received from {ai_name}: {response_content[:100]}...")
        
        # Format the AI response with proper metadata
        ai_message = {
            "role": "assistant",
            "content": response_content,
            "ai_name": ai_name,  # Add AI name to the message
            "model": self.get_model_for_ai(ai_name)  # Get the selected model name
        }
        
        # Check if we're in a branch or main conversation
        if self.app.active_branch:
            # Branch conversation
            branch_id = self.app.active_branch
            if branch_id in self.app.branch_conversations:
                branch_data = self.app.branch_conversations[branch_id]
                conversation = branch_data['conversation']
                
                # Add AI response to conversation
                conversation.append(ai_message)
                
                # Update the conversation display - filter out hidden messages
                visible_conversation = [msg for msg in conversation if not msg.get('hidden', False)]
                self.app.left_pane.display_conversation(visible_conversation, branch_data)
        else:
            # Main conversation
            if not hasattr(self.app, 'main_conversation'):
                self.app.main_conversation = []
            
            # Add AI response to main conversation
            self.app.main_conversation.append(ai_message)
            
            # Update the conversation display - filter out hidden messages
            visible_conversation = [msg for msg in self.app.main_conversation if not msg.get('hidden', False)]
            self.app.left_pane.display_conversation(visible_conversation)
        
        # Update status bar
        self.app.statusBar().showMessage(f"Received response from {ai_name}")
        
    def on_ai_result_received(self, ai_name, result):
        """Handle the complete AI result"""
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
                if hasattr(self.app.left_pane.control_panel, 'auto_image_checkbox') and self.app.left_pane.control_panel.auto_image_checkbox.isChecked():
                    self.app.left_pane.append_text("\nGenerating an image based on this response...\n", "system")
                    self.generate_and_display_image(response_content, ai_name)
        
        # Display result content
        if isinstance(result, dict):
            if "display" in result and SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT:
                self.app.left_pane.append_text(f"\n{ai_name} ({result.get('model', '')}):\n\n", "header")
                cot_parts = result['display'].split('[Final Answer]')
                if len(cot_parts) > 1:
                    self.app.left_pane.append_text(cot_parts[0].strip(), "chain_of_thought")
                    self.app.left_pane.append_text('\n\n[Final Answer]\n', "header")
                    self.app.left_pane.append_text(cot_parts[1].strip(), "ai")
                else:
                    self.app.left_pane.append_text(result['display'], "ai")
            elif "content" in result:
                self.app.left_pane.append_text(f"\n{ai_name} ({result.get('model', '')}):\n\n", "header")
                self.app.left_pane.append_text(result['content'], "ai")
            elif "image_url" in result:
                self.app.left_pane.append_text(f"\n{ai_name} ({result.get('model', '')}):\n\nGenerating an image based on the prompt...\n")
                if hasattr(self.app.left_pane, 'display_image'):
                    self.app.left_pane.display_image(result['image_url'])

        # Optionally trigger Sora video generation from AI-1 responses (no GUI embedding)
        try:
            auto_sora = os.getenv("SORA_AUTO_FROM_AI1", "0").strip() == "1"
            if auto_sora and ai_name == "AI-1" and isinstance(result, dict):
                prompt_text = result.get("content", "")
                # Require a minimally substantive prompt
                if isinstance(prompt_text, str) and len(prompt_text.strip()) > 20:
                    # Inform user in the UI synchronously (short message)
                    self.app.left_pane.append_text("\n[system] Starting Sora video job from AI-1 response...\n", "system")

                    # Read optional overrides from environment
                    sora_model = os.getenv("SORA_MODEL", "sora-2")
                    sora_seconds_env = os.getenv("SORA_SECONDS", "")
                    sora_size = os.getenv("SORA_SIZE", "") or None
                    try:
                        sora_seconds = int(sora_seconds_env) if sora_seconds_env else None
                    except ValueError:
                        sora_seconds = None

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
        
        # Update the conversation display
        visible_conversation = [msg for msg in conversation if not msg.get('hidden', False)]
        if self.app.active_branch:
            branch_id = self.app.active_branch
            branch_data = self.app.branch_conversations[branch_id]
            self.app.left_pane.display_conversation(visible_conversation, branch_data)
        else:
            self.app.left_pane.display_conversation(visible_conversation)
            
    def generate_and_display_image(self, text, ai_name):
        """Generate an image based on text and display it in the UI"""
        # Create a prompt for the image generation
        # Extract the first 100-300 characters to use as the image prompt
        max_length = min(300, len(text))
        prompt = text[:max_length].strip()
        
        # Add artistic direction to the prompt using the user's requested format
        enhanced_prompt = f"Create an image using the following text as inspiration. DO NOT repeat text in the image. Create something new. {prompt}"
        
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
                    # Add the image path to the message
                    msg["generated_image_path"] = image_path
                    print(f"Added generated image {image_path} to message from {ai_name}")
                    break
            
            # Update the conversation HTML to include the new image
            self.update_conversation_html(conversation)
            
            # Run on the main thread
            self.app.left_pane.display_image(image_path)
            
            # Notify the user
            self.app.left_pane.append_text(f"\nGenerated image saved to {image_path}\n", "system")
            
            # Do not automatically open the HTML view
            # open_html_in_browser("conversation_full.html")
    
    def get_model_for_ai(self, ai_name):
        """Get the selected model name for the AI"""
        if ai_name == "AI-1":
            return self.app.left_pane.control_panel.ai1_model_selector.currentText()
        elif ai_name == "AI-2":
            return self.app.left_pane.control_panel.ai2_model_selector.currentText()
        return ""
    
    def on_ai_error(self, error_message):
        """Handle AI errors for both main and branch conversations"""
        # Format the error message
        error_message_formatted = {
            "role": "system",
            "content": f"Error: {error_message}"
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
        self.app.right_pane.add_node(branch_id, f'🐇 {selected_text[:15]}...', 'rabbithole')
        self.app.right_pane.add_edge(parent_node, branch_id)
        
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
        self.app.right_pane.add_node(branch_id, f'🍴 {selected_text[:15]}...', 'fork')
        self.app.right_pane.add_edge(parent_node, branch_id)
        
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
        ai_1_model = self.app.left_pane.control_panel.ai1_model_selector.currentText()
        ai_2_model = self.app.left_pane.control_panel.ai2_model_selector.currentText()
        selected_prompt_pair = self.app.left_pane.control_panel.prompt_pair_selector.currentText()
        
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
        else:
            # After initial exploration, revert to standard prompts
            print("Using standard prompts for continued conversation")
            ai_1_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI_1"]
            ai_2_prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair]["AI_2"]
        
        # Start loading animation
        self.app.left_pane.start_loading()
        
        # Reset turn count ONLY if this is a new conversation or explicit user input
        # Don't reset during automatic iterations
        if user_input is not None or not has_ai_responses:
            self.app.turn_count = 0
            print("Resetting turn count - starting new conversation")
        
        # Get max iterations
        max_iterations = int(self.app.left_pane.control_panel.iterations_selector.currentText())
        
        # Create worker threads for AI-1 and AI-2
        worker1 = Worker("AI-1", conversation, ai_1_model, ai_1_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        worker2 = Worker("AI-2", conversation, ai_2_model, ai_2_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        
        # Store workers to prevent garbage collection
        self.workers.append(worker1)
        self.workers.append(worker2)
        
        # Connect signals
        worker1.signals.response.connect(self.on_ai_response_received)
        worker1.signals.result.connect(self.on_ai_result_received)  # Connect to complete result signal
        worker1.signals.finished.connect(lambda: self.start_ai2_turn(conversation, worker2))
        worker1.signals.error.connect(self.on_ai_error)
        
        worker2.signals.response.connect(self.on_ai_response_received)
        worker2.signals.result.connect(self.on_ai_result_received)  # Connect to complete result signal
        worker2.signals.finished.connect(lambda: self.handle_turn_completion(max_iterations))
        worker2.signals.error.connect(self.on_ai_error)
        
        # Start AI-1's turn
        self.thread_pool.start(worker1)

    def update_conversation_html(self, conversation):
        """Update the full conversation HTML document with all messages"""
        try:
            from datetime import datetime
            
            # Create a filename for the full conversation HTML
            html_file = "conversation_full.html"
            
            # Generate HTML content for the conversation
            html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Full Conversation</title>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; 
            padding: 0;
            line-height: 1.6; 
            color: #b8c2cc;
            background-color: #1a1a1d;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
            background-color: #202124;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            min-height: 100vh;
        }
        header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }
        h1 { 
            color: #4ec9b0; 
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #9ba1a6;
            font-size: 1.2em;
            font-weight: 300;
        }
        .message { 
            margin-bottom: 40px; 
            padding: 20px; 
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
        }
        .message-content {
            flex: 1;
            min-width: 60%;
        }
        .message-image {
            flex: 0 0 35%;
            margin-left: 20px;
            display: flex;
            align-items: flex-start;
            justify-content: center;
        }
        .message-image img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .user {
            background-color: #2a2a30;
            border-left: 4px solid #4ec9b0;
        }
        .assistant {
            background-color: #2c2c35; 
            border-left: 4px solid #569cd6;
        }
        .system {
            background-color: #262630;
            border-left: 4px solid #ce9178;
            font-style: italic;
        }
        .header { 
            font-weight: bold; 
            color: #b8c2cc; 
            margin-bottom: 10px; 
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .timestamp {
            font-size: 0.8em;
            color: #9ba1a6;
            font-weight: normal;
        }
        .content {
            white-space: pre-wrap;
        }
        /* Greentext styling */
        .greentext {
            color: #789922;
            font-family: monospace;
        }
        p {
            margin: 0.5em 0;
        }
        code { 
            background: #333; 
            padding: 2px 4px; 
            border-radius: 3px; 
            font-family: 'Consolas', 'Monaco', monospace;
            color: #dcdcaa;
        }
        pre { 
            background: #2d2d2d; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto; 
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 20px 0;
            border: 1px solid #444;
            color: #d4d4d4;
        }
        .html-contribution {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px dashed #444;
            color: #569cd6;
            font-style: italic;
        }
        footer {
            margin-top: 50px;
            text-align: center;
            color: #9ba1a6;
            font-size: 0.9em;
            padding-top: 20px;
            border-top: 1px solid #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Liminal Conversation</h1>
            <p class="subtitle"></p>
        </header>
        
        <div id="conversation">"""
            
            # Add each message to the HTML content
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")
                ai_name = msg.get("ai_name", "")
                model = msg.get("model", "")
                timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
                
                # Skip special system messages or empty messages
                if role == "system" and msg.get("_type") == "branch_indicator":
                    continue
                if not content.strip():
                    continue
                
                # Process content to properly format code blocks and add greentext styling
                processed_content = self.app.left_pane.process_content_with_code_blocks(content)
                
                # Apply greentext styling to lines starting with '>'
                processed_content = self.apply_greentext_styling(processed_content)
                
                # Message class based on role
                message_class = role
                
                # Check if this message has an associated image
                has_image = False
                image_path = None
                
                # Check for image in this message
                if hasattr(msg, "get") and callable(msg.get):
                    image_path = msg.get("generated_image_path", None)
                    if image_path:
                        has_image = True
                
                # Start message div
                html_content += f'\n        <div class="message {message_class}">'
                
                # Open content div
                html_content += f'\n            <div class="message-content">'
                
                # Add header for assistant messages
                if role == "assistant":
                    display_name = ai_name
                    if model:
                        display_name += f" ({model})"
                    html_content += f'\n                <div class="header">{display_name} <span class="timestamp">{timestamp}</span></div>'
                elif role == "user":
                    html_content += f'\n                <div class="header">User <span class="timestamp">{timestamp}</span></div>'
                
                # Add message content
                html_content += f'\n                <div class="content">{processed_content}</div>'
                
                # Removed HTML contribution artifact block
                
                # Close content div
                html_content += '\n            </div>'
                
                # Add image if present
                if has_image and image_path:
                    # Convert Windows path format to web format if needed
                    web_path = image_path.replace('\\', '/')
                    html_content += f'\n            <div class="message-image">'
                    html_content += f'\n                <img src="{web_path}" alt="Generated image" />'
                    html_content += f'\n            </div>'
                
                # Close message div
                html_content += '\n        </div>'
            
            # Close HTML document
            html_content += """
        </div>
        
        <footer>
            <p>Generated by Liminal Backrooms</p>
        </footer>
    </div>
</body>
</html>"""
            
            # Write the HTML content to file
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"Updated full conversation HTML document: {html_file}")
            return True
        except Exception as e:
            print(f"Error updating conversation HTML: {e}")
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
    main_window = LiminalBackroomsApp()
    
    # Create conversation manager
    manager = ConversationManager(main_window)
    manager.initialize()
    
    return main_window, app

def run_gui(main_window, app):
    """Run the GUI application"""
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main_window, app = create_gui()
    run_gui(main_window, app)