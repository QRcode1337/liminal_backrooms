import os
import time
import json
import threading
from PyQt6.QtCore import QThreadPool, pyqtSignal, QObject

from src.core.config import (
    TURN_DELAY,
    AI_MODELS,
    SYSTEM_PROMPT_PAIRS,
    SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT,
    SHARE_CHAIN_OF_THOUGHT
)
from src.core.command_parser import parse_commands, AgentCommand
from src.core.worker import Worker, WorkerSignals
from src.services.media_service import generate_image_from_text, generate_video_with_sora
from src.services.html_generator import update_conversation_html
from src.utils.shared_utils import open_html_in_browser, web_search
from src.core.session_manager import SessionManager

class ImageUpdateSignals(QObject):
    """Signals for updating UI with generated images from background threads"""
    image_ready = pyqtSignal(dict, str)  # (image_message, image_path)

class VideoUpdateSignals(QObject):
    """Signals for updating UI with generated videos from background threads"""
    video_ready = pyqtSignal(str, str)  # (video_path, prompt)

class ConversationManager:
    """Manages conversation processing and state"""
    def __init__(self, app):
        self.app = app
        self.workers = []  # Keep track of worker threads
        
        # Initialize the worker thread pool
        self.thread_pool = QThreadPool()
        print(f"Conversation Manager initialized with {self.thread_pool.maxThreadCount()} threads")
        
        # Set up image update signals for thread-safe UI updates
        self.image_signals = ImageUpdateSignals()
        self.image_signals.image_ready.connect(self._on_image_ready)
        
        # Set up video update signals for thread-safe UI updates
        self.video_signals = VideoUpdateSignals()
        self.video_signals.video_ready.connect(self._on_video_ready)
        
        # Store per-AI prompt additions (self-modifications)
        self.ai_prompt_additions = {}
        
        # Store per-AI temperature settings (default is 1.0)
        self.ai_temperatures = {}

        # Initialize Session Manager
        self.session_manager = SessionManager()
        
    def _on_video_ready(self, video_path: str, prompt: str):
        """Handle video ready signal - runs on main thread"""
        try:
            print(f"[Agent] Video ready, updating UI: {video_path}")
            # Update the video preview panel
            if hasattr(self.app, 'right_sidebar') and hasattr(self.app.right_sidebar, 'update_video_preview'):
                self.app.right_sidebar.update_video_preview(video_path)
            
            # Update status bar notification with prompt (truncated for display)
            if hasattr(self.app, 'notification_label'):
                # Truncate long prompts for status bar
                display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
                self.app.notification_label.setText(f"🎬 Video completed: {display_prompt}")
        except Exception as e:
            print(f"[Agent] Error handling video ready: {e}")
            import traceback
            traceback.print_exc()
        
    def _on_image_ready(self, image_message: dict, image_path: str):
        """Handle image ready signal - runs on main thread"""
        try:
            # Add image to conversation
            self.app.main_conversation.append(image_message)
            
            # Update the conversation display
            self.app.left_pane.conversation = self.app.main_conversation
            self.app.left_pane.render_conversation()
            
            # Update the image preview panel
            if hasattr(self.app.right_sidebar, 'update_image_preview'):
                self.app.right_sidebar.update_image_preview(image_path)
            
            # Update status bar notification
            ai_name = image_message.get('ai_name', 'AI')
            if hasattr(self.app, 'notification_label'):
                self.app.notification_label.setText(f"🖼️ {ai_name} generated an image")
            
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
                
            self.app.main_conversation.append(user_message)
            
            # Update the conversation display with the new user message
            visible_conversation = [msg for msg in self.app.main_conversation if not msg.get('hidden', False)]
            self.app.left_pane.display_conversation(visible_conversation)
            
            # Update the HTML conversation document when user adds a message
            self.update_conversation_html(self.app.main_conversation)
        
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
                    "hidden": False
                }
                self.app.main_conversation.append(mute_notification)
                # Remove from muted set (only skip one turn)
                muted_ais.discard(ai_name)
                continue
            
            model = self.get_model_for_ai(i)
            prompt = SYSTEM_PROMPT_PAIRS[selected_prompt_pair][ai_name]
            
            worker = Worker(ai_name, self.app.main_conversation, model, prompt, gui=self.app)
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
            
            # Now update the selector to reflect all pending AIs joining
            # This is the correct time to update - when they actually join, not when invited
            final_count = int(self.app.right_sidebar.control_panel.num_ais_selector.currentText()) + len(pending)
            self.app.right_sidebar.control_panel.num_ais_selector.setCurrentText(str(final_count))
            print(f"[Agent] Updated AI count to {final_count}")
            
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
                
                worker = Worker(ai_name, conversation.copy(), model, prompt, gui=self.app)
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
        
        # Create worker threads for AI-1, AI-2, and AI-3
        worker1 = Worker("AI-1", conversation, ai_1_model, ai_1_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        worker2 = Worker("AI-2", conversation, ai_2_model, ai_2_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        worker3 = Worker("AI-3", conversation, ai_3_model, ai_3_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        
        # Connect signals for worker1
        worker1.signals.response.connect(self.on_ai_response_received)
        worker1.signals.result.connect(self.on_ai_result_received)
        worker1.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker1.signals.finished.connect(lambda: self.start_ai2_turn(conversation, worker2))
        worker1.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker2
        worker2.signals.response.connect(self.on_ai_response_received)
        worker2.signals.result.connect(self.on_ai_result_received)
        worker2.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker2.signals.finished.connect(lambda: self.start_ai3_turn(conversation, worker3))
        worker2.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker3
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
        
        # Initialize buffer for this AI if needed
        if ai_name not in self._streaming_buffers:
            self._streaming_buffers[ai_name] = ""
            # Add a header to show this AI is responding
            ai_number = int(ai_name.split('-')[1]) if '-' in ai_name else 1
            model_name = self.get_model_for_ai(ai_number)
            self.app.left_pane.append_text(f"\n{ai_name} ({model_name}):\n\n", "header")
            
            # Calculate and update latency on first chunk
            if hasattr(self, '_request_start_time') and hasattr(self.app, 'update_signal_latency'):
                latency_ms = int((time.time() - self._request_start_time) * 1000)
                self.app.update_signal_latency(latency_ms)
        
        # Append chunk to buffer
        self._streaming_buffers[ai_name] += chunk
        
        # Display the chunk in the GUI
        self.app.left_pane.append_text(chunk, "ai")
    
    def on_ai_response_received(self, ai_name, response_content):
        """Handle AI responses for both main and branch conversations"""
        print(f"Response received from {ai_name}: {response_content[:100]}...")
        
        # Clear streaming buffer for this AI
        if hasattr(self, '_streaming_buffers') and ai_name in self._streaming_buffers:
            del self._streaming_buffers[ai_name]
        
        # Parse response for agentic commands
        cleaned_content, commands = parse_commands(response_content)
        
        # Execute any commands found and add notifications to conversation
        if commands:
            print(f"[Agent] Found {len(commands)} command(s) in {ai_name}'s response")
            
            for cmd in commands:
                success, message = self.execute_agent_command(cmd, ai_name)
                print(f"[Agent] Command result: success={success}, message={message}")
                
                # Add notification as a system message in the conversation
                notification_msg = {
                    "role": "system",
                    "content": message,
                    "_type": "agent_notification"
                }
                
                # Add to the correct conversation
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
        response_content = cleaned_content if cleaned_content else response_content
        
        # Extract AI number from ai_name (e.g., "AI-1" -> 1)
        ai_number = int(ai_name.split('-')[1]) if '-' in ai_name else 1
        
        # Format the AI response with proper metadata
        ai_message = {
            "role": "assistant",
            "content": response_content,
            "ai_name": ai_name,  # Add AI name to the message
            "model": self.get_model_for_ai(ai_number)  # Get the selected model name
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
                
                # Debug: Check for notifications
                notifications = [m for m in conversation if m.get('_type') == 'agent_notification']
                print(f"[Debug] Branch conversation has {len(notifications)} notifications before display")
                
                # Update the conversation display - filter out hidden messages
                visible_conversation = [msg for msg in conversation if not msg.get('hidden', False)]
                self.app.left_pane.display_conversation(visible_conversation, branch_data)
        else:
            # Main conversation
            if not hasattr(self.app, 'main_conversation'):
                self.app.main_conversation = []
            
            # Add AI response to main conversation
            self.app.main_conversation.append(ai_message)
            
            # Debug: Check for notifications
            notifications = [m for m in self.app.main_conversation if m.get('_type') == 'agent_notification']
            print(f"[Debug] Main conversation has {len(notifications)} notifications before display")
            
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
                if hasattr(self.app.right_sidebar.control_panel, 'auto_image_checkbox') and self.app.right_sidebar.control_panel.auto_image_checkbox.isChecked():
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

                    # Use config values with env var override
                    from src.core.config import SORA_SECONDS, SORA_SIZE
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
                    # Add the image path to the message
                    msg["generated_image_path"] = image_path
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
    
    def execute_agent_command(self, command: AgentCommand, ai_name: str) -> tuple[bool, str]:
        """
        Execute an agentic command from an AI response.
        """
        action = command.action
        params = command.params
        
        print(f"[Agent] Executing command: {action} from {ai_name}")
        print(f"[Agent] Params: {params}")
        
        if action == 'image':
            return self._execute_image_command(params.get('prompt', ''), ai_name)
        elif action == 'video':
            return self._execute_video_command(params.get('prompt', ''), ai_name)
        elif action == 'search':
            return self._execute_search_command(params.get('query', ''), ai_name)
        elif action == 'prompt':
            return self._execute_prompt_command(params.get('text', ''), ai_name)
        elif action == 'temperature':
            return self._execute_temperature_command(params.get('value', ''), ai_name)
        elif action == 'add_ai':
            return self._execute_add_ai_command(params.get('model', ''), params.get('persona'), ai_name)
        elif action == 'remove_ai':
            return self._execute_remove_ai_command(params.get('target', ''), ai_name)
        elif action == 'list_models':
            return self._execute_list_models_command(ai_name)
        elif action == 'mute_self':
            return self._execute_mute_command(ai_name)
        else:
            return False, f"Unknown command: {action}"
    
    def _execute_image_command(self, prompt: str, ai_name: str, model_name: str = None) -> tuple[bool, str]:
        """Execute an image generation command."""
        if not prompt or len(prompt.strip()) < 5:
            return False, "Image prompt too short"
        
        # Get model name if not provided
        if not model_name:
            ai_number = int(ai_name.split('-')[1]) if '-' in ai_name else 1
            model_name = self.get_model_for_ai(ai_number)
        
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
                                    "text": f"[{ai_name} ({model_name})]: !image \"{prompt}\"\n<image attached>"
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
                            "_type": "generated_image",
                            "ai_name": ai_name,
                            "model": model_name
                        }
                        
                        # Emit signal to update UI on main thread
                        self.image_signals.image_ready.emit(image_message, image_path)
                        
                    except Exception as e:
                        print(f"[Agent] Could not add image to context: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"[Agent] Image generation failed: {error}")
            except Exception as e:
                print(f"[Agent] Image generation exception: {e}")
        
        threading.Thread(target=_run_image_job, daemon=True).start()
        return True, f"🎨 [{ai_name} ({model_name})]: !image \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\" (generating...)"
    
    def _execute_video_command(self, prompt: str, ai_name: str) -> tuple[bool, str]:
        """Execute a video generation command."""
        if not prompt or len(prompt.strip()) < 5:
            return False, "Video prompt too short"
        
        print(f"[Agent] Generating video for {ai_name}: {prompt[:100]}...")
        
        # Run video generation in background thread to avoid blocking
        import threading
        from src.core.config import SORA_SECONDS, SORA_SIZE
        
        def _run_video_job():
            sora_model = os.getenv("SORA_MODEL", "sora-2")
            
            # Use config values, with env var override
            sora_seconds = int(os.getenv("SORA_SECONDS", str(SORA_SECONDS)))
            sora_size = os.getenv("SORA_SIZE", SORA_SIZE) or None
            
            print(f"[Agent] Sora settings: seconds={sora_seconds}, size={sora_size}")
            
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
                        self.video_signals.video_ready.emit(str(video_path), prompt)
            else:
                print(f"[Agent] Video failed: {result.get('error')}")
        
        threading.Thread(target=_run_video_job, daemon=True).start()
        return True, f"🎬 [{ai_name}]: !video \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\" (generating...)"
    
    def _execute_add_ai_command(self, model_name: str, persona: str, requesting_ai: str) -> tuple[bool, str]:
        """Execute an add AI participant command."""
        # Get the base number of AIs from the selector (this is the starting count for this round)
        # We DON'T update the selector until the AI actually joins - just track pending count
        base_num_ais = int(self.app.right_sidebar.control_panel.num_ais_selector.currentText())
        pending_count = len(getattr(self, '_pending_ais', []))
        
        # The effective count is base + pending (selector is NOT updated during pending phase)
        effective_count = base_num_ais + pending_count
        
        if effective_count >= 5:
            return False, "Maximum of 5 AIs already reached"
        
        new_num = effective_count + 1
        
        # Try to set the model for the new AI slot
        actual_model = model_name  # Track what model was actually set
        selector = getattr(self.app.right_sidebar.control_panel, f'ai{new_num}_model_selector', None)
        if selector:
            # Find if the requested model exists in the selector
            found = False
            for i in range(selector.count()):
                if model_name.lower() in selector.itemText(i).lower():
                    selector.setCurrentIndex(i)
                    actual_model = selector.itemText(i)
                    found = True
                    break
            if not found:
                actual_model = selector.currentText()  # Use whatever is default
        
        # Store persona for later use (could be used to modify system prompt)
        if persona:
            if not hasattr(self.app, 'custom_personas'):
                self.app.custom_personas = {}
            self.app.custom_personas[f"AI-{new_num}"] = persona
        
        # Track this AI as pending so it can join the current round
        if not hasattr(self, '_pending_ais'):
            self._pending_ais = []
        
        # Check if this model is already an active AI (deduplication)
        for i in range(1, base_num_ais + 1):
            existing_selector = getattr(self.app.right_sidebar.control_panel, f'ai{i}_model_selector', None)
            if existing_selector:
                existing_model = existing_selector.currentText()
                if actual_model.lower() in existing_model.lower() or existing_model.lower() in actual_model.lower():
                    print(f"[Agent] {actual_model} already active as AI-{i}, skipping duplicate")
                    return True, f"✨ {actual_model} is already in the conversation as AI-{i}"
        
        # Check if this model was already invited this round (pending deduplication)
        already_pending = any(p['model'].lower() in actual_model.lower() or actual_model.lower() in p['model'].lower() for p in self._pending_ais)
        if already_pending:
            print(f"[Agent] {actual_model} already invited this round, skipping duplicate")
            return True, f"✨ {actual_model} was already invited (by another AI)"
        
        # DON'T update the selector here - it will be updated when the AI actually joins
        # This prevents double-counting when multiple AIs are invited in the same round
        
        self._pending_ais.append({
            'ai_name': f"AI-{new_num}",
            'ai_number': new_num,
            'model': actual_model,
            'persona': persona,
            'invited_by': requesting_ai
        })
        print(f"[Agent] Queued AI-{new_num} ({actual_model}) to join current round")
        print(f"[Agent] Current pending queue: {[p['ai_name'] + ' -> ' + p['model'] for p in self._pending_ais]}")
        
        # Create a friendly notification message that shows the command syntax
        if persona:
            return True, f"✨ [{requesting_ai}]: !add_ai \"{actual_model}\" \"{persona}\""
        else:
            return True, f"✨ [{requesting_ai}]: !add_ai \"{actual_model}\""
    
    def _execute_remove_ai_command(self, target: str, requesting_ai: str) -> tuple[bool, str]:
        """Execute a remove AI participant command (requires consensus in future)."""
        # For now, just log the request - could implement voting system later
        return False, f"🗳️ {requesting_ai} voted to remove {target} (consensus not yet implemented)"
    
    def _execute_list_models_command(self, ai_name: str) -> tuple[bool, str]:
        """Execute a list models command - returns available models for invitation."""
        try:
            models_file = os.path.join(os.path.dirname(__file__), 'available_models.txt')
            if os.path.exists(models_file):
                with open(models_file, 'r', encoding='utf-8') as f:
                    models_content = f.read()
                print(f"[Agent] {ai_name} queried available models")
                return True, f"📋 Available models:\n{models_content}"
            else:
                return False, "Models list not found"
        except Exception as e:
            return False, f"Error reading models: {e}"
    
    def _execute_mute_command(self, ai_name: str) -> tuple[bool, str]:
        """Execute a mute self command - AI skips next turn."""
        if not hasattr(self.app, 'muted_ais'):
            self.app.muted_ais = set()
        
        self.app.muted_ais.add(ai_name)
        return True, f"🔇 [{ai_name}]: !mute_self"
    
    def _execute_prompt_command(self, text: str, ai_name: str) -> tuple[bool, str]:
        """Execute a prompt addition command - AI appends to their own system prompt."""
        if not text or len(text.strip()) < 3:
            return False, "Prompt text too short"
        
        # Initialize if needed
        if ai_name not in self.ai_prompt_additions:
            self.ai_prompt_additions[ai_name] = []
        
        # Add the new prompt text
        self.ai_prompt_additions[ai_name].append(text.strip())
        
        print(f"[Agent] {ai_name} added to their prompt: {text[:50]}...")
        print(f"[Agent] {ai_name} now has {len(self.ai_prompt_additions[ai_name])} prompt additions")
        
        # Add a subtle notification to conversation context (visible to other AIs)
        context_notification = {
            "role": "user",
            "content": f"[{ai_name} modified their system prompt]",
            "_type": "system_notification"
        }
        self.app.main_conversation.append(context_notification)
        
        # Show full untruncated text in notification (only human sees this, not other AIs)
        return True, f"💭 [{ai_name}]: !prompt \"{text}\""
    
    def get_prompt_additions_for_ai(self, ai_name: str) -> str:
        """Get all prompt additions for a specific AI as a formatted string."""
        if ai_name not in self.ai_prompt_additions or not self.ai_prompt_additions[ai_name]:
            return ""
        
        additions = self.ai_prompt_additions[ai_name]
        return "\n\n[Your remembered insights/perspectives]:\n- " + "\n- ".join(additions)
    
    def _execute_temperature_command(self, value: str, ai_name: str) -> tuple[bool, str]:
        """Execute a temperature modification command - AI sets their own sampling temperature."""
        try:
            temp = float(value)
            if temp < 0 or temp > 2:
                return False, f"Temperature must be between 0 and 2 (got {temp})"
            
            self.ai_temperatures[ai_name] = temp
            print(f"[Agent] {ai_name} set their temperature to {temp}")
            
            # Add a subtle notification to conversation context (visible to other AIs)
            context_notification = {
                "role": "user",
                "content": f"[{ai_name} adjusted their temperature]",
                "_type": "system_notification"
            }
            self.app.main_conversation.append(context_notification)
            
            # Show the actual value in notification for human
            return True, f"🌡️ [{ai_name}]: !temperature {temp}"
        except (ValueError, TypeError):
            return False, f"Invalid temperature value: {value}"
    
    def get_temperature_for_ai(self, ai_name: str) -> float:
        """Get the temperature setting for a specific AI (default 1.0)."""
        return self.ai_temperatures.get(ai_name, 1.0)
    
    def _execute_search_command(self, query: str, ai_name: str) -> tuple[bool, str]:
        """Execute a web search command and inject results into conversation."""
        if not query or len(query.strip()) < 3:
            return False, "Search query too short"
        
        # Get model name for the AI
        ai_number = int(ai_name.split('-')[1]) if '-' in ai_name else 1
        model_name = self.get_model_for_ai(ai_number)
        
        print(f"[Agent] Searching for {ai_name} ({model_name}): {query}")
        
        result = web_search(query, max_results=5)
        
        if result.get("success"):
            results = result.get("results", [])
            if results:
                # Format results for conversation context
                formatted = f"🔍 [{ai_name} ({model_name})]: !search \"{query}\"\n\n**Search Results:**\n"
                for i, r in enumerate(results, 1):
                    formatted += f"\n{i}. **{r['title']}**\n"
                    formatted += f"   {r['snippet']}\n"
                    formatted += f"   Source: {r['url']}\n"
                
                # Add search results to conversation so all AIs can see them
                search_message = {
                    "role": "user",
                    "content": formatted,
                    "_type": "search_result",
                    "hidden": False
                }
                self.app.main_conversation.append(search_message)
                
                # Also display in the UI
                self.app.left_pane.append_text(f"\n{formatted}\n", "system")
                
                return True, f"🔍 [{ai_name}]: !search \"{query}\" (found {len(results)} results)"
            else:
                return False, f"No results found for: {query}"
        else:
            error = result.get('error', 'Unknown error')
            return False, f"Search failed: {error}"
    
    def get_model_for_ai(self, ai_number):
        """Get the selected model name for the AI by number (1-5)"""
        selectors = {
            1: self.app.right_sidebar.control_panel.ai1_model_selector,
            2: self.app.right_sidebar.control_panel.ai2_model_selector,
            3: self.app.right_sidebar.control_panel.ai3_model_selector,
            4: self.app.right_sidebar.control_panel.ai4_model_selector,
            5: self.app.right_sidebar.control_panel.ai5_model_selector
        }
        return selectors.get(ai_number, selectors[1]).currentText()
    
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
        
        # Create worker threads for AI-1, AI-2, and AI-3
        worker1 = Worker("AI-1", conversation, ai_1_model, ai_1_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        worker2 = Worker("AI-2", conversation, ai_2_model, ai_2_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        worker3 = Worker("AI-3", conversation, ai_3_model, ai_3_prompt, is_branch=True, branch_id=branch_id, gui=self.app)
        
        # Connect signals for worker1
        worker1.signals.response.connect(self.on_ai_response_received)
        worker1.signals.result.connect(self.on_ai_result_received)
        worker1.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker1.signals.finished.connect(lambda: self.start_ai2_turn(conversation, worker2))
        worker1.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker2
        worker2.signals.response.connect(self.on_ai_response_received)
        worker2.signals.result.connect(self.on_ai_result_received)
        worker2.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker2.signals.finished.connect(lambda: self.start_ai3_turn(conversation, worker3))
        worker2.signals.error.connect(self.on_ai_error)
        
        # Connect signals for worker3
        worker3.signals.response.connect(self.on_ai_response_received)
        worker3.signals.result.connect(self.on_ai_result_received)
        worker3.signals.streaming_chunk.connect(self.on_streaming_chunk)
        worker3.signals.finished.connect(lambda: self.handle_turn_completion(max_iterations))
        worker3.signals.error.connect(self.on_ai_error)
        
        # Start AI-1's turn
        self.thread_pool.start(worker1)

    def update_conversation_html(self, conversation):
        """Update the full conversation HTML document with all messages"""
        return update_conversation_html(conversation)

    def start_ai2_turn(self, conversation, worker):
        time.sleep(TURN_DELAY)
        self.thread_pool.start(worker)

    def start_ai3_turn(self, conversation, worker):
        time.sleep(TURN_DELAY)
        self.thread_pool.start(worker)
