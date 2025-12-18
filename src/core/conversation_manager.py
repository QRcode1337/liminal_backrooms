from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool
from typing import List, Dict, Optional
from src.core.models import Message, Branch
from src.core.config import config
from src.services.llm_service import LLMService
from src.services.image_service import ImageService
from src.core.worker import AIWorker
import time
import uuid

class ConversationManager(QObject):
    # Signals to update UI
    message_added = pyqtSignal(object, object) # message, branch_id
    conversation_updated = pyqtSignal(list, object) # conversation_list, branch_id
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    loading_started = pyqtSignal()
    loading_stopped = pyqtSignal()
    turn_completed = pyqtSignal(int) # turn_count

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.llm_service = LLMService()
        self.image_service = ImageService()

        self.main_conversation: List[Message] = []
        self.branches: Dict[str, Branch] = {}
        self.active_branch_id: Optional[str] = None
        self.turn_count = 0

    def add_user_message(self, content: str):
        msg = Message(role="user", content=content)
        self._append_message(msg)
        self.conversation_updated.emit(self.get_current_conversation(), self.active_branch_id)

    def _append_message(self, message: Message):
        if self.active_branch_id and self.active_branch_id in self.branches:
            self.branches[self.active_branch_id].conversation.append(message)
        else:
            self.main_conversation.append(message)

    def get_current_conversation(self) -> List[Message]:
        if self.active_branch_id and self.active_branch_id in self.branches:
            return self.branches[self.active_branch_id].conversation
        return self.main_conversation

    def start_turn(self, ai1_config: Dict, ai2_config: Dict, iterations: int):
        self.loading_started.emit()
        self.turn_count = 0
        self._process_ai_turn("AI-1", ai1_config, ai2_config, iterations)

    def _process_ai_turn(self, ai_name: str, current_config: Dict, next_config: Dict, max_iterations: int):
        # Create worker
        worker = AIWorker(
            ai_name=ai_name,
            conversation=self.get_current_conversation(),
            model=current_config["model"],
            system_prompt=current_config["prompt"],
            llm_service=self.llm_service,
            is_branch=bool(self.active_branch_id),
            branch_id=self.active_branch_id
        )

        # Connect signals
        worker.signals.result.connect(self._handle_ai_result)
        worker.signals.error.connect(self._handle_error)
        worker.signals.finished.connect(lambda: self._on_turn_finished(ai_name, next_config, current_config, max_iterations))

        self.thread_pool.start(worker)

    def _handle_ai_result(self, ai_name: str, result: Dict):
        content = result.get("content", "")
        model = result.get("model", "")

        msg = Message(
            role="assistant",
            content=content,
            ai_name=ai_name,
            model=model
        )

        # Auto-image generation check (simplified logic)
        # In a real app, this would be triggered by a specific flag in result or config
        # For now, we leave it as a placeholder or explicit call

        self._append_message(msg)
        self.conversation_updated.emit(self.get_current_conversation(), self.active_branch_id)

    def _handle_error(self, error_msg: str):
        self.error_occurred.emit(error_msg)
        # Add system message for error
        msg = Message(role="system", content=f"Error: {error_msg}", _type="error")
        self._append_message(msg)
        self.conversation_updated.emit(self.get_current_conversation(), self.active_branch_id)
        self.loading_stopped.emit()

    def _on_turn_finished(self, current_ai: str, next_config: Dict, prev_config: Dict, max_iterations: int):
        if current_ai == "AI-1":
            # Wait a bit then start AI-2
            time.sleep(config.turn_delay)
            self._process_ai_turn("AI-2", next_config, prev_config, max_iterations)
        else:
            # AI-2 finished, turn complete
            self.turn_count += 1
            self.turn_completed.emit(self.turn_count)

            if self.turn_count < max_iterations:
                time.sleep(config.turn_delay)
                self._process_ai_turn("AI-1", prev_config, next_config, max_iterations)
            else:
                self.loading_stopped.emit()
                self.status_updated.emit("Iterations completed.")

    def create_branch(self, branch_type: str, selected_text: str, parent_id: Optional[str] = None) -> str:
        new_id = str(uuid.uuid4())

        # Determine parent conversation context
        if parent_id and parent_id in self.branches:
            base_history = self.branches[parent_id].conversation
        else:
            base_history = self.main_conversation
            parent_id = "main" # explicit main parent

        # Copy history up to a point or full copy?
        # Original logic had complex truncation.
        # For simplicity, we copy the whole history for now,
        # but in "fork" mode usually you truncate.

        new_conversation = [m for m in base_history] # Deep copy of list structure

        # Add branch indicator
        indicator = Message(role="system", content=f"{branch_type}: {selected_text}", _type="branch_indicator")
        new_conversation.append(indicator)

        # Add prompt for the branch
        if branch_type == "rabbithole":
            new_conversation.append(Message(role="user", content=f"Let's explore '{selected_text}' in depth."))
        elif branch_type == "fork":
             new_conversation.append(Message(role="user", content=f"Continuing from '{selected_text}'..."))

        branch = Branch(
            id=new_id,
            type=branch_type,
            selected_text=selected_text,
            conversation=new_conversation,
            parent=parent_id,
            created_at=str(time.time())
        )

        self.branches[new_id] = branch
        self.active_branch_id = new_id

        self.conversation_updated.emit(new_conversation, new_id)
        return new_id

    def switch_branch(self, branch_id: Optional[str]):
        if branch_id == "main":
            branch_id = None

        self.active_branch_id = branch_id
        self.conversation_updated.emit(self.get_current_conversation(), branch_id)
