from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot
from src.core.models import Message
from src.services.llm_service import LLMService
from typing import List, Dict, Any

class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    response = pyqtSignal(str, str) # ai_name, content
    result = pyqtSignal(str, object) # ai_name, full_result_dict
    progress = pyqtSignal(str)

class AIWorker(QRunnable):
    """Worker thread for processing AI turns"""

    def __init__(self,
                 ai_name: str,
                 conversation: List[Message],
                 model: str,
                 system_prompt: str,
                 llm_service: LLMService,
                 is_branch: bool = False,
                 branch_id: str = None):
        super().__init__()
        self.ai_name = ai_name
        self.conversation = [msg for msg in conversation] # Shallow copy list
        self.model = model
        self.system_prompt = system_prompt
        self.llm_service = llm_service
        self.is_branch = is_branch
        self.branch_id = branch_id
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            self.signals.progress.emit(f"Processing {self.ai_name} with {self.model}...")

            # Prepare prompt logic (extracted from original main.py logic)
            # The original logic had complex "rabbithole" detection inside ai_turn.
            # I will simplify: the prompt is usually the last message,
            # but we pass the whole conversation history.

            prompt = "Let's continue."
            if self.conversation:
                 # Find last message not from this AI
                 for msg in reversed(self.conversation):
                     if msg.role == "user" or msg.ai_name != self.ai_name:
                         prompt = msg.content
                         break

            result = self.llm_service.generate_response(
                prompt=prompt,
                history=self.conversation,
                model=self.model,
                system_prompt=self.system_prompt
            )

            if "error" in result:
                self.signals.error.emit(result["error"])
            else:
                self.signals.response.emit(self.ai_name, result["content"])

                # Add metadata to result for downstream processing
                result["model"] = self.model
                result["ai_name"] = self.ai_name
                self.signals.result.emit(self.ai_name, result)

        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
