import os
import json
import logging
import threading
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

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

class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    response = pyqtSignal(str, str)
    result = pyqtSignal(str, object)  # Signal for complete result object
    progress = pyqtSignal(str)
    streaming_chunk = pyqtSignal(str, str)  # Signal for streaming tokens: (ai_name, chunk)


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

        # Create signals object
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Process the AI turn when the thread is started"""
        print(f"[Worker] >>> Starting run() for {self.ai_name} ({self.model})")
        try:
            # Emit progress update
            self.signals.progress.emit(f"Processing {self.ai_name} turn with {self.model}...")

            # Define streaming callback
            def stream_chunk(chunk: str):
                self.signals.streaming_chunk.emit(self.ai_name, chunk)

            # Process the turn with streaming
            from src.core.engine import ai_turn

            print(f"[Worker] Calling ai_turn for {self.ai_name}...")
            result = ai_turn(
                self.ai_name,
                self.conversation,
                self.model,
                self.system_prompt,
                gui=self.gui,
                streaming_callback=stream_chunk
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
