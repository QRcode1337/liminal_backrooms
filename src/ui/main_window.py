from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt
from src.ui.widgets.chat_widget import ChatWidget
from src.ui.widgets.graph_widget import GraphWidget
from src.ui.widgets.control_panel import ControlPanel
from src.core.conversation_manager import ConversationManager
from src.core.config import config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liminal Backrooms Refactored")
        self.resize(1200, 800)

        self.manager = ConversationManager()

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0,0,0,0)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Chat
        self.chat_widget = ChatWidget()
        splitter.addWidget(self.chat_widget)

        # Right: Graph
        self.graph_widget = GraphWidget()
        splitter.addWidget(self.graph_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Bottom: Control Panel
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel)

    def connect_signals(self):
        # UI -> Manager
        self.chat_widget.input_submitted.connect(self.on_input_submitted)
        self.chat_widget.rabbithole_requested.connect(self.on_rabbithole)
        self.chat_widget.fork_requested.connect(self.on_fork)
        self.graph_widget.node_clicked.connect(self.manager.switch_branch)

        # Manager -> UI
        self.manager.conversation_updated.connect(self.on_conversation_updated)
        self.manager.turn_completed.connect(self.on_turn_completed)
        self.manager.status_updated.connect(self.statusBar().showMessage)
        self.manager.error_occurred.connect(lambda msg: self.statusBar().showMessage(f"Error: {msg}"))

    def on_input_submitted(self, text):
        if not text:
            # Just start processing turns if empty input (Propagate)
            self.start_processing()
        else:
            self.manager.add_user_message(text)
            self.start_processing()

    def start_processing(self):
        cfg = self.control_panel.get_config()

        # Build AI Configs
        prompt_pair = config.system_prompt_pairs.get("Backrooms", {})

        ai1_config = {
            "model": cfg["ai1_model_id"],
            "prompt": prompt_pair.get("AI_1", "You are an AI.")
        }
        ai2_config = {
            "model": cfg["ai2_model_id"],
            "prompt": prompt_pair.get("AI_2", "You are an AI.")
        }

        self.manager.start_turn(ai1_config, ai2_config, cfg["iterations"])

    def on_conversation_updated(self, conversation, active_branch_id):
        self.chat_widget.update_display(conversation)
        self.graph_widget.update_graph(self.manager.branches, active_branch_id)

        # Update stats
        self.control_panel.stats_label.setText(f"Turns: {self.manager.turn_count}")

    def on_turn_completed(self, count):
        self.control_panel.stats_label.setText(f"Turns: {count}")

    def on_rabbithole(self, text):
        self.manager.create_branch("rabbithole", text, self.manager.active_branch_id)

    def on_fork(self, text):
        self.manager.create_branch("fork", text, self.manager.active_branch_id)
