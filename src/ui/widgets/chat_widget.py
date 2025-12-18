from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QImage, QPixmap, QTextCharFormat, QColor, QAction
from src.core.models import Message
import os

class ChatWidget(QWidget):
    input_submitted = pyqtSignal(str)
    rabbithole_requested = pyqtSignal(str)
    fork_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header
        self.header_label = QLabel("Conversation")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #D4D4D4;")
        layout.addWidget(self.header_label)

        # Chat Display
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                color: #D4D4D4;
                border: 1px solid #3E3E42;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        self.display.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.display.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.display, 1)

        # Input Area
        input_container = QVBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(60)
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D30;
                color: #D4D4D4;
                border: 1px solid #3E3E42;
                border-radius: 4px;
            }
        """)
        input_container.addWidget(self.input_field)

        # Buttons
        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton("Propagate")
        self.send_btn.setStyleSheet("background-color: #569CD6; color: white; border: none; padding: 5px 15px; border-radius: 3px;")
        self.send_btn.clicked.connect(self.on_send)
        btn_layout.addStretch()
        btn_layout.addWidget(self.send_btn)

        input_container.addLayout(btn_layout)
        layout.addLayout(input_container)

        # Install event filter for Enter key
        self.input_field.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.input_field and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.on_send()
                return True
        return super().eventFilter(obj, event)

    def on_send(self):
        text = self.input_field.toPlainText().strip()
        self.input_field.clear()
        self.input_submitted.emit(text)

    def update_display(self, conversation):
        self.display.clear()

        # Simple HTML rendering
        html = "<style>body { font-family: Segoe UI, sans-serif; color: #D4D4D4; } .user { color: #4EC9B0; font-weight: bold; } .ai { color: #569CD6; font-weight: bold; } .system { color: #CE9178; font-style: italic; }</style>"

        for msg in conversation:
            if msg.hidden: continue

            role_class = "user" if msg.role == "user" else "ai"
            if msg.role == "system": role_class = "system"

            name_label = "User"
            if msg.role == "assistant":
                name_label = f"{msg.ai_name}" if msg.ai_name else "AI"
                if msg.model:
                    name_label += f" ({msg.model})"
            elif msg.role == "system":
                name_label = "System"

            content = msg.content.replace("\n", "<br>")

            html += f"<div style='margin-bottom: 10px;'><span class='{role_class}'>{name_label}</span><br>{content}</div>"

            if msg.generated_image_path and os.path.exists(msg.generated_image_path):
                 html += f"<img src='{msg.generated_image_path}' width='400'><br>"

        self.display.setHtml(html)
        self.display.verticalScrollBar().setValue(self.display.verticalScrollBar().maximum())

    def show_context_menu(self, pos):
        cursor = self.display.textCursor()
        selected_text = cursor.selectedText()
        if not selected_text:
            return

        menu = self.display.createStandardContextMenu()
        menu.addSeparator()

        rabbit_action = QAction("🕳️ Rabbithole", self)
        rabbit_action.triggered.connect(lambda: self.rabbithole_requested.emit(selected_text))
        menu.addAction(rabbit_action)

        fork_action = QAction("🔱 Fork", self)
        fork_action.triggered.connect(lambda: self.fork_requested.emit(selected_text))
        menu.addAction(fork_action)

        menu.exec(self.display.mapToGlobal(pos))
