from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QLabel, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QRadialGradient, QColor, QPen
import os
import math
import random
import json

from src.ui.colors import COLORS
from src.ui.conversation_pane import ConversationPane
from src.ui.sidebar import RightSidebar
from src.ui.widgets.custom_widgets import SignalIndicator, ScanlineOverlayWidget
from src.core.conversation_manager import ConversationManager

class CentralContainer(QWidget):
    """Central container widget with animated background and overlay support"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Background animation state
        self.bg_offset = 0
        self.noise_offset = 0

        # Animation timer for background
        self.bg_timer = QTimer(self)
        self.bg_timer.timeout.connect(self._animate_bg)
        self.bg_timer.start(80)  # ~12 FPS for subtle movement

        # Create scanline overlay as child widget
        self.scanline_overlay = ScanlineOverlayWidget(self)
        self.scanline_overlay.hide()

    def _animate_bg(self):
        self.bg_offset = (self.bg_offset + 1) % 360
        self.noise_offset = (self.noise_offset + 0.5) % 100
        self.update()

    def set_scanlines_enabled(self, enabled):
        """Toggle scanline effect"""
        if enabled:
            # Ensure overlay has correct geometry before showing
            self.scanline_overlay.setGeometry(self.rect())
            self.scanline_overlay.show()
            self.scanline_overlay.raise_()
            self.scanline_overlay.start_animation()
        else:
            self.scanline_overlay.stop_animation()
            self.scanline_overlay.hide()

    def resizeEvent(self, event):
        """Update scanline overlay size when container resizes"""
        super().resizeEvent(event)
        self.scanline_overlay.setGeometry(self.rect())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ═══ ANIMATED BACKGROUND ═══
        # Create shifting gradient with more visible movement
        center_x = self.width() / 2 + math.sin(math.radians(self.bg_offset)) * 100
        center_y = self.height() / 2 + math.cos(math.radians(self.bg_offset * 0.7)) * 60

        gradient = QRadialGradient(center_x, center_y, max(self.width(), self.height()) * 0.9)

        # More visible atmospheric colors with cyan tint
        pulse = 0.5 + 0.5 * math.sin(math.radians(self.bg_offset * 2))
        center_r = int(10 + 8 * pulse)
        center_g = int(15 + 10 * pulse)
        center_b = int(30 + 15 * pulse)

        gradient.setColorAt(0, QColor(center_r, center_g, center_b))
        gradient.setColorAt(0.4, QColor(10, 14, 26))
        gradient.setColorAt(1, QColor(6, 8, 14))

        painter.fillRect(self.rect(), gradient)

        # Add subtle glow lines at edges
        glow_alpha = int(15 + 10 * pulse)
        glow_color = QColor(6, 182, 212, glow_alpha)  # Cyan glow
        painter.setPen(QPen(glow_color, 2))

        # Top edge glow
        painter.drawLine(0, 0, self.width(), 0)
        # Bottom edge glow
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        # Left edge glow
        painter.drawLine(0, 0, 0, self.height())
        # Right edge glow
        painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())

        # Add subtle noise/grain pattern
        noise_color = QColor(COLORS['accent_cyan'])
        noise_color.setAlpha(8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(noise_color)

        # Sparse random dots for grain effect
        random.seed(int(self.noise_offset))
        for _ in range(50):
            x = random.randint(0, self.width())
            y = random.randint(0, self.height())
            painter.drawEllipse(x, y, 1, 1)

class LiminalBackroomsApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()

        # Main app state
        self.conversation = []
        self.turn_count = 0
        self.images = []
        self.image_paths = []
        self.session_videos = []
        self.branch_conversations = {}
        self.active_branch = None
        self.muted_ais = set()

        # Set up the UI
        self.setup_ui()

        # Connect signals and slots
        self.connect_signals()

        # Dark theme
        self.apply_dark_theme()

        # Restore splitter state if available
        self.restore_splitter_state()

        # Start maximized
        self.showMaximized()

        # Create conversation manager
        self.conversation_manager = ConversationManager(self)
        self.conversation_manager.initialize()

    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("╔═ LIMINAL BACKROOMS v0.7 ═╗")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1200, 800)

        self.central_container = CentralContainer()
        self.setCentralWidget(self.central_container)

        main_layout = QHBoxLayout(self.central_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(8)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['border']};
                border: 1px solid {COLORS['border_highlight']};
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {COLORS['accent_blue']};
            }}
        """)
        main_layout.addWidget(self.splitter)

        self.left_pane = ConversationPane()
        self.right_sidebar = RightSidebar()

        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.right_sidebar)

        total_width = 1600
        self.splitter.setSizes([int(total_width * 0.70), int(total_width * 0.30)])

        self.right_sidebar.add_node('main', 'Seed', 'main')

        # Status bar
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_dim']};
                border-top: 1px solid {COLORS['border']};
                padding: 3px;
                font-size: 11px;
            }}
        """)
        self.statusBar().showMessage("Ready")

        self.notification_label = QLabel("")
        self.notification_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_cyan']};
                font-size: 11px;
                padding: 2px 10px;
                background-color: transparent;
            }}
        """)
        self.notification_label.setMaximumWidth(500)
        self.statusBar().addWidget(self.notification_label, 1)

        self.signal_indicator = SignalIndicator()
        self.statusBar().addPermanentWidget(self.signal_indicator)

        self.crt_checkbox = QCheckBox("CRT")
        self.crt_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_dim']};
                font-size: 10px;
                spacing: 4px;
            }}
            QCheckBox::indicator {{
                width: 12px;
                height: 12px;
                border: 1px solid {COLORS['border_glow']};
                border-radius: 2px;
                background: {COLORS['bg_dark']};
            }}
            QCheckBox::indicator:checked {{
                background: {COLORS['accent_cyan']};
            }}
        """)
        self.crt_checkbox.setToolTip("Toggle CRT scanline effect")
        self.crt_checkbox.toggled.connect(self.toggle_crt_effect)
        self.statusBar().addPermanentWidget(self.crt_checkbox)

    def toggle_crt_effect(self, enabled):
        """Toggle the CRT scanline effect"""
        if hasattr(self, 'central_container'):
            self.central_container.set_scanlines_enabled(enabled)

    def set_signal_active(self, active):
        """Set signal indicator to active (waiting for response)"""
        self.signal_indicator.set_active(active)

    def update_signal_latency(self, latency_ms):
        """Update signal indicator with response latency"""
        self.signal_indicator.set_latency(latency_ms)

    def connect_signals(self):
        """Connect all signals and slots"""
        self.right_sidebar.nodeSelected.connect(self.on_branch_select)

        # Save/Load Session buttons
        if hasattr(self.right_sidebar.control_panel, 'save_session_button'):
            self.right_sidebar.control_panel.save_session_button.clicked.connect(self.save_session)
        if hasattr(self.right_sidebar.control_panel, 'load_session_button'):
            self.right_sidebar.control_panel.load_session_button.clicked.connect(self.load_session)

        if hasattr(self.right_sidebar.network_pane.network_view, 'nodeHovered'):
            self.right_sidebar.network_pane.network_view.nodeHovered.connect(self.on_node_hover)

        self.left_pane.set_rabbithole_callback(self.conversation_manager.rabbithole_callback if hasattr(self, 'conversation_manager') else self.dummy_callback)
        self.left_pane.set_fork_callback(self.conversation_manager.fork_callback if hasattr(self, 'conversation_manager') else self.dummy_callback)

        self.splitter.splitterMoved.connect(self.save_splitter_state)

    def dummy_callback(self, *args):
        pass

    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_normal']};
            }}
            QWidget {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_normal']};
            }}
            QToolTip {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                padding: 5px;
            }}
        """)

    def on_node_hover(self, node_id):
        if node_id == 'main':
            self.statusBar().showMessage("Main conversation")
        elif node_id in self.branch_conversations:
            branch_data = self.branch_conversations[node_id]
            branch_type = branch_data.get('type', 'branch')
            selected_text = branch_data.get('selected_text', '')
            self.statusBar().showMessage(f"{branch_type.capitalize()}: {selected_text[:50]}...")

    def on_branch_select(self, branch_id):
        try:
            if branch_id == 'main':
                self.active_branch = None
                if not hasattr(self, 'main_conversation'):
                    self.main_conversation = []
                self.conversation = self.main_conversation
                self.left_pane.display_conversation(self.conversation)
                self.statusBar().showMessage("Switched to main conversation")
                return

            if branch_id not in self.branch_conversations:
                self.statusBar().showMessage(f"Branch {branch_id} not found")
                return

            branch_data = self.branch_conversations[branch_id]
            self.active_branch = branch_id
            self.conversation = branch_data['conversation']
            self.left_pane.display_conversation(self.conversation, branch_data)
            self.statusBar().showMessage(f"Switched to {branch_data['type']} branch: {branch_id}")

        except Exception as e:
            print(f"Error selecting branch: {e}")
            self.statusBar().showMessage(f"Error selecting branch: {e}")

    def save_splitter_state(self):
        try:
            if not os.path.exists('settings'):
                os.makedirs('settings')
            with open('settings/splitter_state.json', 'w') as f:
                json.dump({'sizes': self.splitter.sizes()}, f)
        except Exception as e:
            print(f"Error saving splitter state: {e}")

    def restore_splitter_state(self):
        try:
            if os.path.exists('settings/splitter_state.json'):
                with open('settings/splitter_state.json', 'r') as f:
                    state = json.load(f)
                    if 'sizes' in state:
                        self.splitter.setSizes(state['sizes'])
        except Exception as e:
            print(f"Error restoring splitter state: {e}")

    def save_session(self):
        """Save the current session"""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        # Get filename
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            os.path.join(os.getcwd(), "sessions"),
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        filename = os.path.basename(file_path)

        # Gather data
        conversation = self.main_conversation if hasattr(self, 'main_conversation') else []

        success, result = self.conversation_manager.session_manager.save_session(
            filename,
            conversation,
            self.branch_conversations,
            self.active_branch,
            {
                "ai_models": {
                    "AI-1": self.right_sidebar.control_panel.ai1_model_selector.currentText(),
                    "AI-2": self.right_sidebar.control_panel.ai2_model_selector.currentText(),
                    "AI-3": self.right_sidebar.control_panel.ai3_model_selector.currentText(),
                    "AI-4": self.right_sidebar.control_panel.ai4_model_selector.currentText(),
                    "AI-5": self.right_sidebar.control_panel.ai5_model_selector.currentText(),
                },
                "scenario": self.right_sidebar.control_panel.prompt_pair_selector.currentText()
            }
        )

        if success:
            self.statusBar().showMessage(f"Session saved to {result}")
            QMessageBox.information(self, "Success", "Session saved successfully!")
        else:
            QMessageBox.critical(self, "Error", f"Failed to save session: {result}")

    def load_session(self):
        """Load a saved session"""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        # Get filename
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            os.path.join(os.getcwd(), "sessions"),
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        filename = os.path.basename(file_path)

        success, result = self.conversation_manager.session_manager.load_session(filename)

        if success:
            data = result

            # Restore conversation
            self.main_conversation = data.get("conversation", [])
            self.branch_conversations = data.get("branch_conversations", {})
            self.active_branch = data.get("active_branch")

            # Restore active branch or main conversation
            if self.active_branch and self.active_branch in self.branch_conversations:
                self.on_branch_select(self.active_branch)
            else:
                self.active_branch = None
                self.conversation = self.main_conversation
                self.left_pane.display_conversation(self.conversation)

            # Restore graph
            # This requires rebuilding the graph structure in network pane
            # For now, we'll just clear and rebuild basic nodes if possible
            # Ideally, NetworkPane should support bulk loading

            # Restore UI state if metadata exists
            metadata = data.get("metadata", {})
            if "ai_models" in metadata:
                models = metadata["ai_models"]
                self.right_sidebar.control_panel.ai1_model_selector.setCurrentText(models.get("AI-1", ""))
                self.right_sidebar.control_panel.ai2_model_selector.setCurrentText(models.get("AI-2", ""))
                self.right_sidebar.control_panel.ai3_model_selector.setCurrentText(models.get("AI-3", ""))
                self.right_sidebar.control_panel.ai4_model_selector.setCurrentText(models.get("AI-4", ""))
                self.right_sidebar.control_panel.ai5_model_selector.setCurrentText(models.get("AI-5", ""))

            if "scenario" in metadata:
                self.right_sidebar.control_panel.prompt_pair_selector.setCurrentText(metadata["scenario"])

            self.statusBar().showMessage(f"Session loaded from {filename}")
            QMessageBox.information(self, "Success", "Session loaded successfully!")
        else:
            QMessageBox.critical(self, "Error", f"Failed to load session: {result}")
