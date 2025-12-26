from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QComboBox, QScrollArea, QCheckBox, QMenu, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QTimer
from PyQt6.QtGui import QTextCursor, QFont, QColor, QTextCharFormat, QPixmap, QImage
from src.ui.colors import COLORS
from src.ui.widgets.custom_widgets import GlowButton
from src.core.config import AI_MODELS, SYSTEM_PROMPT_PAIRS
from src.utils.shared_utils import open_html_in_browser
import base64
import os

class ControlPanel(QWidget):
    """Control panel with mode, model selections, etc."""
    def __init__(self):
        super().__init__()

        # Set up the UI
        self.setup_ui()

        # Initialize with models and prompt pairs
        self.initialize_selectors()

    def setup_ui(self):
        """Set up the user interface for the control panel - vertical sidebar layout"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)

        # Add a title with cyberpunk styling
        title = QLabel("═ CONTROL PANEL ═")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 12px;
            font-weight: bold;
            padding: 10px;
            background-color: {COLORS['bg_medium']};
            border: 1px solid {COLORS['border_glow']};
            border-radius: 0px;
            letter-spacing: 2px;
        """)
        main_layout.addWidget(title)

        # Create scrollable area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_medium']};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border_glow']};
                min-height: 20px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['accent_cyan']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)

        # Container widget for scrollable content
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: transparent;")

        # All controls in vertical layout
        controls_layout = QVBoxLayout(scroll_content)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(10)

        # Mode selection with icon
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(5)

        mode_label = QLabel("▸ MODE")
        mode_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        mode_layout.addWidget(mode_label)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["AI-AI", "Human-AI"])
        self.mode_selector.setStyleSheet(self.get_combobox_style())
        mode_layout.addWidget(self.mode_selector)
        controls_layout.addWidget(mode_container)

        # Iterations with slider
        iterations_container = QWidget()
        iterations_layout = QVBoxLayout(iterations_container)
        iterations_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.setSpacing(5)

        iterations_label = QLabel("▸ ITERATIONS")
        iterations_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        iterations_layout.addWidget(iterations_label)

        self.iterations_selector = QComboBox()
        self.iterations_selector.addItems(["1", "2", "5", "6", "10", "100"])
        self.iterations_selector.setStyleSheet(self.get_combobox_style())
        iterations_layout.addWidget(self.iterations_selector)
        controls_layout.addWidget(iterations_container)

        # Number of AIs selection
        num_ais_container = QWidget()
        num_ais_layout = QVBoxLayout(num_ais_container)
        num_ais_layout.setContentsMargins(0, 0, 0, 0)
        num_ais_layout.setSpacing(5)

        num_ais_label = QLabel("▸ NUMBER OF AIs")
        num_ais_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        num_ais_layout.addWidget(num_ais_label)

        self.num_ais_selector = QComboBox()
        self.num_ais_selector.addItems(["1", "2", "3", "4", "5"])
        self.num_ais_selector.setCurrentText("3")  # Default to 3 AIs
        self.num_ais_selector.setStyleSheet(self.get_combobox_style())
        num_ais_layout.addWidget(self.num_ais_selector)
        controls_layout.addWidget(num_ais_container)

        # AI-1 Model selection
        self.ai1_container = QWidget()
        ai1_layout = QVBoxLayout(self.ai1_container)
        ai1_layout.setContentsMargins(0, 0, 0, 0)
        ai1_layout.setSpacing(5)

        ai1_label = QLabel("AI-1")
        ai1_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai1_layout.addWidget(ai1_label)

        self.ai1_model_selector = QComboBox()
        self.ai1_model_selector.setStyleSheet(self.get_combobox_style())
        ai1_layout.addWidget(self.ai1_model_selector)
        controls_layout.addWidget(self.ai1_container)

        # AI-2 Model selection
        self.ai2_container = QWidget()
        ai2_layout = QVBoxLayout(self.ai2_container)
        ai2_layout.setContentsMargins(0, 0, 0, 0)
        ai2_layout.setSpacing(5)

        ai2_label = QLabel("AI-2")
        ai2_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai2_layout.addWidget(ai2_label)

        self.ai2_model_selector = QComboBox()
        self.ai2_model_selector.setStyleSheet(self.get_combobox_style())
        ai2_layout.addWidget(self.ai2_model_selector)
        controls_layout.addWidget(self.ai2_container)

        # AI-3 Model selection
        self.ai3_container = QWidget()
        ai3_layout = QVBoxLayout(self.ai3_container)
        ai3_layout.setContentsMargins(0, 0, 0, 0)
        ai3_layout.setSpacing(5)

        ai3_label = QLabel("AI-3")
        ai3_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai3_layout.addWidget(ai3_label)

        self.ai3_model_selector = QComboBox()
        self.ai3_model_selector.setStyleSheet(self.get_combobox_style())
        ai3_layout.addWidget(self.ai3_model_selector)
        controls_layout.addWidget(self.ai3_container)

        # AI-4 Model selection
        self.ai4_container = QWidget()
        ai4_layout = QVBoxLayout(self.ai4_container)
        ai4_layout.setContentsMargins(0, 0, 0, 0)
        ai4_layout.setSpacing(5)

        ai4_label = QLabel("AI-4")
        ai4_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai4_layout.addWidget(ai4_label)

        self.ai4_model_selector = QComboBox()
        self.ai4_model_selector.setStyleSheet(self.get_combobox_style())
        ai4_layout.addWidget(self.ai4_model_selector)
        controls_layout.addWidget(self.ai4_container)

        # AI-5 Model selection
        self.ai5_container = QWidget()
        ai5_layout = QVBoxLayout(self.ai5_container)
        ai5_layout.setContentsMargins(0, 0, 0, 0)
        ai5_layout.setSpacing(5)

        ai5_label = QLabel("AI-5")
        ai5_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai5_layout.addWidget(ai5_label)

        self.ai5_model_selector = QComboBox()
        self.ai5_model_selector.setStyleSheet(self.get_combobox_style())
        ai5_layout.addWidget(self.ai5_model_selector)
        controls_layout.addWidget(self.ai5_container)

        # Prompt pair selection
        prompt_container = QWidget()
        prompt_layout = QVBoxLayout(prompt_container)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(5)

        prompt_label = QLabel("Conversation Scenario")
        prompt_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        prompt_layout.addWidget(prompt_label)

        self.prompt_pair_selector = QComboBox()
        self.prompt_pair_selector.setStyleSheet(self.get_combobox_style())
        prompt_layout.addWidget(self.prompt_pair_selector)
        controls_layout.addWidget(prompt_container)

        # Action buttons container
        action_container = QWidget()
        action_layout = QVBoxLayout(action_container)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(5)

        action_label = QLabel("▸ OPTIONS")
        action_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        action_layout.addWidget(action_label)

        # Auto-generate images checkbox
        self.auto_image_checkbox = QCheckBox("Auto-generate images")
        self.auto_image_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_normal']};
                spacing: 5px;
                font-size: 10px;
                padding: 4px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                background-color: {COLORS['bg_medium']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent_cyan']};
                border: 1px solid {COLORS['accent_cyan']};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {COLORS['accent_cyan']};
            }}
        """)
        self.auto_image_checkbox.setToolTip("Automatically generate images from AI responses using Google Gemini 3 Pro Image Preview via OpenRouter")
        action_layout.addWidget(self.auto_image_checkbox)

        # Actions - buttons in vertical layout
        actions_label = QLabel("▸ ACTIONS")
        actions_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        action_layout.addWidget(actions_label)

        # Export button with glow
        self.export_button = self.create_glow_button("📡 EXPORT", COLORS['accent_purple'])
        action_layout.addWidget(self.export_button)

        # Save/Load Session buttons
        session_buttons_layout = QHBoxLayout()
        self.save_session_button = self.create_glow_button("💾 SAVE", COLORS['accent_cyan'])
        self.load_session_button = self.create_glow_button("📂 LOAD", COLORS['accent_cyan'])
        session_buttons_layout.addWidget(self.save_session_button)
        session_buttons_layout.addWidget(self.load_session_button)
        action_layout.addLayout(session_buttons_layout)

        # View HTML button with glow - opens the styled conversation
        self.view_html_button = self.create_glow_button("🌐 VIEW HTML", COLORS['accent_green'])
        self.view_html_button.setToolTip("View conversation as shareable HTML")
        self.view_html_button.clicked.connect(lambda: open_html_in_browser("conversation_full.html"))
        action_layout.addWidget(self.view_html_button)

        # BackroomsBench evaluation button
        self.backroomsbench_button = self.create_glow_button("🌀 BACKROOMSBENCH (beta)", COLORS['accent_purple'])
        self.backroomsbench_button.setToolTip("Run multi-judge AI evaluation (depth/philosophy)")
        action_layout.addWidget(self.backroomsbench_button)

        controls_layout.addWidget(action_container)

        # Add all controls directly to controls_layout (now vertical)
        controls_layout.addWidget(mode_container)
        controls_layout.addWidget(iterations_container)
        controls_layout.addWidget(num_ais_container)

        # Divider
        divider1 = QLabel("─" * 20)
        divider1.setStyleSheet(f"color: {COLORS['border_glow']}; font-size: 8px;")
        controls_layout.addWidget(divider1)

        models_label = QLabel("▸ AI MODELS")
        models_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        controls_layout.addWidget(models_label)

        controls_layout.addWidget(self.ai1_container)
        controls_layout.addWidget(self.ai2_container)
        controls_layout.addWidget(self.ai3_container)
        controls_layout.addWidget(self.ai4_container)
        controls_layout.addWidget(self.ai5_container)

        # Divider
        divider2 = QLabel("─" * 20)
        divider2.setStyleSheet(f"color: {COLORS['border_glow']}; font-size: 8px;")
        controls_layout.addWidget(divider2)

        scenario_label = QLabel("▸ SCENARIO")
        scenario_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        controls_layout.addWidget(scenario_label)

        controls_layout.addWidget(prompt_container)

        # Divider
        divider3 = QLabel("─" * 20)
        divider3.setStyleSheet(f"color: {COLORS['border_glow']}; font-size: 8px;")
        controls_layout.addWidget(divider3)

        controls_layout.addWidget(action_container)

        # Add spacer
        controls_layout.addStretch()

        # Set the scroll area widget and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)  # Stretch to fill

    def get_combobox_style(self):
        """Get the style for comboboxes - cyberpunk themed"""
        return f"""
            QComboBox {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 8px 10px;
                min-height: 30px;
                font-size: 10px;
            }}
            QComboBox:hover {{
                border: 1px solid {COLORS['accent_cyan']};
                color: {COLORS['text_bright']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                image: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_normal']};
                selection-background-color: {COLORS['accent_cyan']};
                selection-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 28px;
                padding: 4px;
            }}
        """

    def get_cyberpunk_button_style(self, accent_color):
        """Get cyberpunk-themed button style with given accent color"""
        return f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {accent_color};
                border: 2px solid {accent_color};
                border-radius: 3px;
                padding: 10px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {accent_color};
                color: {COLORS['bg_dark']};
                border: 2px solid {accent_color};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['bg_light']};
                color: {accent_color};
            }}
        """

    def create_glow_button(self, text, accent_color):
        """Create a button with glow effect"""
        button = GlowButton(text, accent_color)
        button.setStyleSheet(self.get_cyberpunk_button_style(accent_color))
        return button

    def initialize_selectors(self):
        """Initialize the selector dropdowns with values from config"""
        # Add AI models
        self.ai1_model_selector.clear()
        self.ai2_model_selector.clear()
        self.ai3_model_selector.clear()
        self.ai4_model_selector.clear()
        self.ai5_model_selector.clear()
        self.ai1_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai2_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai3_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai4_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai5_model_selector.addItems(list(AI_MODELS.keys()))

        # Add prompt pairs
        self.prompt_pair_selector.clear()
        self.prompt_pair_selector.addItems(list(SYSTEM_PROMPT_PAIRS.keys()))

        # Connect number of AIs selector to update visibility
        self.num_ais_selector.currentTextChanged.connect(self.update_ai_selector_visibility)

        # Set initial visibility based on default number of AIs (3)
        self.update_ai_selector_visibility("3")

    def update_ai_selector_visibility(self, num_ais_text):
        """Show/hide AI model selectors based on number of AIs selected"""
        num_ais = int(num_ais_text)
        self.ai1_container.setVisible(num_ais >= 1)
        self.ai2_container.setVisible(num_ais >= 2)
        self.ai3_container.setVisible(num_ais >= 3)
        self.ai4_container.setVisible(num_ais >= 4)
        self.ai5_container.setVisible(num_ais >= 5)
