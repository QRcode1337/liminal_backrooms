# gui.py
"""
Main GUI module for Liminal Backrooms application.

All styling is imported from styles.py - the single source of truth for colors/fonts.
"""

import os
import json
import requests
import threading
import math
import random
from datetime import datetime
from io import BytesIO
from PIL import Image
import time
from pathlib import Path
import uuid
import shutil
import networkx as nx
import re
import sys
import webbrowser
import subprocess
import base64
from PyQt6.QtCore import Qt, QRect, QTimer, QRectF, QPointF, QSize, pyqtSignal, QEvent, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QFontDatabase, QTextCursor, QAction, QKeySequence, QTextCharFormat, QLinearGradient, QRadialGradient, QPainterPath, QImage, QPixmap, QShortcut
from PyQt6.QtWidgets import QWidget, QApplication, QMainWindow, QSplitter, QVBoxLayout, QHBoxLayout, QTextEdit, QFrame, QLineEdit, QPushButton, QLabel, QComboBox, QMenu, QFileDialog, QMessageBox, QScrollArea, QToolTip, QSizePolicy, QCheckBox, QGraphicsDropShadowEffect

from config import (
    AI_MODELS,
    SYSTEM_PROMPT_PAIRS,
    SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT,
    OUTPUTS_DIR,
    DEVELOPER_TOOLS,
    TURN_DELAY
)

# Import centralized styling - single source of truth for colors and widget styles
from styles import COLORS, FONTS, get_combobox_style, get_button_style, get_checkbox_style, get_scrollbar_style

# Import shared utilities - with fallback for open_html_in_browser
from shared_utils import generate_image_from_text, get_visible_messages
try:
    from shared_utils import open_html_in_browser
except ImportError:
    open_html_in_browser = None

# Add import for grouped model selector functionality
from grouped_model_selector import GroupedModelComboBox


# =============================================================================
# MESSAGE WIDGET CHAT SYSTEM - Each message is a separate widget
# =============================================================================
# This solves scroll jumping because adding/updating messages doesn't destroy
# existing widgets. QScrollArea naturally preserves scroll position.
# =============================================================================

class MessageWidget(QFrame):
    """
    A single message in the chat - renders as a styled frame with content.
    
    Styling rules:
    - No rounded corners (retro CRT theme)
    - bg_medium background on message blocks
    - Transparent backgrounds on text labels (no black text boxes)
    - Left-aligned borders for all messages (including human)
    - AI colors applied per AI number
    """
    
    # AI color mapping - matches styles.py COLORS
    AI_COLORS = {
        1: COLORS['ai_1'],  # Bright phosphor green
        2: COLORS['ai_2'],  # Lighter green
        3: COLORS['ai_3'],  # Mint green
        4: COLORS['ai_4'],  # Pale green
        5: COLORS['ai_5'],  # Near-white green
    }
    HUMAN_COLOR = COLORS['human']  # Amber
    TIMESTAMP_COLOR = '#7a8899'  # Subtle readable gray
    
    def __init__(self, message_data, parent=None):
        super().__init__(parent)
        self.message_data = message_data
        self._content_label = None  # Reference to content label for updates
        self._creation_time = datetime.now()
        self._setup_ui()
        self._setup_tooltip()
    
    def _setup_tooltip(self):
        """Set up hover tooltip showing message timestamp and metadata."""
        # Try to get timestamp from message data, fallback to creation time
        ts = self.message_data.get('timestamp') or self.message_data.get('_timestamp')
        if ts:
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                    time_str = dt.strftime('%H:%M:%S')
                    date_str = dt.strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    time_str = ts
                    date_str = ''
            elif isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts)
                time_str = dt.strftime('%H:%M:%S')
                date_str = dt.strftime('%Y-%m-%d')
            else:
                time_str = str(ts)
                date_str = ''
        else:
            time_str = self._creation_time.strftime('%H:%M:%S')
            date_str = self._creation_time.strftime('%Y-%m-%d')

        # Build tooltip parts
        role = self.message_data.get('role', '')
        ai_name = self.message_data.get('ai_name', '')
        model = self.message_data.get('model', '')

        tooltip_parts = []
        if date_str:
            tooltip_parts.append(f"{date_str} {time_str}")
        else:
            tooltip_parts.append(time_str)

        if model:
            tooltip_parts.append(f"Model: {model}")

        # Estimate token count for this message
        content = self.message_data.get('content', '')
        if isinstance(content, str):
            char_count = len(content)
        elif isinstance(content, list):
            char_count = sum(len(p.get('text', '')) for p in content if isinstance(p, dict) and p.get('type') == 'text')
        else:
            char_count = 0

        if char_count > 0:
            est_tokens = char_count // 4
            tooltip_parts.append(f"~{est_tokens} tokens ({char_count} chars)")

        self.setToolTip('\n'.join(tooltip_parts))

    def _setup_ui(self):
        """Build the widget UI based on message data."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        role = self.message_data.get('role', 'user')
        content = self.message_data.get('content', '')
        msg_type = self.message_data.get('_type', '')
        
        # Extract text from structured content
        text_content = self._extract_text(content)
        
        # Style based on role/type
        if msg_type == 'typing_indicator':
            self._setup_typing_indicator()
        elif msg_type == 'branch_indicator':
            self._setup_branch_indicator(text_content)
        elif msg_type == 'agent_notification':
            self._setup_notification(text_content)
        elif msg_type == 'generated_image':
            self._setup_generated_image()
        elif msg_type == 'generated_video':
            self._setup_generated_video()
        elif role == 'user':
            self._setup_user_message(text_content)
        elif role == 'assistant':
            self._setup_assistant_message(text_content)
        elif role == 'system':
            self._setup_system_message(text_content)
        else:
            self._setup_default_message(text_content)
    
    def _extract_text(self, content):
        """Extract text from content (handles structured content with images)."""
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
            return ''.join(text_parts)
        return str(content) if content else ''
    
    def _format_code_blocks(self, text):
        """
        Convert markdown code blocks and inline code to HTML for Qt RichText.
        
        Uses table-based HTML structure that Qt renders properly.
        """
        import re
        import html
        
        # Split by code blocks first (```...```)
        code_block_pattern = r'```(\w*)\n?(.*?)```'
        
        parts = []
        last_end = 0
        
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            # Text before this code block
            before_text = text[last_end:match.start()]
            parts.append(('text', before_text))
            
            # The code block
            lang = match.group(1) or ''
            code = match.group(2)
            parts.append(('code_block', code, lang))
            
            last_end = match.end()
        
        # Remaining text
        if last_end < len(text):
            parts.append(('text', text[last_end:]))
        
        # Process each part
        result = []
        
        # Colors for code blocks
        code_bg = '#001100'
        header_bg = '#002200'
        border_color = COLORS.get('border', '#0D3B0D')
        code_text_color = '#E0E0E0'
        
        for part in parts:
            if part[0] == 'code_block':
                code = html.escape(part[1].rstrip())
                lang = part[2].lower()
                
                # Language header row
                lang_row = ''
                if lang:
                    lang_row = (
                        f'<tr><td style="background-color: {header_bg}; '
                        f'padding: 4px 10px; border-bottom: 1px solid {border_color};">'
                        f'<span style="color: {COLORS["text_dim"]}; font-size: 9pt; '
                        f'font-family: Consolas, Monaco, monospace; font-weight: bold;">'
                        f'{lang.upper()}</span></td></tr>'
                    )
                
                # Code block with subtle border
                result.append(
                    f'<table cellspacing="0" cellpadding="0" '
                    f'style="margin: 8px 0 8px 10px; border: 1px solid {border_color};">'
                    f'<tr><td style="background-color: {code_bg}; padding: 0;">'
                    f'<table cellspacing="0" cellpadding="0" width="100%">'
                    f'{lang_row}'
                    f'<tr><td style="background-color: {code_bg}; padding: 10px 12px;">'
                    f'<pre style="margin: 0; font-family: Consolas, Monaco, monospace; '
                    f'font-size: 9pt; white-space: pre-wrap; color: {code_text_color};">{code}</pre>'
                    f'</td></tr></table></td></tr></table>'
                )
            else:
                # Regular text - escape and process inline code
                text_part = html.escape(part[1])
                
                # Replace inline code `...` with styled spans
                inline_pattern = r'`([^`]+)`'
                text_part = re.sub(
                    inline_pattern,
                    f'<span style="background-color: {code_bg}; color: {COLORS["accent_cyan"]}; '
                    f'border: 1px solid {border_color}; '
                    f'padding: 1px 4px; font-family: Consolas, Monaco, monospace; font-size: 9pt;">'
                    f'\\1</span>',
                    text_part
                )
                
                # Convert newlines to <br>
                text_part = text_part.replace('\n', '<br/>')
                
                result.append(text_part)
        
        return ''.join(result)
    
    def _create_header_widget(self, name_text, color):
        """Create a header widget with name."""
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: transparent;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        # Name label (left)
        name_label = QLabel(name_text)
        name_label.setStyleSheet(f"background-color: transparent; color: {color}; font-weight: bold; font-size: 9pt;")
        header_layout.addWidget(name_label)

        header_layout.addStretch()

        return header_widget
    
    def _get_ai_color(self):
        """Get the color for this AI based on _ai_number or extracted from ai_name."""
        ai_num = self.message_data.get('_ai_number')
        
        # If no _ai_number, try to extract from ai_name (e.g., "AI-1", "AI-2")
        if ai_num is None:
            ai_name = self.message_data.get('ai_name', '')
            if ai_name:
                import re
                match = re.search(r'AI-?(\d+)', ai_name, re.IGNORECASE)
                if match:
                    ai_num = int(match.group(1))
        
        # Default to 1 if still not found
        if ai_num is None:
            ai_num = 1
            
        return self.AI_COLORS.get(ai_num, self.AI_COLORS[1])

    def _setup_typing_indicator(self):
        """Setup typing indicator style."""
        ai_name = self.message_data.get('ai_name', 'AI')
        model = self.message_data.get('model', '')
        border_color = self._get_ai_color()
        display_name = f"{ai_name} ({model})" if model else ai_name
        
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {COLORS['bg_medium']};
                border-left: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)
        
        header = self._create_header_widget(display_name, border_color)
        self.layout().addWidget(header)
        
        dots = QLabel("thinking...")
        dots.setStyleSheet(f"background-color: transparent; color: {COLORS['text_dim']}; font-style: italic;")
        self.layout().addWidget(dots)
    
    def _setup_branch_indicator(self, text):
        """Setup branch indicator style."""
        if "Rabbitholing" in text:
            color = COLORS.get('accent_magenta', '#ff00ff')
        else:
            color = COLORS.get('accent_cyan', '#00ffff')
        
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: transparent;
                border: 1px dashed {color};
                border-radius: 0px;
                padding: 4px;
            }}
        """)
        
        label = QLabel(text)
        label.setStyleSheet(f"background-color: transparent; color: {color}; font-size: 9pt;")
        label.setWordWrap(True)
        self.layout().addWidget(label)
    
    def _setup_notification(self, text):
        """Setup agent notification style with color-matching backgrounds."""
        command_success = self.message_data.get('_command_success')
        if command_success is False:
            bg_color = "#1A0000"  # Dark red tint
            border_color = "#ff4444"  # Bright red (distinct from human pink)
        elif command_success is True:
            bg_color = "#001A00"  # Dark green tint
            border_color = COLORS.get('notify_success', '#5DFF44')
        else:
            bg_color = "#1A1A00"  # Dark yellow tint
            border_color = COLORS.get('notify_info', '#FFFF48')
        
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {bg_color};
                border-left: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)
        
        label = QLabel(text)
        label.setStyleSheet(f"background-color: transparent; color: {COLORS['text_normal']}; font-size: 9pt;")
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        self.layout().addWidget(label)
    
    def _setup_generated_image(self):
        """Setup generated image display with AI-matching colors."""
        ai_name = self.message_data.get('ai_name', 'AI')
        model = self.message_data.get('model', '')
        # Try multiple field names for image model
        image_model = (self.message_data.get('_image_model') or 
                       self.message_data.get('image_model') or 
                       'image model')
        image_path = self.message_data.get('generated_image_path', '')
        
        # Try multiple field names for prompt, including extracting from content
        image_prompt = (self.message_data.get('_prompt') or 
                        self.message_data.get('image_prompt') or '')
        
        # If no prompt found, try extracting from content (e.g., !image "prompt here")
        if not image_prompt:
            content = self.message_data.get('content', '')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text = part.get('text', '')
                        import re
                        match = re.search(r'!image\s+"([^"]+)"', text)
                        if match:
                            image_prompt = match.group(1)
                            break
            elif isinstance(content, str) and '!image' in content:
                import re
                match = re.search(r'!image\s+"([^"]+)"', content)
                if match:
                    image_prompt = match.group(1)
        
        border_color = self._get_ai_color()
        
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {COLORS['bg_medium']};
                border-left: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)
        
        # Header: "AI-X (model) generated an image using <image_model>"
        display_name = f"{ai_name} ({model})" if model else ai_name
        header_text = f"{display_name} generated an image using {image_model}"
        header = self._create_header_widget(header_text, border_color)
        self.layout().addWidget(header)
        
        # Show prompt if available
        if image_prompt:
            prompt_label = QLabel(f"Prompt: {image_prompt}")
            prompt_label.setStyleSheet(f"background-color: transparent; color: {self.TIMESTAMP_COLOR}; font-size: 9pt; font-style: italic;")
            prompt_label.setWordWrap(True)
            self.layout().addWidget(prompt_label)
        
        # Display image - scales to fill available width
        if image_path and os.path.exists(image_path):
            img_label = QLabel()
            img_label.setStyleSheet("background-color: transparent;")
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                img_label._original_pixmap = pixmap
                # Scale to available width (with padding margin)
                avail_width = max(400, self.width() - 40)
                if pixmap.width() > avail_width:
                    scaled = pixmap.scaledToWidth(avail_width, Qt.TransformationMode.SmoothTransformation)
                else:
                    scaled = pixmap
                img_label.setPixmap(scaled)
                # Re-scale on resize
                orig_resize = img_label.resizeEvent
                def _on_resize(event, lbl=img_label):
                    pm = getattr(lbl, '_original_pixmap', None)
                    if pm and not pm.isNull():
                        w = lbl.parent().width() - 40 if lbl.parent() else lbl.width()
                        w = max(200, w)
                        if pm.width() > w:
                            lbl.setPixmap(pm.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation))
                        else:
                            lbl.setPixmap(pm)
                    if orig_resize:
                        orig_resize(event)
                img_label.resizeEvent = _on_resize
                self.layout().addWidget(img_label)
    
    def _setup_generated_video(self):
        """Setup generated video display with AI-matching colors."""
        ai_name = self.message_data.get('ai_name', 'AI')
        model = self.message_data.get('model', '')
        video_model = self.message_data.get('video_model', 'unknown model')
        video_path = self.message_data.get('generated_video_path', '')
        video_prompt = self.message_data.get('video_prompt', '')
        border_color = self._get_ai_color()
        
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {COLORS['bg_medium']};
                border-left: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)
        
        # Header: "AI-X (model) generated a video using <video_model>"
        display_name = f"{ai_name} ({model})" if model else ai_name
        header_text = f"{display_name} generated a video using {video_model}"
        header = self._create_header_widget(header_text, border_color)
        self.layout().addWidget(header)
        
        # Show prompt if available
        if video_prompt:
            prompt_label = QLabel(f"Prompt: {video_prompt}")
            prompt_label.setStyleSheet(f"background-color: transparent; color: {self.TIMESTAMP_COLOR}; font-size: 9pt; font-style: italic;")
            prompt_label.setWordWrap(True)
            self.layout().addWidget(prompt_label)
        
        # Display video path info
        if video_path:
            path_label = QLabel(f"Video: {os.path.basename(video_path)}")
            path_label.setStyleSheet(f"background-color: transparent; color: {COLORS['text_normal']}; font-size: 9pt;")
            self.layout().addWidget(path_label)
    
    def _setup_user_message(self, text):
        """Setup human user message style - left-aligned like AI messages."""
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {COLORS['bg_medium']};
                border-left: 3px solid {self.HUMAN_COLOR};
                border-radius: 0px;
            }}
        """)

        display_name = self.message_data.get('_user_name', 'Human User')
        header = self._create_header_widget(display_name, self.HUMAN_COLOR)
        self.layout().addWidget(header)
        
        # Format code blocks and use RichText
        formatted_text = self._format_code_blocks(text)
        content = QLabel(formatted_text)
        content.setStyleSheet(f"background-color: transparent; color: {COLORS['text_normal']};")
        content.setWordWrap(True)
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.layout().addWidget(content)
        self._content_label = content
    
    def _setup_assistant_message(self, text):
        """Setup AI assistant message style."""
        ai_name = self.message_data.get('ai_name', 'AI')
        model = self.message_data.get('model', '')
        border_color = self._get_ai_color()
        
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {COLORS['bg_medium']};
                border-left: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)
        
        display_name = f"{ai_name} ({model})" if model else ai_name
        header = self._create_header_widget(display_name, border_color)
        self.layout().addWidget(header)

        # Format code blocks and use RichText
        formatted_text = self._format_code_blocks(text)
        content = QLabel(formatted_text)
        content.setStyleSheet(f"background-color: transparent; color: {COLORS['text_normal']};")
        content.setWordWrap(True)
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.layout().addWidget(content)
        self._content_label = content
    
    def _setup_system_message(self, text):
        """Setup system message style."""
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {COLORS['bg_medium']};
                border-left: 3px solid {COLORS['text_dim']};
                border-radius: 0px;
            }}
        """)
        
        # Format code blocks and use RichText
        formatted_text = self._format_code_blocks(text)
        content = QLabel(formatted_text)
        content.setStyleSheet(f"background-color: transparent; color: {COLORS['text_dim']}; font-style: italic;")
        content.setWordWrap(True)
        content.setTextFormat(Qt.TextFormat.RichText)
        self.layout().addWidget(content)
        self._content_label = content
    
    def _setup_default_message(self, text):
        """Default message style."""
        self.setStyleSheet(f"""
            MessageWidget {{
                background-color: {COLORS['bg_medium']};
                border-radius: 0px;
            }}
        """)
        
        # Format code blocks and use RichText
        formatted_text = self._format_code_blocks(text)
        content = QLabel(formatted_text)
        content.setStyleSheet(f"background-color: transparent; color: {COLORS['text_normal']};")
        content.setWordWrap(True)
        content.setTextFormat(Qt.TextFormat.RichText)
        self.layout().addWidget(content)
        self._content_label = content

    def mouseDoubleClickEvent(self, event):
        """Copy message text to clipboard on double-click."""
        content = self.message_data.get('content', '')
        if isinstance(content, list):
            text = ' '.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
        else:
            text = str(content)

        if text.strip():
            QApplication.clipboard().setText(text)
            # Brief visual feedback - flash the border
            original_style = self.styleSheet()
            self.setStyleSheet(original_style + f"\nMessageWidget {{ border: 1px solid {COLORS['accent_cyan']}; }}")
            QTimer.singleShot(300, lambda: self.setStyleSheet(original_style))

        super().mouseDoubleClickEvent(event)

    def update_content(self, new_text):
        """Update the content of this message (for streaming).
        
        For streaming, we use the simple HTML approach since widgets can't be
        efficiently updated incrementally. Full code block widgets are used
        for final rendered messages.
        """
        if self._content_label:
            # Format code blocks for RichText display (HTML-based for streaming)
            formatted_text = self._format_code_blocks(new_text)
            self._content_label.setText(formatted_text)


class SearchOverlay(QWidget):
    """Search bar overlay for finding text in conversation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.match_indices = []  # List of (widget_index, text_position) for matches
        self.current_match = -1
        self._setup_ui()
        self.hide()  # Hidden by default

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_medium']};
                border-bottom: 1px solid {COLORS['border_glow']};
            }}
        """)

        # Search icon label
        icon_label = QLabel("\U0001f50d")
        icon_label.setStyleSheet(f"background-color: transparent; color: {COLORS['text_dim']}; font-size: 12px; border: none;")
        layout.addWidget(icon_label)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search conversation...")
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_bright']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 4px 8px;
                font-size: 10px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['accent_cyan']};
            }}
        """)
        self.search_input.textChanged.connect(self._on_search_changed)
        self.search_input.returnPressed.connect(self._next_match)
        layout.addWidget(self.search_input)

        # Match counter
        self.match_label = QLabel("")
        self.match_label.setStyleSheet(f"background-color: transparent; color: {COLORS['text_dim']}; font-size: 9px; border: none; min-width: 60px;")
        layout.addWidget(self.match_label)

        # Navigation buttons
        prev_btn = QPushButton("\u25b2")
        prev_btn.setFixedSize(24, 24)
        prev_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
            }}
        """)
        prev_btn.clicked.connect(self._prev_match)
        layout.addWidget(prev_btn)

        next_btn = QPushButton("\u25bc")
        next_btn.setFixedSize(24, 24)
        next_btn.setStyleSheet(prev_btn.styleSheet())
        next_btn.clicked.connect(self._next_match)
        layout.addWidget(next_btn)

        # Close button
        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_dim']};
                border: none;
                font-size: 12px;
            }}
            QPushButton:hover {{
                color: {COLORS['accent_pink']};
            }}
        """)
        close_btn.clicked.connect(self.close_search)
        layout.addWidget(close_btn)

    def toggle(self):
        """Toggle search bar visibility."""
        if self.isVisible():
            self.close_search()
        else:
            self.show()
            self.search_input.setFocus()
            self.search_input.selectAll()

    def close_search(self):
        """Close search and clear highlights."""
        self.hide()
        self.search_input.clear()
        self.match_indices = []
        self.current_match = -1
        self.match_label.setText("")
        self._clear_highlights()

    def _on_search_changed(self, text):
        """Search text changed - find all matches."""
        self.match_indices = []
        self.current_match = -1
        self._clear_highlights()

        if not text or len(text) < 2:
            self.match_label.setText("")
            return

        # Find matches in parent's chat scroll area
        chat_area = self._get_chat_area()
        if not chat_area:
            return

        search_lower = text.lower()
        container = chat_area.widget()
        if not container:
            return

        layout = container.layout()
        if not layout:
            return

        for i in range(layout.count()):
            item = layout.itemAt(i)
            if not item or not item.widget():
                continue
            widget = item.widget()
            if not isinstance(widget, MessageWidget):
                continue

            # Check message content
            msg_text = widget.message_data.get('content', '')
            if isinstance(msg_text, list):
                msg_text = ' '.join(p.get('text', '') for p in msg_text if isinstance(p, dict) and p.get('type') == 'text')

            if search_lower in str(msg_text).lower():
                self.match_indices.append(i)

        if self.match_indices:
            self.current_match = 0
            self.match_label.setText(f"1/{len(self.match_indices)}")
            self._highlight_current()
        else:
            self.match_label.setText("0 results")

    def _next_match(self):
        """Go to next match."""
        if not self.match_indices:
            return
        self.current_match = (self.current_match + 1) % len(self.match_indices)
        self.match_label.setText(f"{self.current_match + 1}/{len(self.match_indices)}")
        self._highlight_current()

    def _prev_match(self):
        """Go to previous match."""
        if not self.match_indices:
            return
        self.current_match = (self.current_match - 1) % len(self.match_indices)
        self.match_label.setText(f"{self.current_match + 1}/{len(self.match_indices)}")
        self._highlight_current()

    def _highlight_current(self):
        """Highlight and scroll to current match."""
        if self.current_match < 0 or not self.match_indices:
            return

        chat_area = self._get_chat_area()
        if not chat_area:
            return

        widget_idx = self.match_indices[self.current_match]
        container = chat_area.widget()
        if not container:
            return

        layout = container.layout()
        if not layout:
            return

        item = layout.itemAt(widget_idx)
        if not item or not item.widget():
            return

        target = item.widget()

        # Clear all highlights first
        self._clear_highlights()

        # Add highlight border to matching widgets
        for idx in self.match_indices:
            match_item = layout.itemAt(idx)
            if match_item and match_item.widget():
                w = match_item.widget()
                # Store original style
                if not hasattr(w, '_original_border_style'):
                    w._original_border_style = w.styleSheet()
                current_style = w.styleSheet()
                if idx == widget_idx:
                    # Current match - bright highlight
                    w.setStyleSheet(current_style + f"\nMessageWidget {{ border-right: 3px solid {COLORS['accent_cyan']}; }}")
                else:
                    # Other matches - dim highlight
                    w.setStyleSheet(current_style + f"\nMessageWidget {{ border-right: 2px solid {COLORS['text_dim']}; }}")

        # Scroll to the target widget
        chat_area.ensureWidgetVisible(target, 50, 50)

    def _clear_highlights(self):
        """Clear all search highlights."""
        chat_area = self._get_chat_area()
        if not chat_area:
            return

        container = chat_area.widget()
        if not container:
            return

        layout = container.layout()
        if not layout:
            return

        for i in range(layout.count()):
            item = layout.itemAt(i)
            if not item or not item.widget():
                continue
            w = item.widget()
            if hasattr(w, '_original_border_style'):
                w.setStyleSheet(w._original_border_style)
                del w._original_border_style

    def _get_chat_area(self):
        """Get the ChatScrollArea from parent hierarchy."""
        parent = self.parent()
        if hasattr(parent, 'conversation_display'):
            return parent.conversation_display
        return None


class ChatScrollArea(QScrollArea):
    """
    Scroll area for chat messages with smart auto-scroll behavior.

    ═══════════════════════════════════════════════════════════════════════════
    SCROLL SYSTEM ARCHITECTURE
    ═══════════════════════════════════════════════════════════════════════════

    This widget solves the "chat scroll problem": auto-scroll to bottom for new
    messages, BUT respect when user scrolls up to read history.

    KEY CONCEPTS:
    ─────────────
    • _should_follow: True = auto-scroll to bottom on new content
                      False = user scrolled away, DON'T auto-scroll
    
    • _programmatic_scroll: True = WE are scrolling (ignore in _on_scroll)
                            False = User might be scrolling (track intent)
    
    • Debouncing: Multiple rapid add_message() calls → single scroll after 50ms
    
    STATE TRANSITIONS:
    ──────────────────
    User scrolls UP (away from bottom):
        → _should_follow = False
        → New messages appear but scroll stays put
    
    User scrolls DOWN to bottom:
        → _should_follow = True  
        → New messages trigger auto-scroll
    
    Rebuild (typing indicator → real message):
        → Save _should_follow
        → Block _on_scroll with _programmatic_scroll=True
        → Rebuild widgets
        → Restore _should_follow
        → Only scroll if was following
    
    ═══════════════════════════════════════════════════════════════════════════
    DEBUG OUTPUT GUIDE
    ═══════════════════════════════════════════════════════════════════════════

    Enable debugging: Set config.DEVELOPER_TOOLS = True
        - ChatScrollArea._debug (for [CHAT-SCROLL] messages)
        - ConversationPane._SCROLL_DEBUG (for [SCROLL] messages)

    Filter logs: grep "SCROLL" to see all scroll-related output
    
    ───────────────────────────────────────────────────────────────────────────
    LOG MESSAGE REFERENCE
    ───────────────────────────────────────────────────────────────────────────
    
    [CHAT-SCROLL] User scrolled UP → auto-follow OFF (pos=X/Y)
        ✓ HEALTHY: User scrolled away from bottom to read history.
        • X = current scroll position, Y = maximum scroll position
        • _should_follow is now False
        • New messages will NOT trigger auto-scroll
    
    [CHAT-SCROLL] User scrolled to BOTTOM → auto-follow ON (pos=X/Y)
        ✓ HEALTHY: User returned to bottom of chat.
        • _should_follow is now True  
        • New messages WILL trigger auto-scroll
    
    [CHAT-SCROLL] Auto-scrolled to bottom (max=Z)
        ✓ HEALTHY: Programmatic scroll executed successfully.
        • Z = new maximum scroll position
        • Only logs when position changes by >100px (reduces spam)
    
    [CHAT-SCROLL] ⚠ Scroll retry limit reached (layout still empty)
        ⚠ WARNING: Tried to scroll 5 times but layout never became ready.
        • Usually means widgets aren't being added properly
        • Check if add_message() is being called
    
    [CHAT-SCROLL] Cleared N messages (scroll intent X: _should_follow=Y)
        • N = number of messages removed
        • X = "RESET to follow" or "preserved"
        • If X="preserved" but Y changed unexpectedly, that's a BUG
    
    [SCROLL] Rebuild starting: N messages, _should_follow=X
        • A full widget rebuild is starting (e.g., typing indicator → message)
        • N = message count, X = scroll state being preserved
    
    [SCROLL] Rebuild complete: ACTION
        • Rebuild finished
        • ACTION = "will scroll" or "NO scroll (user scrolled away)"
        • If user had scrolled away but ACTION="will scroll", that's a BUG
    
    ───────────────────────────────────────────────────────────────────────────
    DEBUGGING COMMON ISSUES
    ───────────────────────────────────────────────────────────────────────────
    
    SYMPTOM: Scroll jumps to bottom unexpectedly
        1. Look for "User scrolled UP → auto-follow OFF" - did it fire?
        2. After that, look for any "_should_follow=True" 
        3. Check "Rebuild complete" - should say "NO scroll (user scrolled away)"
        4. Look for "scroll intent RESET" - that resets to following mode!
    
    SYMPTOM: Scroll doesn't follow new messages  
        1. Look for "User scrolled to BOTTOM" - is user actually at bottom?
        2. Check for "Auto-scrolled to bottom" - is it firing?
        3. If you see "⚠ Scroll retry limit reached", layout isn't becoming ready
    
    SYMPTOM: Too much log spam
        1. Normal: One "Auto-scrolled" per significant scroll change
        2. If seeing rapid repeated messages, debouncing may be broken
        3. Check _scroll_timer.setInterval(50) is set
    
    ═══════════════════════════════════════════════════════════════════════════
    """

    ZOOM_MIN = 50
    ZOOM_MAX = 200
    ZOOM_STEP = 10
    ZOOM_DEFAULT = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # ─── Scroll State ───────────────────────────────────────────────────
        self._should_follow = True    # True = auto-scroll to bottom on new content
        self._programmatic_scroll = False  # True = ignore _on_scroll (we're scrolling)
        self._zoom_level = self.ZOOM_DEFAULT  # Current zoom level (50-200%)
        
        # ─── Debug Settings ─────────────────────────────────────────────────
        # Set to True to enable scroll debug logging to console
        # Logs use prefix [CHAT-SCROLL] for easy filtering: grep "CHAT-SCROLL"
        self._debug = DEVELOPER_TOOLS
        
        # ─── Debounce Timer ─────────────────────────────────────────────────
        # Batches rapid scroll requests (e.g., multiple add_message calls)
        self._scroll_timer = QTimer()
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(50)  # 50ms debounce window
        self._scroll_timer.timeout.connect(self._do_scroll_to_bottom)
        
        # ─── Scroll Area Setup ──────────────────────────────────────────────
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # ─── Message Container ──────────────────────────────────────────────
        # Messages are added to this container's layout
        self.container = QWidget()
        self.container.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        self.message_layout = QVBoxLayout(self.container)
        self.message_layout.setContentsMargins(10, 10, 10, 10)
        self.message_layout.setSpacing(8)
        # Align to top - prevents layout from distributing extra space among widgets
        self.message_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.container)
        
        # ─── User Scroll Detection ──────────────────────────────────────────
        # Connect AFTER setup so we don't get spurious signals during init
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
        # ─── Style ──────────────────────────────────────────────────────────
        # Use standardized scrollbar style from styles.py
        self.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
            }}
            {get_scrollbar_style()}
        """)
        
        # Message widgets list
        self.message_widgets = []
    
    def _on_scroll(self, value):
        """
        Track user scroll intent based on scrollbar position.
        
        Called on every scrollbar valueChanged signal. Determines if user
        has scrolled away from bottom (stop auto-follow) or back to bottom
        (resume auto-follow).
        
        Ignored when:
        - _programmatic_scroll is True (we're scrolling, not user)
        - sb.maximum() is 0 (layout empty/rebuilding)
        """
        # Ignore our own programmatic scrolls
        if self._programmatic_scroll:
            return
            
        sb = self.verticalScrollBar()
        # Ignore when layout is empty or rebuilding
        if sb.maximum() == 0:
            return
        
        # "At bottom" = within 30px of maximum scroll position
        at_bottom = value >= sb.maximum() - 30
        
        # Only update and log on state CHANGES (reduces log spam)
        if at_bottom != self._should_follow:
            self._should_follow = at_bottom
            if self._debug:
                if at_bottom:
                    print(f"[CHAT-SCROLL] User scrolled to BOTTOM → auto-follow ON (pos={value}/{sb.maximum()})")
                else:
                    print(f"[CHAT-SCROLL] User scrolled UP → auto-follow OFF (pos={value}/{sb.maximum()})")
    
    def add_message(self, message_data):
        """
        Add a new message widget to the chat.
        
        Appends widget to layout and schedules auto-scroll if:
        - _should_follow is True (user wants to follow new messages)
        - _programmatic_scroll is False (not in a rebuild operation)
        
        Returns the created MessageWidget for potential updates.
        """
        widget = MessageWidget(message_data)
        
        # Simply add to end of layout (no stretch item to work around)
        self.message_layout.addWidget(widget)
        self.message_widgets.append(widget)
        
        # Schedule debounced scroll (if following and not during programmatic operation)
        if self._should_follow and not self._programmatic_scroll:
            self._schedule_scroll()
        
        return widget
    
    def replace_last_message(self, message_data):
        """
        Replace the last message widget with a new one.
        
        Used for smooth transition from typing indicator to real message
        without rebuilding the entire chat. Much faster and no visual flash.
        """
        if self.message_widgets:
            # Remove old widget
            old_widget = self.message_widgets.pop()
            self.message_layout.removeWidget(old_widget)
            old_widget.deleteLater()
            
            # Add new widget
            widget = MessageWidget(message_data)
            self.message_layout.addWidget(widget)
            self.message_widgets.append(widget)
            
            # Schedule scroll if following
            if self._should_follow and not self._programmatic_scroll:
                self._schedule_scroll()
            
            return widget
        else:
            # No widget to replace, just add
            return self.add_message(message_data)
    
    def _schedule_scroll(self):
        """
        Schedule a debounced scroll to bottom.
        
        Uses a 50ms timer to batch multiple rapid calls (e.g., during rebuild).
        Each call restarts the timer, so scroll only happens after calls stop.
        """
        self._scroll_retries = 0  # Reset retry counter for _do_scroll_to_bottom
        self._scroll_timer.start()  # Restart timer (debounces rapid calls)
    
    def _do_scroll_to_bottom(self):
        """
        Actually execute scroll to bottom (called by debounce timer).
        
        May retry up to 5 times if layout isn't ready (sb.maximum() == 0).
        Sets _programmatic_scroll during scroll to prevent _on_scroll from
        misinterpreting our scroll as user scroll.
        """
        if not self._should_follow:
            return
            
        sb = self.verticalScrollBar()
        
        # Layout not ready - retry (max 5 times)
        if sb.maximum() == 0:
            if not hasattr(self, '_scroll_retries'):
                self._scroll_retries = 0
            self._scroll_retries += 1
            if self._scroll_retries < 5:
                QTimer.singleShot(50, self._do_scroll_to_bottom)
            elif self._debug:
                print(f"[CHAT-SCROLL] ⚠ Scroll retry limit reached (layout still empty)")
            return
        
        # Execute scroll with programmatic flag
        self._programmatic_scroll = True
        sb.setValue(sb.maximum())
        self._programmatic_scroll = False

        # Double-tap: schedule another scroll after layout settles
        QTimer.singleShot(0, self._finalize_scroll)

        # Log significant position changes (reduces spam)
        if self._debug:
            if not hasattr(self, '_last_logged_max') or abs(sb.maximum() - self._last_logged_max) > 100:
                print(f"[CHAT-SCROLL] Auto-scrolled to bottom (max={sb.maximum()})")
                self._last_logged_max = sb.maximum()
    
    def _finalize_scroll(self):
        """Second scroll pass to catch late layout changes."""
        if self._should_follow:
            sb = self.verticalScrollBar()
            if sb.value() < sb.maximum():
                self._programmatic_scroll = True
                sb.setValue(sb.maximum())
                self._programmatic_scroll = False

    def clear_messages(self, reset_scroll=False):
        """
        Remove all message widgets from the chat.
        
        Args:
            reset_scroll: If True, also set _should_follow=True (for new conversations).
                         If False (default), preserve current scroll intent.
        
        IMPORTANT: During rebuilds (typing indicator → real message), use
        reset_scroll=False to preserve user's scroll position!
        """
        num_cleared = len(self.message_widgets)
        
        for widget in self.message_widgets:
            self.message_layout.removeWidget(widget)
            widget.deleteLater()
        self.message_widgets.clear()
        
        if reset_scroll:
            self._should_follow = True
            
        if self._debug:
            action = "RESET to follow" if reset_scroll else "preserved"
            print(f"[CHAT-SCROLL] Cleared {num_cleared} messages (scroll intent {action}: _should_follow={self._should_follow})")
    
    def reset_scroll_state(self):
        """Reset to auto-follow mode and scroll to bottom."""
        self._should_follow = True
        self._schedule_scroll()
        if self._debug:
            print("[CHAT-SCROLL] Reset scroll state to follow mode")

    def set_zoom(self, level):
        """Set zoom level (50-200%). Applies font scaling to all messages."""
        level = max(self.ZOOM_MIN, min(self.ZOOM_MAX, level))
        if level == self._zoom_level:
            return
        self._zoom_level = level
        scale = level / 100.0
        base_size = max(7, int(10 * scale))  # Base 10pt, min 7pt

        # Update container font
        font = self.container.font()
        font.setPointSize(base_size)
        self.container.setFont(font)

        # Update all existing message widgets
        for widget in self.message_widgets:
            if hasattr(widget, '_content_label') and widget._content_label:
                wfont = widget._content_label.font()
                wfont.setPointSize(base_size)
                widget._content_label.setFont(wfont)

        # Notify parent for status bar update
        parent = self.parent()
        while parent:
            if hasattr(parent, '_update_zoom_label'):
                parent._update_zoom_label(level)
                break
            parent = parent.parent()

    def zoom_in(self):
        self.set_zoom(self._zoom_level + self.ZOOM_STEP)

    def zoom_out(self):
        self.set_zoom(self._zoom_level - self.ZOOM_STEP)

    def zoom_reset(self):
        self.set_zoom(self.ZOOM_DEFAULT)

    def _scroll_to_bottom(self):
        """Public method to trigger scroll to bottom (debounced)."""
        self._schedule_scroll()
    
    def get_last_message_widget(self):
        """Get the last message widget (for streaming updates)."""
        return self.message_widgets[-1] if self.message_widgets else None


def open_html_file(filepath):
    """Open an HTML file in the default browser. Works cross-platform.
    
    Uses threading to avoid blocking main thread, and debouncing to prevent
    rapid consecutive opens of the same file (which can freeze Windows Shell).
    """
    import threading
    import time
    
    if not os.path.exists(filepath):
        print(f"[GUI] HTML file not found: {filepath}")
        QMessageBox.warning(None, "File Not Found", f"HTML file not found:\n{filepath}\n\nStart a conversation first to generate the HTML.")
        return False
    
    # Debounce: prevent opening the same file within 2 seconds
    if not hasattr(open_html_file, '_last_opened'):
        open_html_file._last_opened = {}
    
    abs_path = os.path.abspath(filepath)
    now = time.time()
    
    # Check if we opened this file recently
    last_time = open_html_file._last_opened.get(abs_path, 0)
    if now - last_time < 2.0:
        print(f"[GUI] Skipping duplicate open request for: {abs_path}")
        return True
    
    open_html_file._last_opened[abs_path] = now
    
    def _do_open():
        """Run the actual file open in a background thread to avoid blocking UI"""
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', abs_path], check=True)
            elif sys.platform == 'win32':  # Windows
                os.startfile(abs_path)
            else:  # Linux
                subprocess.run(['xdg-open', abs_path], check=True)
            
            print(f"[GUI] Opened HTML file: {abs_path}")
        except Exception as e:
            print(f"[GUI] Error opening HTML file: {e}")
            # Fallback to webbrowser module
            try:
                file_url = f"file://{abs_path}"
                webbrowser.open(file_url)
            except Exception as e2:
                print(f"[GUI] Fallback also failed: {e2}")
    
    # Run in background thread to avoid blocking main thread
    # os.startfile() can block on Windows when Shell is busy
    thread = threading.Thread(target=_do_open, daemon=True)
    thread.start()
    return True


def apply_glow_effect(widget, color, blur_radius=15, offset=(0, 2)):
    """Apply a glowing drop shadow effect to a widget"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur_radius)
    shadow.setColor(QColor(color))
    shadow.setOffset(offset[0], offset[1])
    widget.setGraphicsEffect(shadow)
    return shadow


class NoScrollComboBox(QComboBox):
    """A QComboBox that ignores wheel events so parent scroll area can scroll."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def wheelEvent(self, event):
        """Ignore wheel events - let parent scroll instead."""
        event.ignore()


class GlowButton(QPushButton):
    """Enhanced button with glow effect on hover"""
    
    def __init__(self, text, glow_color=COLORS['accent_cyan'], parent=None):
        super().__init__(text, parent)
        self.glow_color = glow_color
        self.base_blur = 8
        self.hover_blur = 20
        
        # Create shadow effect
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(self.base_blur)
        self.shadow.setColor(QColor(glow_color))
        self.shadow.setOffset(0, 2)
        self.setGraphicsEffect(self.shadow)
        
        # Track hover state for animation
        self.setMouseTracking(True)
    
    def enterEvent(self, event):
        """Increase glow on hover"""
        self.shadow.setBlurRadius(self.hover_blur)
        self.shadow.setColor(QColor(self.glow_color))
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Decrease glow when not hovering"""
        self.shadow.setBlurRadius(self.base_blur)
        super().leaveEvent(event)

# Load custom fonts
def load_fonts():
    """Load custom fonts for the application"""
    font_dir = Path("fonts")
    font_dir.mkdir(exist_ok=True)
    
    # List of fonts to load - these would need to be included with the application
    fonts = [
        ("IosevkaTerm-Regular.ttf", "Iosevka Term"),
        ("IosevkaTerm-Bold.ttf", "Iosevka Term"),
        ("IosevkaTerm-Italic.ttf", "Iosevka Term"),
    ]
    
    loaded_fonts = []
    for font_file, font_name in fonts:
        font_path = font_dir / font_file
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id >= 0:
                if font_name not in loaded_fonts:
                    loaded_fonts.append(font_name)
                print(f"Loaded font: {font_name} from {font_file}")
            else:
                print(f"Failed to load font: {font_file}")
        else:
            print(f"Font file not found: {font_path}")
    
    return loaded_fonts


# ═══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERIC EFFECT WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class DepthGauge(QWidget):
    """Vertical gauge showing conversation depth/turn progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_turn = 0
        self.max_turns = 10
        self.setFixedWidth(24)
        self.setMinimumHeight(100)
        
        # Animation
        self.pulse_offset = 0
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._animate_pulse)
        self.pulse_timer.start(50)
        
    def _animate_pulse(self):
        self.pulse_offset = (self.pulse_offset + 2) % 360
        self.update()
    
    def set_progress(self, current, maximum):
        """Update the gauge progress"""
        self.current_turn = current
        self.max_turns = max(maximum, 1)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 4
        gauge_width = w - margin * 2
        gauge_height = h - margin * 2
        
        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS['bg_dark']))
        painter.drawRoundedRect(margin, margin, gauge_width, gauge_height, 4, 4)
        
        # Border
        painter.setPen(QPen(QColor(COLORS['border_glow']), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(margin, margin, gauge_width, gauge_height, 4, 4)
        
        # Calculate fill height (fills from bottom to top)
        progress = min(self.current_turn / self.max_turns, 1.0)
        fill_height = int(gauge_height * progress)
        fill_y = margin + gauge_height - fill_height
        
        if fill_height > 0:
            # Gradient fill
            gradient = QLinearGradient(0, fill_y, 0, margin + gauge_height)
            
            # Color shifts based on depth - deeper = brighter green
            if progress < 0.33:
                gradient.setColorAt(0, QColor('#00FF41'))
                gradient.setColorAt(1, QColor('#006B1A'))
            elif progress < 0.66:
                gradient.setColorAt(0, QColor('#33FF66'))
                gradient.setColorAt(1, QColor('#00FF41'))
            else:
                gradient.setColorAt(0, QColor('#CCFFEE'))
                gradient.setColorAt(1, QColor('#33FF66'))
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(margin + 2, fill_y, gauge_width - 4, fill_height, 2, 2)
            
            # Pulsing glow line at top of fill
            pulse_alpha = int(100 + 80 * math.sin(math.radians(self.pulse_offset)))
            glow_color = QColor('#00FF41')
            glow_color.setAlpha(pulse_alpha)
            painter.setPen(QPen(glow_color, 2))
            painter.drawLine(margin + 2, fill_y, margin + gauge_width - 2, fill_y)
        
        # Turn counter text
        painter.setPen(QColor(COLORS['text_dim']))
        font = painter.font()
        font.setPixelSize(9)
        painter.setFont(font)
        text = f"{self.current_turn}"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)


class SignalIndicator(QWidget):
    """Signal strength/latency indicator"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 20)
        self.signal_strength = 1.0  # 0.0 to 1.0
        self.latency_ms = 0
        self.is_active = False
        
        # Animation for activity
        self.bar_offset = 0
        self.activity_timer = QTimer(self)
        self.activity_timer.timeout.connect(self._animate)
        
    def _animate(self):
        self.bar_offset = (self.bar_offset + 1) % 5
        self.update()
    
    def set_active(self, active):
        """Set whether we're actively waiting for a response"""
        self.is_active = active
        if active:
            self.activity_timer.start(100)
        else:
            self.activity_timer.stop()
        self.update()
    
    def set_latency(self, latency_ms):
        """Update the latency display"""
        self.latency_ms = latency_ms
        # Calculate signal strength based on latency
        if latency_ms < 500:
            self.signal_strength = 1.0
        elif latency_ms < 1500:
            self.signal_strength = 0.75
        elif latency_ms < 3000:
            self.signal_strength = 0.5
        else:
            self.signal_strength = 0.25
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw signal bars
        bar_heights = [4, 7, 10, 13, 16]
        bar_width = 4
        spacing = 2
        start_x = 5
        base_y = 18
        
        for i, bar_h in enumerate(bar_heights):
            x = start_x + i * (bar_width + spacing)
            y = base_y - bar_h
            
            # Determine if this bar should be lit
            threshold = (i + 1) / len(bar_heights)
            is_lit = self.signal_strength >= threshold
            
            if self.is_active:
                # Animated pattern when active
                is_lit = ((i + self.bar_offset) % 5) < 3
                color = QColor(COLORS['accent_cyan']) if is_lit else QColor(COLORS['bg_light'])
            else:
                if is_lit:
                    # Color based on signal strength
                    if self.signal_strength > 0.7:
                        color = QColor(COLORS['accent_green'])
                    elif self.signal_strength > 0.4:
                        color = QColor(COLORS['accent_yellow'])
                    else:
                        color = QColor(COLORS['accent_pink'])
                else:
                    color = QColor(COLORS['bg_light'])
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(x, y, bar_width, bar_h, 1, 1)
        
        # Draw latency text
        painter.setPen(QColor(COLORS['text_dim']))
        font = painter.font()
        font.setPixelSize(9)
        painter.setFont(font)
        
        if self.is_active:
            text = "···"
        elif self.latency_ms > 0:
            text = f"{self.latency_ms}ms"
        else:
            text = ""
        
        painter.drawText(40, 3, 40, 16, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)


class NetworkGraphWidget(QWidget):
    nodeSelected = pyqtSignal(str)
    nodeHovered = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Graph data
        self.nodes = []
        self.edges = []
        self.node_positions = {}
        self.node_colors = {}
        self.node_labels = {}
        self.node_sizes = {}
        
        # Edge animation data
        self.growing_edges = {}  # Dictionary to track growing edges: {(source, target): growth_progress}
        self.edge_growth_speed = 0.05  # Increased speed of edge growth animation (was 0.02)
        
        # Visual settings
        self.margin = 50
        self.selected_node = None
        self.hovered_node = None
        self.animation_progress = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 FPS animation
        
        # Mycelial node settings
        self.hyphae_count = 5  # Number of hyphae per node
        self.hyphae_length_factor = 0.4  # Length of hyphae relative to node radius
        self.hyphae_variation = 0.3  # Random variation in hyphae
        
        # Node colors - use global color palette with mycelial theme
        self.node_colors_by_type = {
            'main': '#8E9DCC',  # Soft blue-purple
            'rabbithole': '#7FB069',  # Soft green
            'fork': '#F2C14E',  # Soft yellow
            'branch': '#F78154'   # Soft orange
        }
        
        # Collision dynamics
        self.node_velocities = {}  # Store velocities for each node
        self.repulsion_strength = 0.5  # Strength of repulsion between nodes
        self.attraction_strength = 0.1  # Strength of attraction along edges
        self.damping = 0.8  # Damping factor to prevent oscillation
        self.apply_physics = True  # Toggle for physics simulation
        
        # Set up the widget
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        
    def add_edge(self, source, target):
        """Add an edge with growth animation"""
        if (source, target) not in self.edges:
            self.edges.append((source, target))
            # Initialize edge growth at 0
            self.growing_edges[(source, target)] = 0.0
            # Force update to start animation immediately
            self.update()
        
    def update_animation(self):
        """Update animation state"""
        self.animation_progress = (self.animation_progress + 0.05) % 1.0
        
        # Update growing edges
        edges_to_remove = []
        has_growing_edges = False
        
        for edge, progress in self.growing_edges.items():
            if progress < 1.0:
                self.growing_edges[edge] = min(progress + self.edge_growth_speed, 1.0)
                has_growing_edges = True
            else:
                # Mark fully grown edges for removal from animation tracking
                edges_to_remove.append(edge)
        
        # Remove fully grown edges from tracking
        for edge in edges_to_remove:
            if edge in self.growing_edges:
                self.growing_edges.pop(edge)
        
        # Apply collision dynamics if enabled
        if self.apply_physics and len(self.nodes) > 1:
            self.apply_collision_dynamics()
        
        # Update the widget
        self.update()
    
    def apply_collision_dynamics(self):
        """Apply collision dynamics to prevent node overlap"""
        # Initialize velocities if needed
        for node_id in self.nodes:
            if node_id not in self.node_velocities:
                self.node_velocities[node_id] = (0, 0)
        
        # Calculate repulsive forces between nodes
        new_velocities = {}
        for node_id in self.nodes:
            if node_id not in self.node_positions:
                continue
                
            vx, vy = self.node_velocities.get(node_id, (0, 0))
            x1, y1 = self.node_positions[node_id]
            
            # Apply repulsion between nodes
            for other_id in self.nodes:
                if other_id == node_id or other_id not in self.node_positions:
                    continue
                    
                x2, y2 = self.node_positions[other_id]
                
                # Calculate distance
                dx = x1 - x2
                dy = y1 - y2
                distance = max(0.1, math.sqrt(dx*dx + dy*dy))  # Avoid division by zero
                
                # Get node sizes
                size1 = math.sqrt(self.node_sizes.get(node_id, 400))
                size2 = math.sqrt(self.node_sizes.get(other_id, 400))
                min_distance = (size1 + size2) / 2
                
                # Apply repulsive force if nodes are too close
                if distance < min_distance * 2:
                    # Normalize direction vector
                    nx = dx / distance
                    ny = dy / distance
                    
                    # Calculate repulsion strength (stronger when closer)
                    strength = self.repulsion_strength * (1.0 - distance / (min_distance * 2))
                    
                    # Apply force
                    vx += nx * strength
                    vy += ny * strength
            
            # Apply attraction along edges
            for edge in self.edges:
                source, target = edge
                
                # Skip edges that are still growing
                if (source, target) in self.growing_edges and self.growing_edges[(source, target)] < 1.0:
                    continue
                
                if source == node_id and target in self.node_positions:
                    # This node is the source, attract towards target
                    x2, y2 = self.node_positions[target]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = max(0.1, math.sqrt(dx*dx + dy*dy))
                    
                    # Normalize and apply attraction
                    vx += (dx / distance) * self.attraction_strength
                    vy += (dy / distance) * self.attraction_strength
                    
                elif target == node_id and source in self.node_positions:
                    # This node is the target, attract towards source
                    x2, y2 = self.node_positions[source]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = max(0.1, math.sqrt(dx*dx + dy*dy))
                    
                    # Normalize and apply attraction
                    vx += (dx / distance) * self.attraction_strength
                    vy += (dy / distance) * self.attraction_strength
            
            # Apply damping to prevent oscillation
            vx *= self.damping
            vy *= self.damping
            
            # Store new velocity
            new_velocities[node_id] = (vx, vy)
        
        # Update positions based on velocities
        for node_id, (vx, vy) in new_velocities.items():
            if node_id in self.node_positions:
                # Skip the main node to keep it centered
                if node_id == 'main':
                    continue
                    
                x, y = self.node_positions[node_id]
                self.node_positions[node_id] = (x + vx, y + vy)
        
        # Update velocities for next frame
        self.node_velocities = new_velocities
        
    def paintEvent(self, event):
        """Paint the network graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Set background with subtle gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor('#1A1A1E'))  # Dark blue-gray
        gradient.setColorAt(1, QColor('#0F0F12'))  # Darker at bottom
        painter.fillRect(0, 0, width, height, gradient)
        
        # Draw subtle grid lines
        painter.setPen(QPen(QColor(COLORS['border']).darker(150), 0.5, Qt.PenStyle.DotLine))
        grid_size = 40
        for x in range(0, width, grid_size):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, grid_size):
            painter.drawLine(0, y, width, y)
        
        # Calculate center point and scale factor
        center_x = width / 2
        center_y = height / 2
        scale = min(width, height) / 500
        
        # Draw edges first so they appear behind nodes
        for edge in self.edges:
            source, target = edge
            if source in self.node_positions and target in self.node_positions:
                src_x, src_y = self.node_positions[source]
                dst_x, dst_y = self.node_positions[target]
                
                # Transform coordinates to screen space
                screen_src_x = center_x + src_x * scale
                screen_src_y = center_y + src_y * scale
                screen_dst_x = center_x + dst_x * scale
                screen_dst_y = center_y + dst_y * scale
                
                # Get growth progress for this edge (default to 1.0 if not growing)
                growth_progress = self.growing_edges.get((source, target), 1.0)
                
                # Calculate the actual destination based on growth progress
                if growth_progress < 1.0:
                    # Interpolate between source and destination
                    actual_dst_x = screen_src_x + (screen_dst_x - screen_src_x) * growth_progress
                    actual_dst_y = screen_src_y + (screen_dst_y - screen_src_y) * growth_progress
                else:
                    actual_dst_x = screen_dst_x
                    actual_dst_y = screen_dst_y
                
                # Draw mycelial connection (multiple thin lines with variations)
                source_color = QColor(self.node_colors.get(source, self.node_colors_by_type['main']))
                target_color = QColor(self.node_colors.get(target, self.node_colors_by_type['main']))
                
                # Number of filaments per connection
                num_filaments = 3
                
                for i in range(num_filaments):
                    # Create a path with multiple segments for organic look
                    path = QPainterPath()
                    path.moveTo(screen_src_x, screen_src_y)
                    
                    # Calculate distance between points
                    distance = math.sqrt((actual_dst_x - screen_src_x)**2 + (actual_dst_y - screen_src_y)**2)
                    
                    # Number of segments increases with distance
                    num_segments = max(3, int(distance / 40))
                    
                    # Create intermediate points with slight random variations
                    prev_x, prev_y = screen_src_x, screen_src_y
                    
                    for j in range(1, num_segments):
                        # Calculate position along the line
                        ratio = j / num_segments
                        
                        # Base position
                        base_x = screen_src_x + (actual_dst_x - screen_src_x) * ratio
                        base_y = screen_src_y + (actual_dst_y - screen_src_y) * ratio
                        
                        # Add random variation perpendicular to the line
                        angle = math.atan2(actual_dst_y - screen_src_y, actual_dst_x - screen_src_x) + math.pi/2
                        variation = (random.random() - 0.5) * 10 * scale
                        
                        # Variation decreases near endpoints
                        endpoint_factor = min(ratio, 1 - ratio) * 4  # Maximum at middle
                        variation *= endpoint_factor
                        
                        # Apply variation
                        point_x = base_x + variation * math.cos(angle)
                        point_y = base_y + variation * math.sin(angle)
                        
                        # Add point to path
                        path.lineTo(point_x, point_y)
                        prev_x, prev_y = point_x, point_y
                    
                    # Complete the path to destination
                    path.lineTo(actual_dst_x, actual_dst_y)
                    
                    # Create gradient along the path
                    gradient = QLinearGradient(screen_src_x, screen_src_y, actual_dst_x, actual_dst_y)
                    
                    # Make colors more transparent for mycelial effect
                    source_color_trans = QColor(source_color)
                    target_color_trans = QColor(target_color)
                    
                    # Vary transparency by filament
                    alpha = 70 + i * 20
                    source_color_trans.setAlpha(alpha)
                    target_color_trans.setAlpha(alpha)
                    
                    gradient.setColorAt(0, source_color_trans)
                    gradient.setColorAt(1, target_color_trans)
                    
                    # Animate flow along edge
                    flow_pos = (self.animation_progress + i * 0.3) % 1.0
                    flow_color = QColor(255, 255, 255, 100)
                    gradient.setColorAt(flow_pos, flow_color)
                    
                    # Draw the edge with varying thickness
                    thickness = 1.0 + (i * 0.5)
                    pen = QPen(QBrush(gradient), thickness)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.drawPath(path)
                
                # Draw small nodes along the path for mycelial effect
                if growth_progress == 1.0:  # Only for fully grown edges
                    num_nodes = int(distance / 50)
                    for j in range(1, num_nodes):
                        ratio = j / num_nodes
                        node_x = screen_src_x + (screen_dst_x - screen_src_x) * ratio
                        node_y = screen_src_y + (screen_dst_y - screen_src_y) * ratio
                        
                        # Add small random offset
                        offset_angle = random.random() * math.pi * 2
                        offset_dist = random.random() * 5
                        node_x += math.cos(offset_angle) * offset_dist
                        node_y += math.sin(offset_angle) * offset_dist
                        
                        # Draw small node
                        node_color = QColor(source_color)
                        node_color.setAlpha(100)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QBrush(node_color))
                        node_size = 1 + random.random() * 2
                        painter.drawEllipse(QPointF(node_x, node_y), node_size, node_size)
        
        # Draw nodes
        for node_id in self.nodes:
            if node_id in self.node_positions:
                x, y = self.node_positions[node_id]
                
                # Transform coordinates to screen space
                screen_x = center_x + x * scale
                screen_y = center_y + y * scale
                
                # Get node properties
                node_color = self.node_colors.get(node_id, self.node_colors_by_type['branch'])
                node_label = self.node_labels.get(node_id, 'Node')
                node_size = self.node_sizes.get(node_id, 400)
                
                # Scale the node size
                radius = math.sqrt(node_size) * scale / 2
                
                # Adjust radius for hover/selection
                if node_id == self.selected_node:
                    radius *= 1.1  # Larger when selected
                elif node_id == self.hovered_node:
                    radius *= 1.05  # Slightly larger when hovered
                
                # Draw node glow for selected/hovered nodes
                if node_id == self.selected_node or node_id == self.hovered_node:
                    glow_radius = radius * 1.5
                    glow_color = QColor(node_color)
                    
                    for i in range(5):
                        r = glow_radius - (i * radius * 0.1)
                        alpha = 40 - (i * 8)
                        glow_color.setAlpha(alpha)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(glow_color)
                        painter.drawEllipse(QPointF(screen_x, screen_y), r, r)
                
                # Draw mycelial node (irregular shape with hyphae)
                painter.setPen(Qt.PenStyle.NoPen)
                
                # Create gradient fill for node
                gradient = QRadialGradient(screen_x, screen_y, radius)
                base_color = QColor(node_color)
                lighter_color = QColor(node_color).lighter(130)
                darker_color = QColor(node_color).darker(130)
                
                gradient.setColorAt(0, lighter_color)
                gradient.setColorAt(0.7, base_color)
                gradient.setColorAt(1, darker_color)
                
                # Fill main node body
                painter.setBrush(QBrush(gradient))
                
                # Draw irregular node shape
                path = QPainterPath()
                
                # Create irregular circle with random variations
                num_points = 20
                start_angle = random.random() * math.pi * 2
                
                for i in range(num_points + 1):
                    angle = start_angle + (i * 2 * math.pi / num_points)
                    # Vary radius slightly for organic look
                    variation = 1.0 + (random.random() - 0.5) * 0.2
                    point_radius = radius * variation
                    
                    x_point = screen_x + math.cos(angle) * point_radius
                    y_point = screen_y + math.sin(angle) * point_radius
                    
                    if i == 0:
                        path.moveTo(x_point, y_point)
                    else:
                        # Use quadratic curves for smoother shape
                        control_angle = start_angle + ((i - 0.5) * 2 * math.pi / num_points)
                        control_radius = radius * (1.0 + (random.random() - 0.5) * 0.1)
                        control_x = screen_x + math.cos(control_angle) * control_radius
                        control_y = screen_y + math.sin(control_angle) * control_radius
                        
                        path.quadTo(control_x, control_y, x_point, y_point)
                
                # Draw the main node body
                painter.drawPath(path)
                
                # Draw hyphae (mycelial extensions)
                hyphae_count = self.hyphae_count
                if node_id == 'main':
                    hyphae_count += 3  # More hyphae for main node
                
                for i in range(hyphae_count):
                    # Random angle for hyphae
                    angle = random.random() * math.pi * 2
                    
                    # Base length varies by node type
                    base_length = radius * self.hyphae_length_factor
                    if node_id == 'main':
                        base_length *= 1.5
                    
                    # Random variation in length
                    length = base_length * (1.0 + (random.random() - 0.5) * self.hyphae_variation)
                    
                    # Calculate end point
                    end_x = screen_x + math.cos(angle) * (radius + length)
                    end_y = screen_y + math.sin(angle) * (radius + length)
                    
                    # Start point is on the node perimeter
                    start_x = screen_x + math.cos(angle) * radius * 0.9
                    start_y = screen_y + math.sin(angle) * radius * 0.9
                    
                    # Create hyphae path with slight curve
                    hypha_path = QPainterPath()
                    hypha_path.moveTo(start_x, start_y)
                    
                    # Control point for curve
                    ctrl_angle = angle + (random.random() - 0.5) * 0.5  # Slight angle variation
                    ctrl_dist = radius + length * 0.5
                    ctrl_x = screen_x + math.cos(ctrl_angle) * ctrl_dist
                    ctrl_y = screen_y + math.sin(ctrl_angle) * ctrl_dist
                    
                    hypha_path.quadTo(ctrl_x, ctrl_y, end_x, end_y)
                    
                    # Draw hypha with gradient
                    hypha_gradient = QLinearGradient(start_x, start_y, end_x, end_y)
                    
                    # Hypha color starts as node color and fades out
                    hypha_start_color = QColor(node_color)
                    hypha_end_color = QColor(node_color)
                    hypha_start_color.setAlpha(150)
                    hypha_end_color.setAlpha(30)
                    
                    hypha_gradient.setColorAt(0, hypha_start_color)
                    hypha_gradient.setColorAt(1, hypha_end_color)
                    
                    # Draw hypha with varying thickness
                    thickness = 1.0 + random.random() * 1.5
                    hypha_pen = QPen(QBrush(hypha_gradient), thickness)
                    hypha_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(hypha_pen)
                    painter.drawPath(hypha_path)
                    
                    # Add small nodes at the end of some hyphae
                    if random.random() > 0.5:
                        small_node_color = QColor(node_color)
                        small_node_color.setAlpha(100)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QBrush(small_node_color))
                        small_node_size = 1 + random.random() * 2
                        painter.drawEllipse(QPointF(end_x, end_y), small_node_size, small_node_size)
    
    def draw_arrow_head(self, painter, x1, y1, x2, y2):
        """Draw an arrow head at the end of a line"""
        # For mycelial style, we don't need arrow heads
        pass
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position
            pos = event.position()
            
            # Check if a node was clicked
            clicked_node = self.get_node_at_position(pos)
            if clicked_node:
                self.selected_node = clicked_node
                self.update()
                self.nodeSelected.emit(clicked_node)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover effects"""
        pos = event.position()
        hovered_node = self.get_node_at_position(pos)
        
        if hovered_node != self.hovered_node:
            self.hovered_node = hovered_node
            self.update()
            if hovered_node:
                self.nodeHovered.emit(hovered_node)
                
                # Show tooltip with node info
                if hovered_node in self.node_labels:
                    # Get node type from the ID
                    node_type = "main"
                    if "rabbithole_" in hovered_node:
                        node_type = "rabbithole"
                    elif "fork_" in hovered_node:
                        node_type = "fork"
                    
                    # Set emoji based on node type
                    emoji = "🌱"  # Default/main
                    if node_type == "rabbithole":
                        emoji = "🕳️"  # Rabbithole emoji
                    elif node_type == "fork":
                        emoji = "🔱"  # Fork emoji
                    
                    # Show tooltip with emoji and label
                    QToolTip.showText(
                        event.globalPosition().toPoint(),
                        f"{emoji} {self.node_labels[hovered_node]}",
                        self
                    )
    
    def get_node_at_position(self, pos):
        """Get the node at the given position"""
        # Calculate center point and scale factor
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        scale = min(width, height) / 500
                    
        # Check each node
        for node_id in self.nodes:
            if node_id in self.node_positions:
                    x, y = self.node_positions[node_id]
                    screen_x = center_x + x * scale
                    screen_y = center_y + y * scale
                    
                    # Get node size
                    node_size = self.node_sizes.get(node_id, 400)
                    radius = math.sqrt(node_size) * scale / 2
                    
            # Check if click is inside the node
                    distance = math.sqrt((pos.x() - screen_x)**2 + (pos.y() - screen_y)**2)
                    if distance <= radius:
                        return node_id
        
        return None
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        self.update()

class NetworkPane(QWidget):
    nodeSelected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title with consistent tab header styling
        title = QLabel("PROPAGATION NETWORK")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 13px;
            font-weight: bold;
            padding: 12px;
            background-color: {COLORS['bg_medium']};
            border-bottom: 1px solid {COLORS['border_glow']};
            letter-spacing: 3px;
            text-transform: uppercase;
        """)
        layout.addWidget(title)
        
        # Network view - set to expand to fill available space
        self.network_view = NetworkGraphWidget()
        self.network_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.network_view, 1)  # Add stretch factor of 1 to make it expand
        
        # Connect signals
        self.network_view.nodeSelected.connect(self.nodeSelected)
    
        # Initialize graph
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_colors = {}
        self.node_labels = {}
        self.node_sizes = {}
        
        # Add main node
        self.add_node('main', 'Seed', 'main')
    
    def add_node(self, node_id, label, node_type='branch'):
        """Add a node to the graph"""
        try:
            # Add the node to the graph
            self.graph.add_node(node_id)
            
            # Set node properties based on type
            if node_type == 'main':
                color = '#569CD6'  # Blue
                size = 800
            elif node_type == 'rabbithole':
                color = '#B5CEA8'  # Green
                size = 600
            elif node_type == 'fork':
                color = '#DCDCAA'  # Yellow
                size = 600
            else:
                color = '#CE9178'  # Orange
                size = 400
            
            # Store node properties
            self.node_colors[node_id] = color
            self.node_labels[node_id] = label
            self.node_sizes[node_id] = size
            
            # Calculate position based on existing nodes
            self.calculate_node_position(node_id, node_type)
            
            # Redraw the graph
            self.update_graph()
            
        except Exception as e:
            print(f"Error adding node: {e}")
    
    def add_edge(self, source_id, target_id):
        """Add an edge between two nodes"""
        try:
            # Add the edge to the graph
            self.graph.add_edge(source_id, target_id)
            
            # Redraw the graph
            self.update_graph()
            
        except Exception as e:
            print(f"Error adding edge: {e}")
    
    def calculate_node_position(self, node_id, node_type):
        """Calculate position for a new node"""
        # Get number of existing nodes
        num_nodes = len(self.graph.nodes) - 1  # Exclude the main node
        
        if node_type == 'main':
            # Main node is at center
            self.node_positions[node_id] = (0, 0)
        else:
            # Calculate angle based on node count with better distribution
            # Use golden ratio to distribute nodes more evenly
            golden_ratio = 1.618033988749895
            angle = 2 * math.pi * golden_ratio * num_nodes
            
            # Calculate distance from center based on node type and node count
            # Increase distance as more nodes are added
            base_distance = 200
            count_factor = min(1.0, num_nodes / 20)  # Scale up to 20 nodes
            
            if node_type == 'rabbithole':
                distance = base_distance * (1.0 + count_factor * 0.5)
            elif node_type == 'fork':
                distance = base_distance * (1.2 + count_factor * 0.5)
            else:
                distance = base_distance * (1.4 + count_factor * 0.5)
            
            # Calculate position using polar coordinates
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            # Add some random offset for natural appearance
            x += random.uniform(-30, 30)
            y += random.uniform(-30, 30)
            
            # Check for potential overlaps with existing nodes and adjust if needed
            overlap = True
            max_attempts = 5
            attempt = 0
            
            while overlap and attempt < max_attempts:
                overlap = False
                for existing_id, (ex, ey) in self.node_positions.items():
                    # Skip comparing with self
                    if existing_id == node_id:
                        continue
                        
                    # Calculate distance between nodes
                    dx = x - ex
                    dy = y - ey
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Get node sizes
                    new_size = math.sqrt(self.node_sizes.get(node_id, 400))
                    existing_size = math.sqrt(self.node_sizes.get(existing_id, 400))
                    min_distance = (new_size + existing_size) / 2
                    
                    # If too close, adjust position
                    if distance < min_distance * 1.5:
                        overlap = True
                        # Move away from the overlapping node
                        angle = math.atan2(dy, dx)
                        adjustment = min_distance * 1.5 - distance
                        x += math.cos(angle) * adjustment * 1.2
                        y += math.sin(angle) * adjustment * 1.2
                        break
                
                attempt += 1
            
            # Store the position
            self.node_positions[node_id] = (x, y)
    
    def update_graph(self):
        """Update the network graph visualization"""
        if hasattr(self, 'network_view'):
            # Update the network view with current graph data
            self.network_view.nodes = list(self.graph.nodes())
            self.network_view.edges = list(self.graph.edges())
            self.network_view.node_positions = self.node_positions
            self.network_view.node_colors = self.node_colors
            self.network_view.node_labels = self.node_labels
            self.network_view.node_sizes = self.node_sizes
            
            # Redraw
            self.network_view.update()

class ImagePreviewPane(QWidget):
    """Pane to display generated images with navigation"""
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.session_images = []  # List of all images generated this session
        self.session_metadata = []  # List of metadata dicts {ai_name, prompt} for each image
        self.current_index = -1   # Current image index
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title with consistent tab header styling
        self.title = QLabel("GENERATED IMAGES")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 13px;
            font-weight: bold;
            padding: 12px;
            background-color: {COLORS['bg_medium']};
            border-bottom: 1px solid {COLORS['border_glow']};
            letter-spacing: 3px;
            text-transform: uppercase;
        """)
        layout.addWidget(self.title)
        
        # AI name label (below title)
        self.ai_label = QLabel("")
        self.ai_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_bright']};
                font-size: 10px;
                font-weight: bold;
                padding: 2px 5px;
            }}
        """)
        self.ai_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.ai_label)
        
        # Prompt label (below AI name)
        self.prompt_label = QLabel("")
        self.prompt_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 11px;
                font-style: italic;
                padding: 2px 5px;
            }}
        """)
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setWordWrap(True)
        layout.addWidget(self.prompt_label)
        
        # Image display label
        self.image_label = QLabel("No images generated yet")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                padding: 20px;
                min-height: 200px;
            }}
        """)
        self.image_label.setWordWrap(True)
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label, 1)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)
        
        # Previous button
        self.prev_button = QPushButton("◀ Prev")
        self.prev_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_cyan']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_dim']};
                background-color: {COLORS['bg_dark']};
            }}
        """)
        self.prev_button.clicked.connect(self.show_previous)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        
        # Position indicator
        self.position_label = QLabel("")
        self.position_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 11px;
            }}
        """)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.position_label, 1)
        
        # Next button
        self.next_button = QPushButton("Next ▶")
        self.next_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_cyan']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_dim']};
                background-color: {COLORS['bg_dark']};
            }}
        """)
        self.next_button.clicked.connect(self.show_next)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
        layout.addLayout(nav_layout)
        
        # Image info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 10px;
                padding: 5px;
            }}
        """)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Open in folder button
        self.open_button = QPushButton("📂 Open Images Folder")
        self.open_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_cyan']};
            }}
        """)
        self.open_button.clicked.connect(self.open_images_folder)
        layout.addWidget(self.open_button)
    
    def add_image(self, image_path, ai_name="", prompt=""):
        """Add a new image to the session gallery and display it"""
        if image_path and os.path.exists(image_path):
            # Avoid duplicates
            if image_path not in self.session_images:
                self.session_images.append(image_path)
                self.session_metadata.append({"ai_name": ai_name, "prompt": prompt})
            # Jump to the new image
            self.current_index = len(self.session_images) - 1
            self._display_current()
    
    def set_image(self, image_path, ai_name="", prompt=""):
        """Display an image - also adds to gallery if new"""
        self.add_image(image_path, ai_name, prompt)
    
    def _display_current(self):
        """Display the image at current_index"""
        if not self.session_images or self.current_index < 0:
            self.image_label.setText("No images generated yet")
            self.info_label.setText("")
            self.position_label.setText("")
            self.ai_label.setText("")
            self.prompt_label.setText("")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        
        image_path = self.session_images[self.current_index]
        self.current_image_path = image_path
        
        # Get metadata for this image
        metadata = self.session_metadata[self.current_index] if self.current_index < len(self.session_metadata) else {}
        ai_name = metadata.get("ai_name", "")
        prompt = metadata.get("prompt", "")
        
        # Update AI name and prompt labels
        self.ai_label.setText(ai_name if ai_name else "")
        self.prompt_label.setText(f'"{prompt}"' if prompt else "")
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale to fit the label while maintaining aspect ratio
                scaled = pixmap.scaled(
                    self.image_label.size() - QSize(20, 20),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
                self.image_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {COLORS['bg_medium']};
                        border: 1px solid {COLORS['border']};
                        padding: 10px;
                    }}
                """)
                
                # Update info
                filename = os.path.basename(image_path)
                self.info_label.setText(filename)
            else:
                self.image_label.setText("Failed to load image")
                self.info_label.setText("")
        else:
            self.image_label.setText("Image not found")
            self.info_label.setText("")
        
        # Update navigation
        total = len(self.session_images)
        current = self.current_index + 1
        self.position_label.setText(f"{current} of {total}")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < total - 1)
    
    def show_previous(self):
        """Show the previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current()
    
    def show_next(self):
        """Show the next image"""
        if self.current_index < len(self.session_images) - 1:
            self.current_index += 1
            self._display_current()
    
    def clear_session(self):
        """Clear all session images (e.g., when starting a new conversation)"""
        self.session_images = []
        self.session_metadata = []
        self.current_index = -1
        self.current_image_path = None
        self.image_label.setText("No images generated yet")
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                padding: 20px;
                min-height: 200px;
            }}
        """)
        self.info_label.setText("")
        self.position_label.setText("")
        self.ai_label.setText("")
        self.prompt_label.setText("")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
    
    def open_images_folder(self):
        """Open the images folder in file explorer"""
        import subprocess
        import sys
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        os.makedirs(images_dir, exist_ok=True)
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', images_dir])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', images_dir])
        else:
            subprocess.Popen(['xdg-open', images_dir])
    
    def resizeEvent(self, event):
        """Re-scale image when pane is resized"""
        super().resizeEvent(event)
        if self.current_image_path:
            self._display_current()


class VideoPreviewPane(QWidget):
    """Pane to display generated videos with navigation"""
    def __init__(self):
        super().__init__()
        self.current_video_path = None
        self.session_videos = []  # List of all videos generated this session
        self.session_metadata = []  # List of metadata dicts {ai_name, prompt} for each video
        self.current_index = -1   # Current video index
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title with consistent tab header styling
        self.title = QLabel("GENERATED VIDEOS")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 13px;
            font-weight: bold;
            padding: 12px;
            background-color: {COLORS['bg_medium']};
            border-bottom: 1px solid {COLORS['border_glow']};
            letter-spacing: 3px;
            text-transform: uppercase;
        """)
        layout.addWidget(self.title)
        
        # AI name label (below title)
        self.ai_label = QLabel("")
        self.ai_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_bright']};
                font-size: 10px;
                font-weight: bold;
                padding: 2px 5px;
            }}
        """)
        self.ai_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.ai_label)
        
        # Prompt label (below AI name)
        self.prompt_label = QLabel("")
        self.prompt_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 11px;
                font-style: italic;
                padding: 2px 5px;
            }}
        """)
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setWordWrap(True)
        layout.addWidget(self.prompt_label)
        
        # Video display area - we'll show a thumbnail or placeholder
        self.video_label = QLabel("No videos generated yet")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                padding: 20px;
                min-height: 150px;
            }}
        """)
        self.video_label.setWordWrap(True)
        layout.addWidget(self.video_label, 1)
        
        # Play button
        self.play_button = QPushButton("▶ Play Video")
        self.play_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
                border: none;
                border-radius: 0px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_purple']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
            }}
        """)
        self.play_button.clicked.connect(self.play_current_video)
        self.play_button.setEnabled(False)
        layout.addWidget(self.play_button)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)
        
        # Previous button
        self.prev_button = QPushButton("◀ Prev")
        self.prev_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_cyan']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_dim']};
                background-color: {COLORS['bg_dark']};
            }}
        """)
        self.prev_button.clicked.connect(self.show_previous)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        
        # Position indicator
        self.position_label = QLabel("")
        self.position_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 11px;
            }}
        """)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.position_label, 1)
        
        # Next button
        self.next_button = QPushButton("Next ▶")
        self.next_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_cyan']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_dim']};
                background-color: {COLORS['bg_dark']};
            }}
        """)
        self.next_button.clicked.connect(self.show_next)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
        layout.addLayout(nav_layout)
        
        # Video info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 10px;
                padding: 5px;
            }}
        """)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Open in folder button
        self.open_button = QPushButton("📂 Open Videos Folder")
        self.open_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_cyan']};
            }}
        """)
        self.open_button.clicked.connect(self.open_videos_folder)
        layout.addWidget(self.open_button)
    
    def add_video(self, video_path, ai_name="", prompt=""):
        """Add a new video to the session gallery and display it"""
        if video_path and os.path.exists(video_path):
            # Avoid duplicates
            if video_path not in self.session_videos:
                self.session_videos.append(video_path)
                self.session_metadata.append({"ai_name": ai_name, "prompt": prompt})
            # Jump to the new video
            self.current_index = len(self.session_videos) - 1
            self._display_current()
    
    def set_video(self, video_path, ai_name="", prompt=""):
        """Display a video - also adds to gallery if new"""
        self.add_video(video_path, ai_name, prompt)
    
    def _display_current(self):
        """Display the video at current_index"""
        if not self.session_videos or self.current_index < 0:
            self.video_label.setText("No videos generated yet")
            self.info_label.setText("")
            self.position_label.setText("")
            self.ai_label.setText("")
            self.prompt_label.setText("")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.play_button.setEnabled(False)
            return
        
        video_path = self.session_videos[self.current_index]
        self.current_video_path = video_path
        
        # Get metadata for this video
        metadata = self.session_metadata[self.current_index] if self.current_index < len(self.session_metadata) else {}
        ai_name = metadata.get("ai_name", "")
        prompt = metadata.get("prompt", "")
        
        # Update AI name and prompt labels
        self.ai_label.setText(ai_name if ai_name else "")
        self.prompt_label.setText(f'"{prompt}"' if prompt else "")
        
        if os.path.exists(video_path):
            filename = os.path.basename(video_path)
            # Show video info
            self.video_label.setText(f"🎬 {filename}\n\n(Click Play to view)")
            self.video_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS['bg_medium']};
                    border: 1px solid {COLORS['border']};
                    color: {COLORS['text_bright']};
                    padding: 20px;
                    min-height: 150px;
                }}
            """)
            self.info_label.setText(filename)
            self.play_button.setEnabled(True)
        else:
            self.video_label.setText("Video not found")
            self.info_label.setText("")
            self.play_button.setEnabled(False)
        
        # Update navigation
        total = len(self.session_videos)
        current = self.current_index + 1
        self.position_label.setText(f"{current} of {total}")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < total - 1)
    
    def show_previous(self):
        """Show the previous video"""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current()
    
    def show_next(self):
        """Show the next video"""
        if self.current_index < len(self.session_videos) - 1:
            self.current_index += 1
            self._display_current()
    
    def play_current_video(self):
        """Open the current video in the default video player"""
        if self.current_video_path and os.path.exists(self.current_video_path):
            import subprocess
            import sys
            if sys.platform == 'win32':
                os.startfile(self.current_video_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', self.current_video_path])
            else:
                subprocess.Popen(['xdg-open', self.current_video_path])
    
    def clear_session(self):
        """Clear all session videos (e.g., when starting a new conversation)"""
        self.session_videos = []
        self.session_metadata = []
        self.current_index = -1
        self.current_video_path = None
        self.video_label.setText("No videos generated yet")
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                padding: 20px;
                min-height: 150px;
            }}
        """)
        self.info_label.setText("")
        self.position_label.setText("")
        self.ai_label.setText("")
        self.prompt_label.setText("")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.play_button.setEnabled(False)
    
    def open_videos_folder(self):
        """Open the videos folder in file explorer"""
        import subprocess
        import sys
        videos_dir = os.path.join(os.path.dirname(__file__), 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', videos_dir])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', videos_dir])
        else:
            subprocess.Popen(['xdg-open', videos_dir])


class StatsWidget(QWidget):
    """Live conversation statistics display"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stats = {
            'turns': 0,
            'active_ais': [],
            'est_tokens': 0,
            'avg_response_ms': 0,
            'images_generated': 0,
            'commands_executed': 0,
            'response_times': [],
        }
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Title
        title = QLabel("LIVE STATS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 13px;
            font-weight: bold;
            padding: 12px;
            background-color: {COLORS['bg_medium']};
            border-bottom: 1px solid {COLORS['border_glow']};
            letter-spacing: 3px;
        """)
        layout.addWidget(title)

        # Stats grid
        stats_container = QWidget()
        stats_container.setStyleSheet(f"background-color: transparent;")
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setSpacing(8)

        # Create stat rows
        self.stat_labels = {}
        stat_defs = [
            ('turns', 'TURNS', '0'),
            ('active_ais', 'ACTIVE AIs', '0'),
            ('est_tokens', 'EST. TOKENS', '0'),
            ('words', 'WORDS', '0'),
            ('avg_response', 'AVG RESPONSE', '\u2014'),
            ('images', 'IMAGES', '0'),
            ('commands', 'COMMANDS', '0'),
        ]

        for key, label_text, default in stat_defs:
            row = QWidget()
            row.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-left: 2px solid {COLORS['border_glow']};")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)

            name_label = QLabel(label_text)
            name_label.setStyleSheet(f"background-color: transparent; color: {COLORS['text_dim']}; font-size: 9px; font-weight: bold; letter-spacing: 1px;")

            value_label = QLabel(default)
            value_label.setStyleSheet(f"background-color: transparent; color: {COLORS['accent_cyan']}; font-size: 12px; font-weight: bold;")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row_layout.addWidget(name_label)
            row_layout.addStretch()
            row_layout.addWidget(value_label)

            stats_layout.addWidget(row)
            self.stat_labels[key] = value_label

        layout.addWidget(stats_container)
        layout.addStretch()

    def update_stats(self, conversation, images_count=0, commands_count=0):
        """Update stats from conversation data"""
        if not conversation:
            return

        # Count turns (assistant messages only)
        turns = sum(1 for m in conversation if m.get('role') == 'assistant')
        self.stat_labels['turns'].setText(str(turns))

        # Active AIs
        ai_names = set()
        for m in conversation:
            name = m.get('ai_name', '')
            if name:
                ai_names.add(name)
        self.stat_labels['active_ais'].setText(str(len(ai_names)))

        # Estimate tokens (~4 chars per token)
        total_chars = 0
        for m in conversation:
            content = m.get('content', '')
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        total_chars += len(part.get('text', ''))
        est_tokens = total_chars // 4
        if est_tokens > 1000:
            self.stat_labels['est_tokens'].setText(f"{est_tokens/1000:.1f}k")
        else:
            self.stat_labels['est_tokens'].setText(str(est_tokens))

        # Word count
        total_words = 0
        for m in conversation:
            content = m.get('content', '')
            if isinstance(content, str):
                total_words += len(content.split())
            elif isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        total_words += len(part.get('text', '').split())
        if total_words > 1000:
            self.stat_labels['words'].setText(f"{total_words/1000:.1f}k")
        else:
            self.stat_labels['words'].setText(str(total_words))

        # Response time
        if self.stats['response_times']:
            avg = sum(self.stats['response_times']) / len(self.stats['response_times'])
            if avg > 1000:
                self.stat_labels['avg_response'].setText(f"{avg/1000:.1f}s")
            else:
                self.stat_labels['avg_response'].setText(f"{int(avg)}ms")

        # Images and commands
        self.stat_labels['images'].setText(str(images_count))
        self.stat_labels['commands'].setText(str(commands_count))

    def record_response_time(self, ms):
        """Record a response time for averaging"""
        self.stats['response_times'].append(ms)
        # Keep last 20 for rolling average
        if len(self.stats['response_times']) > 20:
            self.stats['response_times'] = self.stats['response_times'][-20:]


class RightSidebar(QWidget):
    """Right sidebar with tabbed interface for Setup and Network Graph"""
    nodeSelected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(300)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the tabbed sidebar interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Match left panel padding
        layout.setSpacing(0)
        
        # Create tab bar at the top (custom styled)
        tab_container = QWidget()
        tab_container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_medium']};
                border-bottom: 1px solid {COLORS['border_glow']};
            }}
        """)
        tab_layout = QHBoxLayout(tab_container)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)
        
        # Tab buttons
        self.setup_button = QPushButton("⚙ SETUP")
        self.graph_button = QPushButton("🌐 GRAPH")
        self.image_button = QPushButton("🖼 IMAGES")
        self.video_button = QPushButton("🎬 VIDEOS")
        self.stats_button = QPushButton("📊 STATS")

        # Cyberpunk tab button styling
        tab_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                border: none;
                border-bottom: 2px solid transparent;
                padding: 12px 12px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
                text-transform: uppercase;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border-bottom: 2px solid {COLORS['accent_cyan']};
            }}
        """
        
        self.setup_button.setStyleSheet(tab_style)
        self.graph_button.setStyleSheet(tab_style)
        self.image_button.setStyleSheet(tab_style)
        self.video_button.setStyleSheet(tab_style)
        self.stats_button.setStyleSheet(tab_style)

        # Make buttons checkable for tab behavior
        self.setup_button.setCheckable(True)
        self.graph_button.setCheckable(True)
        self.image_button.setCheckable(True)
        self.video_button.setCheckable(True)
        self.stats_button.setCheckable(True)
        self.setup_button.setChecked(True)  # Start with setup tab active
        
        # Connect tab buttons
        self.setup_button.clicked.connect(lambda: self.switch_tab(0))
        self.graph_button.clicked.connect(lambda: self.switch_tab(1))
        self.image_button.clicked.connect(lambda: self.switch_tab(2))
        self.video_button.clicked.connect(lambda: self.switch_tab(3))
        self.stats_button.clicked.connect(lambda: self.switch_tab(4))

        tab_layout.addWidget(self.setup_button)
        tab_layout.addWidget(self.graph_button)
        tab_layout.addWidget(self.image_button)
        tab_layout.addWidget(self.video_button)
        tab_layout.addWidget(self.stats_button)

        # Settings button
        tab_layout.addStretch()
        self.settings_button = QPushButton("⚙ KEYS")
        self.settings_button.setStyleSheet(tab_style.replace(
            "border-bottom: 2px solid transparent;",
            "border-bottom: 2px solid transparent; border-left: 1px solid #333;"
        ))
        self.settings_button.setToolTip("API Keys & Provider Routing")
        self.settings_button.clicked.connect(self._open_settings)
        tab_layout.addWidget(self.settings_button)

        layout.addWidget(tab_container)
        
        # Create stacked widget for tab content
        from PyQt6.QtWidgets import QStackedWidget
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"""
            QStackedWidget {{
                background-color: {COLORS['bg_dark']};
                border: none;
            }}
        """)
        
        # Create tab pages
        self.control_panel = ControlPanel()
        self.network_pane = NetworkPane()
        self.image_preview_pane = ImagePreviewPane()
        self.video_preview_pane = VideoPreviewPane()
        self.stats_widget = StatsWidget()

        # Add pages to stack
        self.stack.addWidget(self.control_panel)
        self.stack.addWidget(self.network_pane)
        self.stack.addWidget(self.image_preview_pane)
        self.stack.addWidget(self.video_preview_pane)
        self.stack.addWidget(self.stats_widget)
        
        layout.addWidget(self.stack, 1)  # Stretch to fill
        
        # Connect network pane signal to forward it
        self.network_pane.nodeSelected.connect(self.nodeSelected)
    
    def _open_settings(self):
        """Open the API Keys & Provider Routing settings dialog."""
        try:
            from settings_dialog import SettingsDialog
            dlg = SettingsDialog(self)
            dlg.exec()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Settings Error", str(e))

    def switch_tab(self, index):
        """Switch between tabs"""
        self.stack.setCurrentIndex(index)
        
        # Update button states
        self.setup_button.setChecked(index == 0)
        self.graph_button.setChecked(index == 1)
        self.image_button.setChecked(index == 2)
        self.video_button.setChecked(index == 3)
        self.stats_button.setChecked(index == 4)
    
    def update_image_preview(self, image_path, ai_name="", prompt=""):
        """Update the image preview pane with a new image"""
        if hasattr(self, 'image_preview_pane'):
            self.image_preview_pane.set_image(image_path, ai_name, prompt)
    
    def update_video_preview(self, video_path, ai_name="", prompt=""):
        """Update the video preview pane with a new video"""
        if hasattr(self, 'video_preview_pane'):
            self.video_preview_pane.set_video(video_path, ai_name, prompt)
    
    def add_node(self, node_id, label, node_type):
        """Forward to network pane"""
        self.network_pane.add_node(node_id, label, node_type)
    
    def add_edge(self, source_id, target_id):
        """Forward to network pane"""
        self.network_pane.add_edge(source_id, target_id)
    
    def update_graph(self):
        """Forward to network pane"""
        self.network_pane.update_graph()

class RetroSlider(QWidget):
    """Terminal-style segmented slider with ASCII gauge display.

    Visual: |####......| 6
    Green fill bars on black, clickable segments, scroll wheel support.
    """
    valueChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)

    def __init__(self, values=None, parent=None):
        super().__init__(parent)
        self.values = values or [1, 2, 4, 6, 12, 100]
        self._index = 0
        self.setFixedHeight(28)
        self.setMinimumWidth(140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    @property
    def current_value(self):
        return self.values[self._index]

    def currentText(self):
        """Compatibility with NoScrollComboBox interface."""
        return str(self.current_value)

    def set_value(self, val):
        if val in self.values:
            self._index = self.values.index(val)
            self.update()
            self.valueChanged.emit(self.current_value)
            self.currentTextChanged.emit(str(self.current_value))

    def setCurrentText(self, text):
        """Compatibility with QComboBox interface."""
        try:
            val = int(text)
            self.set_value(val)
        except (ValueError, TypeError):
            pass

    def mousePressEvent(self, event):
        segment_width = self.width() / len(self.values)
        clicked_index = int(event.position().x() / segment_width)
        clicked_index = max(0, min(len(self.values) - 1, clicked_index))
        if clicked_index != self._index:
            self._index = clicked_index
            self.update()
            self.valueChanged.emit(self.current_value)
            self.currentTextChanged.emit(str(self.current_value))

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            new_idx = min(len(self.values) - 1, self._index + 1)
        else:
            new_idx = max(0, self._index - 1)
        if new_idx != self._index:
            self._index = new_idx
            self.update()
            self.valueChanged.emit(self.current_value)
            self.currentTextChanged.emit(str(self.current_value))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(COLORS['bg_dark']))

        # Border
        painter.setPen(QPen(QColor(COLORS['border_glow']), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Fill segments
        pad = 3
        usable_w = w - (pad * 2)
        segment_w = usable_w / len(self.values)
        gap = 1

        for i in range(len(self.values)):
            x = pad + int(i * segment_w)
            seg_w = int(segment_w) - gap
            if seg_w < 2:
                seg_w = 2

            if i <= self._index:
                if i == self._index:
                    color = QColor(COLORS['text_bright'])
                else:
                    color = QColor(COLORS['accent_cyan'])
            else:
                color = QColor(COLORS['border'])

            painter.fillRect(x, pad + 1, seg_w, h - (pad * 2) - 2, color)

        # Value text overlay
        font = painter.font()
        font.setFamily("Iosevka Term")
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)

        text = str(self.current_value)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        text_x = (w - text_w) // 2
        text_y = (h + fm.ascent() - fm.descent()) // 2

        # Text shadow for readability
        painter.setPen(QColor(COLORS['bg_dark']))
        painter.drawText(text_x + 1, text_y + 1, text)
        painter.drawText(text_x - 1, text_y, text)

        # Actual text
        if self._index >= len(self.values) // 2:
            painter.setPen(QColor(COLORS['bg_dark']))
        else:
            painter.setPen(QColor(COLORS['text_bright']))
        painter.drawText(text_x, text_y, text)

        painter.end()


class ControlPanel(QWidget):
    """Control panel with mode, model selections, etc."""
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize with models and prompt pairs
        self.initialize_selectors()
    
    def _ascii_section_header(self, text, width=26):
        """Create an ASCII box-drawing section header label."""
        inner = f" {text} "
        remaining = width - len(inner) - 2
        left_dashes = remaining // 2
        right_dashes = remaining - left_dashes
        dash = "\u2500"
        line = "\u250c" + dash * left_dashes + inner + dash * right_dashes + "\u2510"
        label = QLabel(line)
        label.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-family: {FONTS['family_mono']};
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 0px;
            background-color: transparent;
            padding: 2px 0px;
        """)
        return label

    def _ascii_divider(self, width=26):
        """Create an ASCII box-drawing divider."""
        dash = "\u2500"
        line = "\u251c" + dash * (width - 2) + "\u2524"
        label = QLabel(line)
        label.setStyleSheet(f"""
            color: {COLORS['border_glow']};
            font-family: {FONTS['family_mono']};
            font-size: 10px;
            background-color: transparent;
            padding: 4px 0px 2px 0px;
        """)
        return label

    def setup_ui(self):
        """Set up the user interface for the control panel - vertical sidebar layout"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Add a title with consistent tab header styling
        title = QLabel("CONTROL PANEL")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 13px;
            font-weight: bold;
            padding: 12px;
            background-color: {COLORS['bg_medium']};
            border-bottom: 1px solid {COLORS['border_glow']};
            letter-spacing: 3px;
            text-transform: uppercase;
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
            {get_scrollbar_style()}
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
        
        mode_label = self._ascii_section_header("MODE")
        mode_layout.addWidget(mode_label)
        
        self.mode_selector = NoScrollComboBox()
        self.mode_selector.addItems(["AI-AI", "Human-AI"])
        self.mode_selector.setStyleSheet(get_combobox_style())
        mode_layout.addWidget(self.mode_selector)

        # Username input
        username_container = QWidget()
        username_layout = QVBoxLayout(username_container)
        username_layout.setContentsMargins(0, 0, 0, 0)
        username_layout.setSpacing(5)

        username_label = self._ascii_section_header("YOUR NAME")
        username_layout.addWidget(username_label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Human User")
        self.username_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
                padding: 6px 8px;
                font-family: {FONTS['family_mono']};
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {COLORS['accent_cyan']};
            }}
        """)
        self.username_input.setToolTip("Your display name — shown in chat and sent to AI models")
        username_layout.addWidget(self.username_input)

        # Iterations with slider
        iterations_container = QWidget()
        iterations_layout = QVBoxLayout(iterations_container)
        iterations_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.setSpacing(5)
        
        iterations_label = self._ascii_section_header("ITERATIONS")
        iterations_layout.addWidget(iterations_label)
        
        self.iterations_selector = RetroSlider(values=[1, 2, 4, 6, 12, 100])
        self.iterations_selector.set_value(1)
        iterations_layout.addWidget(self.iterations_selector)

        # Step Mode toggle - single AI turn per propagate click
        step_mode_container = QWidget()
        step_mode_layout = QVBoxLayout(step_mode_container)
        step_mode_layout.setContentsMargins(0, 0, 0, 0)
        step_mode_layout.setSpacing(5)

        step_mode_label = self._ascii_section_header("STEP MODE")
        step_mode_layout.addWidget(step_mode_label)

        self.step_mode_checkbox = QCheckBox("Single AI turn per click")
        self.step_mode_checkbox.setStyleSheet(get_checkbox_style())
        self.step_mode_checkbox.setToolTip("When enabled, each Propagate click advances one AI turn instead of a full round. Lets you interject between AI turns.")
        self.step_mode_checkbox.stateChanged.connect(self._on_step_mode_changed)
        step_mode_layout.addWidget(self.step_mode_checkbox)

        # Next Turn selector - pick which AI goes next (only visible in step mode)
        self.next_turn_container = QWidget()
        next_turn_layout = QHBoxLayout(self.next_turn_container)
        next_turn_layout.setContentsMargins(0, 0, 0, 0)
        next_turn_layout.setSpacing(5)

        next_turn_label = QLabel("Next:")
        next_turn_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        next_turn_layout.addWidget(next_turn_label)

        self.next_turn_selector = NoScrollComboBox()
        self.next_turn_selector.addItems(["Auto", "AI-1", "AI-2", "AI-3", "AI-4", "AI-5"])
        self.next_turn_selector.setStyleSheet(get_combobox_style())
        self.next_turn_selector.setToolTip("Auto follows normal sequence. Pick an AI to override which one speaks next.")
        next_turn_layout.addWidget(self.next_turn_selector, 1)

        step_mode_layout.addWidget(self.next_turn_container)
        self.next_turn_container.setVisible(False)  # Hidden until step mode enabled

        # Number of AIs selection
        num_ais_container = QWidget()
        num_ais_layout = QVBoxLayout(num_ais_container)
        num_ais_layout.setContentsMargins(0, 0, 0, 0)
        num_ais_layout.setSpacing(5)
        
        num_ais_label = self._ascii_section_header("NUMBER OF AIs")
        num_ais_layout.addWidget(num_ais_label)
        
        self.num_ais_selector = RetroSlider(values=[1, 2, 3, 4, 5])
        self.num_ais_selector.set_value(3)
        num_ais_layout.addWidget(self.num_ais_selector)
        
        # AI Invite Tier Setting - Button Group
        invite_tier_container = QWidget()
        invite_tier_layout = QVBoxLayout(invite_tier_container)
        invite_tier_layout.setContentsMargins(0, 0, 0, 0)
        invite_tier_layout.setSpacing(5)
        
        invite_tier_label = self._ascii_section_header("AI INVITE TIER")
        invite_tier_layout.addWidget(invite_tier_label)
        
        # Info text
        invite_tier_info = QLabel("Controls which models AIs can add to the chat")
        invite_tier_info.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 9px;")
        invite_tier_layout.addWidget(invite_tier_info)
        
        # Button group container
        btn_group_container = QWidget()
        btn_group_layout = QHBoxLayout(btn_group_container)
        btn_group_layout.setContentsMargins(0, 0, 0, 0)
        btn_group_layout.setSpacing(0)
        
        # Create toggle buttons
        self.invite_sota_btn = QPushButton("SOTA")
        self.invite_free_btn = QPushButton("Free")
        self.invite_paid_btn = QPushButton("Paid")
        self.invite_both_btn = QPushButton("All")

        # Store the buttons for easy access
        self._invite_tier_buttons = [self.invite_sota_btn, self.invite_free_btn, self.invite_paid_btn, self.invite_both_btn]
        
        # Style for toggle buttons
        toggle_btn_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                border: 1px solid {COLORS['border']};
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_bright']};
                border: 1px solid {COLORS['accent_cyan']};
            }}
        """

        for btn in self._invite_tier_buttons:
            btn.setCheckable(True)
            btn.setStyleSheet(toggle_btn_style)
            btn.clicked.connect(self._on_invite_tier_clicked)
            btn_group_layout.addWidget(btn)
        
        # Round corners on first and last buttons
        self.invite_sota_btn.setStyleSheet(toggle_btn_style + """
            QPushButton { border-radius: 3px 0px 0px 3px; }
        """)
        self.invite_both_btn.setStyleSheet(toggle_btn_style + """
            QPushButton { border-radius: 0px 3px 3px 0px; }
        """)

        # Set default selection (SOTA)
        self.invite_sota_btn.setChecked(True)

        # Tooltips
        self.invite_sota_btn.setToolTip("AIs can only invite SOTA models")
        self.invite_free_btn.setToolTip("AIs can only invite free models")
        self.invite_paid_btn.setToolTip("AIs can only invite paid models")
        self.invite_both_btn.setToolTip("AIs can invite any model")
        
        invite_tier_layout.addWidget(btn_group_container)
        
        # Allow duplicate models checkbox
        self.allow_duplicate_models_checkbox = QCheckBox("Allow duplicate models")
        self.allow_duplicate_models_checkbox.setChecked(False)  # Default to restricted
        self.allow_duplicate_models_checkbox.setStyleSheet(get_checkbox_style())
        self.allow_duplicate_models_checkbox.setToolTip("Allow AIs to add models that are already in the conversation")
        invite_tier_layout.addWidget(self.allow_duplicate_models_checkbox)
        
        # AI-1 Model selection
        self.ai1_container = QWidget()
        ai1_layout = QVBoxLayout(self.ai1_container)
        ai1_layout.setContentsMargins(0, 0, 0, 0)
        ai1_layout.setSpacing(5)
        
        ai1_label = QLabel("AI-1")
        ai1_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai1_layout.addWidget(ai1_label)
        
        self.ai1_model_selector = GroupedModelComboBox(colors=COLORS, parent=self)
        self.ai1_model_selector.setStyleSheet(get_combobox_style())
        ai1_layout.addWidget(self.ai1_model_selector)
        
        # AI-2 Model selection
        self.ai2_container = QWidget()
        ai2_layout = QVBoxLayout(self.ai2_container)
        ai2_layout.setContentsMargins(0, 0, 0, 0)
        ai2_layout.setSpacing(5)
        
        ai2_label = QLabel("AI-2")
        ai2_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai2_layout.addWidget(ai2_label)
        
        self.ai2_model_selector = GroupedModelComboBox(colors=COLORS, parent=self)
        self.ai2_model_selector.setStyleSheet(get_combobox_style())
        ai2_layout.addWidget(self.ai2_model_selector)
        
        # AI-3 Model selection
        self.ai3_container = QWidget()
        ai3_layout = QVBoxLayout(self.ai3_container)
        ai3_layout.setContentsMargins(0, 0, 0, 0)
        ai3_layout.setSpacing(5)
        
        ai3_label = QLabel("AI-3")
        ai3_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai3_layout.addWidget(ai3_label)
        
        self.ai3_model_selector = GroupedModelComboBox(colors=COLORS, parent=self)
        self.ai3_model_selector.setStyleSheet(get_combobox_style())
        ai3_layout.addWidget(self.ai3_model_selector)
        
        # AI-4 Model selection
        self.ai4_container = QWidget()
        ai4_layout = QVBoxLayout(self.ai4_container)
        ai4_layout.setContentsMargins(0, 0, 0, 0)
        ai4_layout.setSpacing(5)
        
        ai4_label = QLabel("AI-4")
        ai4_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai4_layout.addWidget(ai4_label)
        
        self.ai4_model_selector = GroupedModelComboBox(colors=COLORS, parent=self)
        self.ai4_model_selector.setStyleSheet(get_combobox_style())
        ai4_layout.addWidget(self.ai4_model_selector)
        
        # AI-5 Model selection
        self.ai5_container = QWidget()
        ai5_layout = QVBoxLayout(self.ai5_container)
        ai5_layout.setContentsMargins(0, 0, 0, 0)
        ai5_layout.setSpacing(5)
        
        ai5_label = QLabel("AI-5")
        ai5_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai5_layout.addWidget(ai5_label)
        
        self.ai5_model_selector = GroupedModelComboBox(colors=COLORS, parent=self)
        self.ai5_model_selector.setStyleSheet(get_combobox_style())
        ai5_layout.addWidget(self.ai5_model_selector)
        
        # Prompt pair selection
        prompt_container = QWidget()
        prompt_layout = QVBoxLayout(prompt_container)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(5)
        
        prompt_label = QLabel("Conversation Scenario")
        prompt_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        prompt_layout.addWidget(prompt_label)
        
        self.prompt_pair_selector = NoScrollComboBox()
        self.prompt_pair_selector.setStyleSheet(get_combobox_style())
        prompt_layout.addWidget(self.prompt_pair_selector)
        
        # Add all controls directly to controls_layout (now vertical)
        controls_layout.addWidget(mode_container)
        controls_layout.addWidget(username_container)
        controls_layout.addWidget(iterations_container)
        controls_layout.addWidget(step_mode_container)
        controls_layout.addWidget(num_ais_container)
        controls_layout.addWidget(invite_tier_container)
        
        # Divider
        divider1 = self._ascii_divider()
        controls_layout.addWidget(divider1)
        
        models_label = self._ascii_section_header("AI MODELS")
        controls_layout.addWidget(models_label)
        
        controls_layout.addWidget(self.ai1_container)
        controls_layout.addWidget(self.ai2_container)
        controls_layout.addWidget(self.ai3_container)
        controls_layout.addWidget(self.ai4_container)
        controls_layout.addWidget(self.ai5_container)
        
        # Divider
        divider2 = self._ascii_divider()
        controls_layout.addWidget(divider2)
        
        scenario_label = self._ascii_section_header("SCENARIO")
        controls_layout.addWidget(scenario_label)
        
        controls_layout.addWidget(prompt_container)
        
        # Divider
        divider3 = self._ascii_divider()
        controls_layout.addWidget(divider3)
        
        # OPTIONS section (in scrollable area)
        options_label = self._ascii_section_header("OPTIONS")
        controls_layout.addWidget(options_label)
        
        # Auto-generate images checkbox
        self.auto_image_checkbox = QCheckBox("Create images from responses")
        self.auto_image_checkbox.setStyleSheet(get_checkbox_style())
        self.auto_image_checkbox.setToolTip("Automatically generate images from AI responses using Google Gemini 3 Pro Image Preview via OpenRouter")
        controls_layout.addWidget(self.auto_image_checkbox)
        
        # Add spacer to push content to top
        controls_layout.addStretch()
        
        # Set the scroll area widget and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)  # Stretch to fill
        
        # ═══ STICKY FOOTER: ACTIONS ONLY ═══
        # This stays visible at bottom regardless of scroll position
        action_container = QWidget()
        action_container.setObjectName("actionFooter")
        action_container.setStyleSheet(f"""
            QWidget#actionFooter {{
                background-color: transparent;
                border-top: 1px solid {COLORS['border_glow']};
            }}
        """)
        action_layout = QVBoxLayout(action_container)
        action_layout.setContentsMargins(10, 8, 10, 8)
        action_layout.setSpacing(8)
        
        # Actions - buttons in vertical layout
        actions_label = self._ascii_section_header("ACTIONS")
        action_layout.addWidget(actions_label)
        
        # Export button
        self.export_button = QPushButton("📡 EXPORT")
        self.export_button.setStyleSheet(get_button_style(COLORS['accent_purple']))
        action_layout.addWidget(self.export_button)
        
        # View HTML button - opens the styled conversation
        self.view_html_button = QPushButton("🌐 VIEW HTML")
        self.view_html_button.setStyleSheet(get_button_style(COLORS['accent_green']))
        self.view_html_button.clicked.connect(self._open_current_html)
        action_layout.addWidget(self.view_html_button)

        # BackroomsBench evaluation button
        self.backroomsbench_button = QPushButton("🌀 BACKROOMSBENCH (beta)")
        self.backroomsbench_button.setStyleSheet(get_button_style(COLORS['accent_purple']))
        self.backroomsbench_button.setToolTip("Run multi-judge AI evaluation (depth/philosophy)")
        action_layout.addWidget(self.backroomsbench_button)

        main_layout.addWidget(action_container)  # Sticky at bottom
    
    # NOTE: get_combobox_style() has been moved to styles.py
    # Use the imported get_combobox_style() function instead of get_combobox_style()
    
    # NOTE: get_cyberpunk_button_style() has been moved to styles.py as get_button_style()
    # Use the imported get_button_style() function instead of self.get_cyberpunk_button_style()
    
    def create_glow_button(self, text, accent_color):
        """Create a button with glow effect"""
        button = GlowButton(text, accent_color)
        button.setStyleSheet(get_button_style(accent_color))
        return button
    
    def _on_invite_tier_clicked(self):
        """Handle invite tier button clicks - ensure only one is selected"""
        clicked_btn = self.sender()
        for btn in self._invite_tier_buttons:
            if btn != clicked_btn:
                btn.setChecked(False)
        # Ensure at least one is always selected
        if not clicked_btn.isChecked():
            clicked_btn.setChecked(True)
    
    def get_ai_invite_tier(self):
        """Get the current AI invite tier setting"""
        if self.invite_sota_btn.isChecked():
            return "SOTA"
        elif self.invite_free_btn.isChecked():
            return "Free"
        elif self.invite_paid_btn.isChecked():
            return "Paid"
        else:
            return "Both"
    
    def _on_step_mode_changed(self, state):
        """Show/hide the next turn selector when step mode is toggled."""
        enabled = state == 2  # Qt.CheckState.Checked
        self.next_turn_container.setVisible(enabled)
        if not enabled:
            self.next_turn_selector.setCurrentText("Auto")

    def is_step_mode(self):
        """Return whether step mode is currently enabled."""
        return self.step_mode_checkbox.isChecked()

    def get_username(self):
        """Return the user's display name, defaulting to 'Human User'."""
        name = self.username_input.text().strip()
        return name if name else "Human User"

    def initialize_selectors(self):
        """Initialize the selector dropdowns with values from config"""
        # AI model selectors are GroupedModelComboBox instances that self-populate
        # from config.AI_MODELS - no need to manually add items here
        
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
        
        # AI-1 is always visible
        # AI-2 visible if num_ais >= 2
        # AI-3 visible if num_ais >= 3
        # AI-4 visible if num_ais >= 4
        # AI-5 visible if num_ais >= 5
        
        self.ai1_container.setVisible(num_ais >= 1)
        self.ai2_container.setVisible(num_ais >= 2)
        self.ai3_container.setVisible(num_ais >= 3)
        self.ai4_container.setVisible(num_ais >= 4)
        self.ai5_container.setVisible(num_ais >= 5)
    
    def _open_current_html(self):
        """Open the current session's HTML file in browser"""
        try:
            # Get main window to access current_html_file
            main_window = self.window()
            current_file = getattr(main_window, 'current_html_file', None)
            
            if current_file and os.path.exists(current_file):
                open_html_file(current_file)
            else:
                # Fallback: try to find the most recent conversation file in outputs
                from config import OUTPUTS_DIR
                import glob
                
                pattern = os.path.join(OUTPUTS_DIR, "conversation_*.html")
                files = glob.glob(pattern)
                
                if files:
                    # Get the most recent file
                    latest_file = max(files, key=os.path.getmtime)
                    open_html_file(latest_file)
                else:
                    # No files found
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self,
                        "No Conversation",
                        "No conversation HTML file found.\nStart a conversation first."
                    )
        except Exception as e:
            import traceback
            print(f"[ERROR] Error opening HTML file: {e}")
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error opening HTML file:\n{e}")

class ConversationContextMenu(QMenu):
    """Context menu for the conversation display"""
    rabbitholeSelected = pyqtSignal()
    forkSelected = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create actions
        self.rabbithole_action = QAction("🕳️ Rabbithole", self)
        self.fork_action = QAction("🔱 Fork", self)
        
        # Add actions to menu
        self.addAction(self.rabbithole_action)
        self.addAction(self.fork_action)

        # Connect actions to signals
        self.rabbithole_action.triggered.connect(self.on_rabbithole_selected)
        self.fork_action.triggered.connect(self.on_fork_selected)
        
        # Apply styling
        self.setStyleSheet("""
            QMenu {
                background-color: #2D2D30;
                color: #D4D4D4;
                border: 1px solid #3E3E42;
            }
            QMenu::item {
                padding: 5px 20px 5px 20px;
            }
            QMenu::item:selected {
                background-color: #3E3E42;
            }
        """)
    
    def on_rabbithole_selected(self):
        """Signal that rabbithole action was selected."""
        selected = self._get_selected_text()
        if selected:
            self.rabbitholeSelected.emit()

    def on_fork_selected(self):
        """Signal that fork action was selected."""
        selected = self._get_selected_text()
        if selected:
            self.forkSelected.emit()

    @staticmethod
    def _get_selected_text() -> str:
        """Get selected text from the currently focused message widget."""
        from PyQt6.QtWidgets import QApplication
        widget = QApplication.focusWidget()
        if widget and hasattr(widget, 'textCursor'):
            cursor = widget.textCursor()
            if cursor.hasSelection():
                return cursor.selectedText()
        return ""

class ConversationPane(QWidget):
    """Left pane containing the conversation and input area"""
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setup_ui()
        
        # Connect signals and slots
        self.connect_signals()
        
        # =====================================================================
        # SCROLL STATE MANAGEMENT (ChatScrollArea approach)
        # =====================================================================
        # Scroll state is now managed by ChatScrollArea + MessageWidget:
        # - Each message is a separate widget (no setHtml destroying everything)
        # - ChatScrollArea tracks user scroll intent via valueChanged
        # - Adding messages just appends widgets, preserving scroll position
        # =====================================================================
        
        # Debug flag (ChatScrollArea has its own _debug flag)
        if self._SCROLL_DEBUG:
            print("[SCROLL] ConversationPane initialized with ChatScrollArea")
        
        # Initialize state
        self.conversation = []
        self.input_callback = None
        self.rabbithole_callback = None
        self.fork_callback = None
        self.loading = False
        self.loading_dots = 0
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_timer.setInterval(300)  # Update every 300ms for smoother animation
        
        # Context menu
        self.context_menu = ConversationContextMenu(self)
        
        # Initialize with empty conversation
        self.update_conversation([])
        
        # Images list - to prevent garbage collection
        self.images = []
        self.image_paths = []
        
        # Uploaded image for current message
        self.uploaded_image_path = None
        self.uploaded_image_base64 = None

        # Create text formats with different colors
        self.text_formats = {
            "user": QTextCharFormat(),
            "ai": QTextCharFormat(),
            "system": QTextCharFormat(),
            "ai_label": QTextCharFormat(),
            "normal": QTextCharFormat(),
            "error": QTextCharFormat()
        }

        # Configure text formats using global color palette
        self.text_formats["user"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["ai"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["system"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["ai_label"].setForeground(QColor(COLORS['accent_blue']))
        self.text_formats["normal"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["error"].setForeground(QColor(COLORS['text_error']))
        
        # Make AI labels bold
        self.text_formats["ai_label"].setFontWeight(QFont.Weight.Bold)
    
    def setup_ui(self):
        """Set up the user interface for the conversation pane"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)  # Reduced spacing
        
        # Title and info area
        title_layout = QHBoxLayout()
        self.title_label = QLabel("╔═ LIMINAL BACKROOMS ═╗")
        self.title_label.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 14px;
            font-weight: bold;
            padding: 4px;
            letter-spacing: 2px;
        """)
        
        self.info_label = QLabel("[ AI-TO-AI CONVERSATION ]")
        self.info_label.setStyleSheet(f"""
            color: {COLORS['text_glow']};
            font-size: 10px;
            padding: 2px;
            letter-spacing: 1px;
        """)
        
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.info_label)
        
        layout.addLayout(title_layout)
        
        # Conversation display (widget-based chat scroll area)
        # Each message is a separate widget - no setHtml() means no scroll jumping!
        self.conversation_display = ChatScrollArea()
        self.conversation_display.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conversation_display.customContextMenuRequested.connect(self.show_context_menu)
        
        # Set font for the container (will cascade to message widgets)
        font = QFont("Iosevka Term", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.conversation_display.container.setFont(font)
        
        # Input area with label
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)  # Better spacing between input and buttons
        
        input_label = QLabel("Your message:")
        input_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        input_layout.addWidget(input_label)
        
        # Input field with modern styling
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Seed the conversation or just click propagate...")
        self.input_field.setMaximumHeight(60)  # Reduced height
        self.input_field.setFont(font)
        self.input_field.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 8px;
                selection-background-color: {COLORS['accent_cyan']};
                selection-color: {COLORS['bg_dark']};
            }}
        """)
        input_layout.addWidget(self.input_field)
        
        # Button container for better layout
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)  # Reduced spacing
        
        # Upload image button
        self.upload_image_button = QPushButton("📎 IMAGE")
        self.upload_image_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
                min-width: 70px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['accent_cyan']};
                color: {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['border_glow']};
            }}
        """)
        self.upload_image_button.setToolTip("Upload an image to include in your message")
        
        # Clear button - no glow effect
        self.clear_button = QPushButton("CLEAR")
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
                min-width: 70px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['accent_pink']};
                color: {COLORS['accent_pink']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['border_glow']};
            }}
        """)
        
        # Submit button with cyberpunk styling and glow effect
        self.submit_button = GlowButton("⚡ PROPAGATE", COLORS['accent_cyan'])
        self.submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['accent_cyan']};
                border-radius: 0px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border: 1px solid {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_cyan_active']};
                color: {COLORS['text_bright']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_dim']};
                border: 1px solid {COLORS['border']};
            }}
        """)
        
        # Reset button - clears conversation context (no glow)
        self.reset_button = QPushButton("↺ RESET")
        self.reset_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['accent_pink']};
                border: 1px solid {COLORS['accent_pink']};
                border-radius: 0px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_pink']};
                color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['accent_pink']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_pink']};
                color: {COLORS['text_bright']};
            }}
            QPushButton:disabled {{
                background-color: transparent;
                color: {COLORS['text_dim']};
                border: 1px solid {COLORS['border']};
            }}
        """)
        self.reset_button.setToolTip("Clear conversation and start fresh")
        
        # Add buttons to layout
        button_layout.addWidget(self.upload_image_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.submit_button)
        
        # Add input container to main layout
        input_layout.addWidget(button_container)
        
        # Search overlay
        self.search_overlay = SearchOverlay(self)

        # Add widgets to layout with adjusted stretch factors
        layout.addWidget(self.search_overlay)
        layout.addWidget(self.conversation_display, 1)  # Main conversation area gets most space
        layout.addWidget(input_container, 0)  # Input area gets minimal space
    
    def connect_signals(self):
        """Connect signals and slots"""
        # Submit button
        self.submit_button.clicked.connect(self.handle_propagate_click)
        
        # Reset button
        self.reset_button.clicked.connect(self.handle_reset_click)
        
        # Upload image button
        self.upload_image_button.clicked.connect(self.handle_upload_image)
        
        # Clear button
        self.clear_button.clicked.connect(self.clear_input)
        
        # Enter key in input field
        self.input_field.installEventFilter(self)
    
    def clear_input(self):
        """Clear the input field"""
        self.input_field.clear()
        self.uploaded_image_path = None
        self.uploaded_image_base64 = None
        self.upload_image_button.setText("📎 IMAGE")
        self.input_field.setFocus()
    
    def handle_upload_image(self):
        """Handle image upload button click"""
        # Open file dialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.webp);;All Files (*)"
        )
        
        if file_path:
            try:
                # Read and encode the image to base64
                with open(file_path, 'rb') as image_file:
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Determine media type
                file_extension = os.path.splitext(file_path)[1].lower()
                media_type_map = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                media_type = media_type_map.get(file_extension, 'image/jpeg')
                
                # Store the image data
                self.uploaded_image_path = file_path
                self.uploaded_image_base64 = {
                    'data': image_base64,
                    'media_type': media_type
                }
                
                # Update button text to show an image is attached
                file_name = os.path.basename(file_path)
                self.upload_image_button.setText(f"📎 {file_name[:15]}...")
                
                # Update placeholder text
                self.input_field.setPlaceholderText("Add a message about your image (optional)...")
                
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Upload Error",
                    f"Failed to load image: {str(e)}"
                )
    
    def eventFilter(self, obj, event):
        """Filter events to handle Enter key in input field"""
        if obj is self.input_field and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.handle_propagate_click()
                return True
        return super().eventFilter(obj, event)
    
    def handle_propagate_click(self):
        """Handle click on the propagate button"""
        # Get the input text (might be empty)
        input_text = self.input_field.toPlainText().strip()
        
        # Prepare message data (text + optional image)
        message_data = {
            'text': input_text,
            'image': None
        }
        
        # Include image if one was uploaded
        if self.uploaded_image_base64:
            message_data['image'] = {
                'path': self.uploaded_image_path,
                'base64': self.uploaded_image_base64['data'],
                'media_type': self.uploaded_image_base64['media_type']
            }
        
        # Clear the input box and image
        self.input_field.clear()
        self.uploaded_image_path = None
        self.uploaded_image_base64 = None
        self.upload_image_button.setText("📎 IMAGE")
        self.input_field.setPlaceholderText("Seed the conversation or just click propagate...")
        
        # Always call the input callback, even with empty input
        if hasattr(self, 'input_callback') and self.input_callback:
            self.input_callback(message_data)
        
        # Start loading animation
        self.start_loading()
    
    def handle_reset_click(self):
        """Handle click on the reset button - clears conversation context"""
        # Get the main window reference
        main_window = self.window()
        
        # Clear main conversation
        if hasattr(main_window, 'main_conversation'):
            main_window.main_conversation = []
        
        # Clear branch conversations
        if hasattr(main_window, 'branch_conversations'):
            main_window.branch_conversations = {}
        
        # Clear active branch
        if hasattr(main_window, 'active_branch'):
            main_window.active_branch = None
        
        # Clear local conversation reference
        self.conversation = []
        
        # Clear the input field
        self.input_field.clear()
        self.uploaded_image_path = None
        self.uploaded_image_base64 = None
        self.upload_image_button.setText("📎 IMAGE")
        
        # Re-render empty conversation
        self.render_conversation()
        
        # Update status bar
        if hasattr(main_window, 'statusBar'):
            main_window.statusBar().showMessage("Conversation reset - ready for new session")
        
        print("[UI] Conversation reset by user")
    
    def toggle_search(self):
        """Toggle the search overlay."""
        self.search_overlay.toggle()

    def set_input_callback(self, callback):
        """Set callback function for input submission"""
        self.input_callback = callback

    def set_rabbithole_callback(self, callback):
        """Set callback function for rabbithole creation"""
        self.rabbithole_callback = callback

    def set_fork_callback(self, callback):
        """Set callback function for fork creation"""
        self.fork_callback = callback
    
    # =========================================================================
    # SCROLL MANAGEMENT
    # =========================================================================
    # 
    # This section handles scroll behavior for the chat display.
    # See ChatScrollArea docstring for full architecture documentation.
    #
    # DEBUG LOGGING:
    # - Set _SCROLL_DEBUG = True to enable [SCROLL] prefixed log messages
    # - ChatScrollArea has its own _debug flag for [CHAT-SCROLL] messages
    # - Both should be enabled together for full debugging
    #
    # QUICK REFERENCE:
    # - [CHAT-SCROLL] messages come from ChatScrollArea (low-level scroll ops)
    # - [SCROLL] messages come from ConversationPane (high-level render ops)
    # =========================================================================

    _SCROLL_DEBUG = DEVELOPER_TOOLS  # Enable [SCROLL] logging (controlled by config.DEVELOPER_TOOLS)
    
    def reset_scroll_state(self):
        """Reset to auto-scroll mode (delegates to ChatScrollArea)."""
        self.conversation_display.reset_scroll_state()
    
    def update_conversation(self, conversation):
        """Update conversation display"""
        self.conversation = conversation
        self.render_conversation()
    
    def update_streaming_content(self, ai_name: str, new_content: str):
        """
        Fast-path for streaming updates - bypasses full render pipeline.
        
        Only updates the content of a matching streaming widget without rebuilding
        the displayable list or checking all widgets.
        
        Args:
            ai_name: The AI whose message is being streamed
            new_content: The new text content (full content, not delta)
        
        Returns:
            True if update was applied, False if full render is needed
        """
        try:
            # Find the streaming widget that matches this AI
            for widget in reversed(self.conversation_display.message_widgets):
                if hasattr(widget, 'message_data'):
                    widget_ai = widget.message_data.get('ai_name', '')
                    widget_role = widget.message_data.get('role', '')
                    is_streaming = widget.message_data.get('_streaming', False)
                    # Match by AI name AND streaming flag to avoid updating wrong widget
                    if widget_ai == ai_name and widget_role == 'assistant' and is_streaming:
                        widget.update_content(new_content)
                        # Schedule scroll if following
                        if self.conversation_display._should_follow:
                            self.conversation_display._schedule_scroll()
                        return True
            return False
        except Exception as e:
            print(f"[STREAM] Fast-path failed: {e}")
            return False
    
    def update_streaming_widget(self, ai_name: str, new_content: str):
        """
        Update a specific AI's streaming widget directly.
        
        This is the primary method for streaming updates - it finds the exact
        widget for this AI's streaming message and updates it in place.
        Does NOT trigger re-render which could cause cross-contamination.
        
        Args:
            ai_name: The AI whose message is being streamed (e.g., "AI-1")
            new_content: The complete current content (not a delta)
        """
        try:
            # Find the streaming widget for this specific AI
            for widget in reversed(self.conversation_display.message_widgets):
                if hasattr(widget, 'message_data'):
                    msg_data = widget.message_data
                    if (msg_data.get('ai_name') == ai_name and 
                        msg_data.get('role') == 'assistant' and
                        msg_data.get('_streaming', False)):
                        # Found it - update content directly
                        widget.update_content(new_content)
                        # Auto-scroll if user is following
                        if self.conversation_display._should_follow:
                            self.conversation_display._schedule_scroll()
                        return
            # Widget not found - might need a full render
            print(f"[STREAM] Widget not found for {ai_name}, triggering render")
            self.render_conversation()
        except Exception as e:
            print(f"[STREAM] update_streaming_widget failed: {e}")
            import traceback
            traceback.print_exc()
    
    def render_conversation(self, immediate=False):
        """Render conversation in the display (debounced by default)
        
        Args:
            immediate: If True, skip debounce and render now (for critical updates)
        """
        try:
            # Initialize render timer once
            if not hasattr(self, '_render_timer') or self._render_timer is None:
                self._render_timer = QTimer()
                self._render_timer.setSingleShot(True)
                self._render_timer.timeout.connect(self._do_render)
            
            if immediate:
                self._render_timer.stop()
                self._do_render()
            else:
                # Debounce - batches rapid calls (reduced from 50ms to 16ms for responsiveness)
                self._render_timer.stop()
                self._render_timer.start(16)
        except Exception as e:
            print(f"[SCROLL ERROR] render_conversation: {e}")
            import traceback
            traceback.print_exc()
    
    def _do_render(self):
        """Actually perform the render using ChatScrollArea + MessageWidgets.
        
        ARCHITECTURE:
        ═══════════════════════════════════════════════════════════════════════════
        
        Two modes based on whether streaming is active:
        
        1. STREAMING MODE (when any message has _streaming=True):
           - Uses incremental updates for performance
           - Only adds new widgets, doesn't touch existing ones
           - Widget content updates happen via update_streaming_widget() directly
        
        2. REBUILD MODE (when no streaming is active):
           - Always rebuilds all widgets
           - Ensures correctness when content changes, messages replaced, etc.
           - Uses setUpdatesEnabled(False) to prevent flicker
           - Preserves scroll state
        
        This solves edge cases where count is same but content differs:
        - Streaming complete (raw content → cleaned content)
        - Notification removed + image added (same count, different widgets)
        - Any message content modification
        ═══════════════════════════════════════════════════════════════════════════
        """
        try:
            existing_count = len(self.conversation_display.message_widgets)
            
            # Build list of displayable messages
            displayable = []
            has_streaming = False
            
            for message in self.conversation:
                content = message.get('content', '')
                text_content = self._extract_text_content(content)
                msg_type = message.get('_type', '')
                
                # Track if any message is actively streaming
                if message.get('_streaming'):
                    has_streaming = True
                
                # Always show notifications
                if msg_type == 'agent_notification':
                    displayable.append(message)
                    continue
                
                # Always show generated images and videos
                if msg_type in ('generated_image', 'generated_video'):
                    displayable.append(message)
                    continue
                
                # Always show streaming placeholders (even if empty)
                if message.get('_streaming'):
                    displayable.append(message)
                    continue
                
                # Skip empty messages (no text and no image)
                if not text_content or not text_content.strip():
                    has_image = message.get('generated_image_path') or self._has_image_content(content)
                    if not has_image:
                        continue
                
                displayable.append(message)
            
            new_count = len(displayable)
            
            if has_streaming:
                # STREAMING MODE: Incremental updates for performance
                if new_count > existing_count:
                    # Add only the new messages
                    for i in range(existing_count, new_count):
                        self.conversation_display.add_message(displayable[i])
                elif new_count < existing_count:
                    # Count decreased during streaming (rare) - rebuild
                    self._rebuild_all_messages(displayable)
                # If count same: do nothing - streaming updates widget directly
            else:
                # NON-STREAMING MODE: Append-or-rebuild strategy
                # Build a lightweight fingerprint from message count + content lengths
                fp_parts = []
                for m in displayable:
                    c = m.get('content', '')
                    if isinstance(c, str):
                        fp_parts.append((m.get('role', ''), m.get('ai_name', ''), len(c), m.get('_type', '')))
                    else:
                        fp_parts.append((m.get('role', ''), m.get('ai_name', ''), -1, m.get('_type', '')))
                fingerprint = hash(tuple(fp_parts))

                if hasattr(self, '_last_render_fingerprint') and self._last_render_fingerprint == fingerprint:
                    return  # Nothing changed, skip entirely
                self._last_render_fingerprint = fingerprint

                # If only new messages were appended (existing are unchanged), just append
                if new_count > existing_count and existing_count > 0:
                    # Verify existing messages haven't changed (compare first few fingerprint parts)
                    old_fp = getattr(self, '_last_render_fp_parts', [])
                    if len(old_fp) == existing_count and old_fp == fp_parts[:existing_count]:
                        # Safe to append only the new ones
                        for i in range(existing_count, new_count):
                            self.conversation_display.add_message(displayable[i])
                        self._last_render_fp_parts = fp_parts
                        return

                self._last_render_fp_parts = fp_parts
                if existing_count > 0 or new_count > 0:
                    self._rebuild_all_messages(displayable)
                
        except Exception as e:
            print(f"[RENDER ERROR] _do_render: {e}")
            import traceback
            traceback.print_exc()
    
    def _rebuild_all_messages(self, displayable):
        """
        Rebuild all message widgets with scroll state preservation.
        
        Called when message count decreased (e.g., message deletion, conversation clear).
        
        CRITICAL: Preserves _should_follow so user's scroll position is respected!
        
        Process:
        1. Save current _should_follow state
        2. Disable visual updates (prevents flash)
        3. Block scroll detection (_programmatic_scroll = True)
        4. Clear and rebuild all widgets
        5. Restore states in correct order (scroll intent → programmatic flag → visual)
        6. Only scroll to bottom if user WAS following
        
        Debug log: "[SCROLL] Rebuild: ..." shows the preserved state
        """
        # Save scroll state before rebuild
        saved_should_follow = self.conversation_display._should_follow
        sb = self.conversation_display.verticalScrollBar()
        saved_scroll_value = sb.value()
        saved_scroll_max = sb.maximum()
        num_messages = len(displayable)
        
        if self._SCROLL_DEBUG:
            print(f"[SCROLL] Rebuild starting: {num_messages} messages, _should_follow={saved_should_follow}")
        
        # Disable updates to prevent visual flash
        self.conversation_display.setUpdatesEnabled(False)
        
        # Block scroll detection during rebuild
        self.conversation_display._programmatic_scroll = True
        
        try:
            self.conversation_display.clear_messages()  # reset_scroll=False by default
            for msg in displayable:
                self.conversation_display.add_message(msg)
        finally:
            # Restore states - ORDER MATTERS!
            # 1. Restore scroll intent BEFORE allowing scroll detection
            self.conversation_display._should_follow = saved_should_follow
            # 2. Then allow scroll detection again
            self.conversation_display._programmatic_scroll = False
            # 3. Re-enable visual updates
            self.conversation_display.setUpdatesEnabled(True)
            
            # Only scroll if user was following
            if saved_should_follow:
                self.conversation_display._schedule_scroll()
            elif saved_scroll_max > 0:
                # Restore approximate scroll position for users reading history
                QTimer.singleShot(16, lambda sv=saved_scroll_value, sm=saved_scroll_max: self._restore_scroll_position(sv, sm))

            if self._SCROLL_DEBUG:
                action = "will scroll" if saved_should_follow else "NO scroll (user scrolled away)"
                print(f"[SCROLL] Rebuild complete: {action}")
    
    def _restore_scroll_position(self, saved_value, saved_max, retries=3):
        """Restore scroll position after rebuild with retry.

        Uses absolute position when possible (content above didn't change),
        falls back to proportional positioning.
        """
        sb = self.conversation_display.verticalScrollBar()
        if sb.maximum() <= 0:
            if retries > 0:
                QTimer.singleShot(32, lambda: self._restore_scroll_position(saved_value, saved_max, retries - 1))
            return

        # Try absolute restore first (works when content above scroll position is unchanged)
        # Fall back to proportional if the content shifted significantly
        if sb.maximum() >= saved_value:
            target = saved_value
        elif saved_max > 0:
            ratio = saved_value / saved_max
            target = int(ratio * sb.maximum())
        else:
            return

        self.conversation_display._programmatic_scroll = True
        sb.setValue(target)
        self.conversation_display._programmatic_scroll = False

    def _extract_text_content(self, content):
        """Extract text from content (handles structured content with images)."""
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
            return ''.join(text_parts)
        return str(content) if content else ''
    
    def _has_image_content(self, content):
        """Check if content contains image data."""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image':
                    return True
        return False
    
    def _get_conversation_as_text(self):
        """Build plain text version of conversation for export."""
        lines = []
        for message in self.conversation:
            role = message.get('role', '')
            content = message.get('content', '')
            ai_name = message.get('ai_name', 'AI')
            model = message.get('model', '')
            
            # Extract text from structured content
            text_content = self._extract_text_content(content)
            
            # Skip empty messages
            if not text_content.strip():
                continue
            
            # Format based on role
            if role == 'user':
                lines.append(f"You:\n{text_content}\n")
            elif role == 'assistant':
                display_name = f"{ai_name} ({model})" if model else ai_name
                lines.append(f"{display_name}:\n{text_content}\n")
            elif role == 'system':
                lines.append(f"[System: {text_content}]\n")
        
        return '\n'.join(lines)
    
    # Keep _build_html_content for HTML export functionality
    def _build_html_content_for_export(self):
        """Build HTML content for conversation as a standalone HTML document."""

        # Build a full standalone HTML document for export
        html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Liminal Backrooms - Conversation Export</title>
<style>
"""
        # Add Google Fonts for better typography
        html += """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@700&display=swap');
"""
        html += f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
    font-size: 14px;
    line-height: 1.6;
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_normal']};
    padding: 0;
    margin: 0;
}}
.export-header {{
    background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, {COLORS['bg_medium']} 100%);
    border-bottom: 2px solid {COLORS['border_glow']};
    padding: 30px 40px;
    text-align: center;
}}
.export-header h1 {{
    font-family: 'Orbitron', sans-serif;
    color: {COLORS['accent_cyan']};
    font-size: 24px;
    letter-spacing: 4px;
    margin-bottom: 8px;
    text-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
}}
.export-header .meta {{
    color: {COLORS['text_dim']};
    font-size: 12px;
}}
.conversation-container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 40px;
}}
"""
        html += f".message {{ margin: 16px 0; padding: 14px 16px; border-radius: 0px; background-color: {COLORS['bg_medium']}; }}"
        html += f".user {{ border-right: 3px solid {COLORS['human']}; }}"
        html += f".ai-1 {{ border-left: 3px solid {COLORS['ai_1']}; }}"
        html += f".ai-2 {{ border-left: 3px solid {COLORS['ai_2']}; }}"
        html += f".ai-3 {{ border-left: 3px solid {COLORS['ai_3']}; }}"
        html += f".ai-4 {{ border-left: 3px solid {COLORS['ai_4']}; }}"
        html += f".ai-5 {{ border-left: 3px solid {COLORS['ai_5']}; }}"
        html += f".system {{ border-left: 3px solid {COLORS['text_dim']}; font-style: italic; }}"
        html += f".header {{ font-weight: bold; margin-bottom: 8px; }}"
        html += f".header-human {{ color: {COLORS['human']}; text-align: right; }}"
        html += f".header-ai-1 {{ color: {COLORS['ai_1']}; }}"
        html += f".header-ai-2 {{ color: {COLORS['ai_2']}; }}"
        html += f".header-ai-3 {{ color: {COLORS['ai_3']}; }}"
        html += f".header-ai-4 {{ color: {COLORS['ai_4']}; }}"
        html += f".header-ai-5 {{ color: {COLORS['ai_5']}; }}"
        html += f".content {{ white-space: pre-wrap; color: {COLORS['text_normal']}; }}"
        html += f".content-right {{ text-align: right; }}"
        html += f".branch-indicator {{ color: {COLORS['text_dim']}; font-style: italic; text-align: center; margin: 12px 4px; }}"
        html += f".rabbithole {{ color: {COLORS['accent_green']}; }}"
        html += f".fork {{ color: {COLORS['accent_yellow']}; }}"
        # Notification styles - error (pink), success (green), info (yellow)
        html += f".notify-error {{ background-color: #1A0000; border-left: 3px solid {COLORS['notify_error']}; padding: 10px 12px; margin: 12px 4px; color: {COLORS['notify_error']}; border-radius: 0px; }}"
        html += f".notify-success {{ background-color: #001A00; border-left: 3px solid {COLORS['notify_success']}; padding: 10px 12px; margin: 12px 4px; color: {COLORS['notify_success']}; border-radius: 0px; }}"
        html += f".notify-info {{ background-color: #1A1A00; border-left: 3px solid {COLORS['notify_info']}; padding: 10px 12px; margin: 12px 4px; color: {COLORS['notify_info']}; border-radius: 0px; }}"
        # Legacy agent-notification class (defaults to info style)
        html += f".agent-notification {{ background-color: #1A1A00; border-left: 3px solid {COLORS['notify_info']}; padding: 10px 12px; margin: 12px 4px; color: {COLORS['notify_info']}; border-radius: 0px; }}"
        # Code block styling - indented, visually distinct, contained within message
        html += f"pre {{ background-color: #001100; border: 1px solid #0D3B0D; border-left: 3px solid {COLORS['accent_purple']}; border-radius: 4px; padding: 12px 14px; margin: 12px 0 12px 12px; overflow-x: auto; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }}"
        html += f"code {{ font-family: 'Consolas', 'Monaco', 'Courier New', monospace; color: {COLORS['text_bright']}; font-size: 13px; line-height: 1.5; }}"
        # Inline code (not in pre block) - subtle background
        html += f".inline-code {{ background-color: #001100; color: #00FF41; border: 1px solid #0D3B0D; border-radius: 3px; padding: 2px 6px; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 12px; }}"
        # Typing indicator styles
        html += f".typing-indicator {{ background-color: {COLORS['bg_medium']}; padding: 10px 12px; border-radius: 0px; margin: 12px 4px; }}"
        html += f".typing-dots {{ color: {COLORS['text_dim']}; font-style: italic; }}"
        html += """
</style>
</head>
<body>
<div class="export-header">
    <h1>LIMINAL BACKROOMS</h1>
    <div class="meta">"""
        html += f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {sum(1 for m in self.conversation if m.get('role') == 'assistant')} AI turns"
        html += """</div>
</div>
<div class="conversation-container">
"""

        for i, message in enumerate(self.conversation):
            role = message.get("role", "")
            content = message.get("content", "")
            ai_name = message.get("ai_name", "")
            model = message.get("model", "")
            
            # Handle structured content (with images)
            has_image = False
            image_base64 = None
            generated_image_path = None
            text_content = ""
            
            # Check for generated image path (from AI image generation)
            if hasattr(message, "get") and callable(message.get):
                generated_image_path = message.get("generated_image_path", None)
                if generated_image_path and os.path.exists(generated_image_path):
                    has_image = True
            
            if isinstance(content, list):
                # Structured content with potential images
                for part in content:
                    if part.get('type') == 'text':
                        text_content += part.get('text', '')
                    elif part.get('type') == 'image':
                        has_image = True
                        source = part.get('source', {})
                        if source.get('type') == 'base64':
                            image_base64 = source.get('data', '')
            else:
                # Plain text content
                text_content = content
            
            # Skip empty or whitespace-only messages (no text and no image) - but NOT typing indicators
            if (not text_content or not text_content.strip()) and not has_image and message.get('_type') != 'typing_indicator':
                continue
            
            # Handle typing indicators with animated dots
            if message.get('_type') == 'typing_indicator':
                ai_name = message.get('ai_name', 'AI')
                model = message.get('model', '')
                ai_num = message.get('_ai_number', 1)
                display_name = f"{ai_name} ({model})" if model else ai_name
                
                # Use the AI's color for the border
                html += f'<div class="message typing-indicator ai-{ai_num}">'
                html += f'<div class="header header-ai-{ai_num}">{display_name}</div>'
                html += f'<div class="typing-dots">thinking...</div>'
                html += f'</div>'
                continue
                
            # Handle branch indicators with special styling
            if role == 'system' and message.get('_type') == 'branch_indicator':
                if "Rabbitholing down:" in content:
                    html += f'<div class="branch-indicator rabbithole">{content}</div>'
                elif "Forking off:" in content:
                    html += f'<div class="branch-indicator fork">{content}</div>'
                continue
            
            # Handle agent notifications with special styling based on type
            if role == 'system' and message.get('_type') == 'agent_notification':
                print(f"[GUI] Rendering agent notification: {text_content[:50]}...")
                # Determine notification type based on _command_success field
                command_success = message.get('_command_success')
                if command_success is False:
                    # Error/failure notification (pink)
                    notify_class = "notify-error"
                elif command_success is True:
                    # Success notification (green)
                    notify_class = "notify-success"
                else:
                    # Info notification (yellow) - default for neutral messages
                    notify_class = "notify-info"
                html += f'<div class="{notify_class}">{text_content}</div>'
                continue
            
            # Handle generated images with special styling (success notification style)
            if message.get('_type') == 'generated_image':
                creator = message.get('ai_name', 'AI')
                model = message.get('model', '')
                creator_display = f"{creator} ({model})" if model else creator
                
                # Get prompt - try _prompt field first, fallback to extracting from text content
                prompt = message.get('_prompt', '')
                if not prompt and isinstance(content, list):
                    # Try to extract prompt from text content like: !image "prompt here"
                    for part in content:
                        if part.get('type') == 'text':
                            text = part.get('text', '')
                            import re
                            match = re.search(r'!image\s+"([^"]+)"', text)
                            if match:
                                prompt = match.group(1)
                                break
                
                truncated_prompt = (prompt[:50] + '...') if len(prompt) > 50 else prompt
                
                # Just show text notification - image displays in the image preview tab
                html += f'<div class="notify-success" style="padding: 10px 12px;">'
                html += f'<div style="color: {COLORS["notify_success"]};">🎨 [{creator_display}]: !image "{truncated_prompt}" — generated successfully</div>'
                html += f'</div>'
                continue  # Always continue for generated_image type
            
            # Process content to handle code blocks
            processed_content = self.process_content_with_code_blocks(text_content) if text_content else ""
            
            # Add image display if present
            image_html = ""
            if has_image:
                if image_base64:
                    image_html = f'<div style="margin: 10px 0;"><img src="data:image/jpeg;base64,{image_base64}" style="max-width: 100%; border-radius: 8px;"></div>'
                elif generated_image_path and os.path.exists(generated_image_path):
                    from urllib.parse import quote
                    clean_path = generated_image_path.replace(os.sep, '/')
                    file_url = "file:///" + quote(clean_path, safe='/:')
                    image_html = f'<div style="margin: 10px 0;"><img src="{file_url}" style="max-width: 400px; border-radius: 8px; border: 1px solid {COLORS["border"]};"></div>'
            
            # Format based on role
            if role == 'user':
                # User message
                html += f'<div class="message user">'
                user_display = message.get('_user_name', 'Human User')
                html += f'<div class="header header-human">{user_display}</div>'
                if image_html:
                    html += image_html
                if processed_content:
                    html += f'<div class="content content-right">{processed_content}</div>'
                html += f'</div>'
            elif role == 'assistant':
                # AI message - determine AI number for color
                ai_num = 1
                if ai_name and '-' in ai_name:
                    try:
                        ai_num = int(ai_name.split('-')[1])
                    except (ValueError, IndexError):
                        ai_num = 1
                ai_num = max(1, min(5, ai_num))
                
                display_name = ai_name
                model_badge = ''
                if model:
                    display_name += f" ({model})"
                    # Create a model badge
                    model_badge = f' <span style="background-color: {COLORS["bg_dark"]}; color: {COLORS["text_dim"]}; font-size: 10px; padding: 2px 6px; border: 1px solid {COLORS["border"]}; margin-left: 8px;">{model}</span>'

                html += f'<div class="message ai-{ai_num}">'
                html += f'<div class="header header-ai-{ai_num}">{ai_name}{model_badge}</div>'
                if image_html:
                    html += image_html
                if processed_content:
                    html += f'<div class="content">{processed_content}</div>'
                html += f'</div>'
            elif role == 'system':
                # System message
                html += f'<div class="message system">'
                html += f'<div class="content">{processed_content}</div>'
                html += f'</div>'
        
        # Close the HTML document structure
        html += """
</div>
</body>
</html>
"""
        return html
    
    def process_content_with_code_blocks(self, content):
        """Process content to properly format code blocks for HTML export."""
        import re
        from html import escape
        
        # Extract code blocks BEFORE any escaping
        code_block_pattern = r'```(\w*)\n?(.*?)```'
        
        parts = []
        last_end = 0
        
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            # Text before this code block
            before_text = content[last_end:match.start()]
            parts.append(('text', before_text))
            
            # The code block itself
            lang = match.group(1) or ''
            code = match.group(2)
            parts.append(('code_block', code, lang.lower()))
            
            last_end = match.end()
        
        # Remaining text
        if last_end < len(content):
            parts.append(('text', content[last_end:]))
        
        # Process each part
        result = []
        
        for part in parts:
            if part[0] == 'code_block':
                code_content = part[1].rstrip('\n')
                language = part[2]
                
                # Escape the code content for HTML
                escaped_code = escape(code_content)
                
                # Language header
                lang_header = ''
                if language:
                    lang_header = (
                        f'<div style="background-color: #002200; padding: 6px 12px; '
                        f'border-bottom: 1px solid #0D3B0D;">'
                        f'<span style="color: #64748B; font-size: 11px; '
                        f'font-family: Consolas, Monaco, monospace; font-weight: bold; '
                        f'text-transform: uppercase;">{escape(language)}</span></div>'
                    )
                
                # Build code block - simple, no syntax highlighting to avoid regex issues
                result.append(
                    f'<div style="background-color: #001100; border: 1px solid #0D3B0D; '
                    f'border-radius: 4px; margin: 12px 0 12px 10px; overflow: hidden;">'
                    f'{lang_header}'
                    f'<pre style="margin: 0; padding: 12px 14px; background: transparent; '
                    f'font-family: Consolas, Monaco, monospace; font-size: 13px; '
                    f'line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; '
                    f'color: #E0E0E0;">{escaped_code}</pre></div>'
                )
            else:
                # Regular text - escape and process
                text_part = escape(part[1])
                
                # Convert markdown bold/italic
                text_part = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text_part)
                text_part = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text_part)
                
                # Process inline code
                text_part = re.sub(
                    r'`([^`\n]+)`',
                    r'<code style="background-color: #001100; color: #00FF41; padding: 2px 6px; '
                    r'border-radius: 3px; font-family: Consolas, Monaco, monospace; font-size: 12px;">\1</code>',
                    text_part
                )
                
                # Convert newlines
                text_part = text_part.replace('\n', '<br/>')
                
                result.append(text_part)
        
        return ''.join(result)
    
    def start_loading(self):
        """Start loading animation"""
        self.loading = True
        self.loading_dots = 0
        self.input_field.setEnabled(False)
        self.submit_button.setEnabled(False)
        self.reset_button.setEnabled(False)  # Disable reset during processing
        
        # Capture the current width before changing text
        current_width = self.submit_button.width()
        
        self.submit_button.setText("Processing...")
        self.loading_timer.start()
        
        # Update glow effect for processing state - dimmer cyan glow
        if hasattr(self.submit_button, 'shadow'):
            self.submit_button.shadow.setBlurRadius(12)
            self.submit_button.shadow.setColor(QColor(COLORS['accent_cyan']))
            self.submit_button.shadow.setOffset(0, 0)
        
        # Set disabled style with fixed width to prevent resizing
        self.submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border: 1px solid {COLORS['accent_cyan']};
                border-radius: 0px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
                min-width: {current_width - 30}px;
                max-width: {current_width}px;
            }}
        """)
    
    def stop_loading(self):
        """Stop loading animation"""
        self.loading = False
        self.loading_timer.stop()
        self.input_field.setEnabled(True)
        self.submit_button.setEnabled(True)
        self.reset_button.setEnabled(True)  # Re-enable reset button
        self.submit_button.setText("⚡ PROPAGATE")
        
        # Reset glow effect
        if hasattr(self.submit_button, 'shadow'):
            self.submit_button.shadow.setBlurRadius(8)
            self.submit_button.shadow.setColor(QColor(COLORS['accent_cyan']))
            self.submit_button.shadow.setOffset(0, 2)
            
        # Reset button style to match original PROPAGATE button (no fixed width)
        self.submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['accent_cyan']};
                border-radius: 0px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border: 1px solid {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_cyan_active']};
                color: {COLORS['text_bright']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_dim']};
                border: 1px solid {COLORS['border']};
            }}
        """)
    
    def update_loading_animation(self):
        """Update loading animation dots - always 3 characters for fixed width"""
        self.loading_dots = (self.loading_dots + 1) % 4
        # Use different dot patterns that are always 3 chars wide
        patterns = ["   ", ".  ", ".. ", "..."]
        self.submit_button.setText(f"Processing{patterns[self.loading_dots]}")
    
    def show_context_menu(self, position):
        """Show context menu at the given position."""
        from PyQt6.QtWidgets import QApplication
        widget = QApplication.focusWidget()
        if widget and hasattr(widget, 'textCursor'):
            cursor = widget.textCursor()
            if cursor.hasSelection():
                self.context_menu.exec(widget.mapToGlobal(position))

    def rabbithole_from_selection(self):
        """Create a rabbithole branch from selected text."""
        from gui import ConversationContextMenu
        text = ConversationContextMenu._get_selected_text()
        if text:
            self.context_menu.rabbitholeSelected.emit()

    def fork_from_selection(self):
        """Create a fork branch from selected text."""
        from gui import ConversationContextMenu
        text = ConversationContextMenu._get_selected_text()
        if text:
            self.context_menu.forkSelected.emit()
    
    def append_text(self, text, format_type="normal", ai_name=None):
        """Append text to a message widget (for streaming).
        
        With widget-based chat, this updates the matching message widget's content.
        
        Args:
            text: Text to append
            format_type: Style hint (currently unused with widgets)
            ai_name: Optional - if provided, finds the widget for this AI
        """
        try:
            target_widget = None
            
            if ai_name:
                # Find the widget that matches this AI (search from end)
                for widget in reversed(self.conversation_display.message_widgets):
                    if hasattr(widget, 'message_data'):
                        widget_ai = widget.message_data.get('ai_name', '')
                        widget_role = widget.message_data.get('role', '')
                        if widget_ai == ai_name and widget_role == 'assistant':
                            target_widget = widget
                            break
            
            # Fall back to last widget if no specific AI or not found
            if target_widget is None:
                target_widget = self.conversation_display.get_last_message_widget()
            
            if target_widget and target_widget._content_label:
                current_text = target_widget._content_label.text()
                target_widget._content_label.setText(current_text + text)
            
            # Auto-scroll on newlines or substantial text (debounced)
            if self.conversation_display._should_follow and ('\n' in text or len(text) > 20):
                self.conversation_display._scroll_to_bottom()
                
        except Exception as e:
            print(f"[RENDER ERROR] append_text: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_conversation(self):
        """Clear the conversation display for a new conversation."""
        # Clear with scroll reset since this is a fresh start
        self.conversation_display.clear_messages(reset_scroll=True)
        self.images = []
        
    def display_conversation(self, conversation, branch_data=None):
        """Display the conversation in the text edit widget"""
        # NOTE: Don't clear here! Let _do_render handle incremental updates.
        # If conversation shrinks (e.g., loading different branch), _do_render will clear.
        
        # Store conversation data
        self.conversation = conversation
        
        # Check if we're in a branch
        is_branch = branch_data is not None
        branch_type = branch_data.get('type', '') if is_branch else ''
        selected_text = branch_data.get('selected_text', '') if is_branch else ''
        
        # Update title if in a branch
        if is_branch:
            branch_emoji = "🐇" if branch_type == "rabbithole" else "🍴"
            self.title_label.setText(f"{branch_emoji} {branch_type.capitalize()}: {selected_text[:30]}...")
            self.info_label.setText(f"Branch conversation")
        else:
            self.title_label.setText("Liminal Backrooms")
            # Don't override info_label here - let mode selector control it
        
        # Debug: Print conversation to console
        print("\n--- DEBUG: Conversation Content ---")
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if "```" in content:
                print(f"Found code block in {role} message")
                print(f"Content snippet: {content[:100]}...")
        print("--- End Debug ---\n")
        
        # Render conversation
        self.render_conversation()
        
    def display_image(self, image_path):
        """Display an image in the conversation
        
        With widget-based chat, creates an image message widget.
        """
        try:
            # Check if the image path is valid
            if not image_path or not os.path.exists(image_path):
                self.append_text("[Image not found]\n", "error")
                return
            
            # Create a message data structure for the image
            image_message = {
                'role': 'system',
                '_type': 'generated_image',
                'generated_image_path': image_path,
                'ai_name': 'Image',
                'model': '',
                'content': '[Image]'
            }
            
            # Add as a message widget
            self.conversation_display.add_message(image_message)
            
            # Store the image path to prevent orphaning
            self.image_paths.append(image_path)
            
        except Exception as e:
            self.append_text(f"[Error displaying image: {str(e)}]\n", "error")
    
    def export_conversation(self):
        """Export the conversation and all session media to a timestamped folder"""
        # Set default directory - custom Dropbox path with fallbacks
        base_dir = r"C:\Users\sjeff\Dropbox\Stephen Work\LiminalBackrooms"
        
        # Fallback if that path doesn't exist
        if not os.path.exists(os.path.dirname(base_dir)):
            documents_path = os.path.join(os.path.expanduser("~"), "Documents")
            if os.path.exists(documents_path):
                base_dir = os.path.join(documents_path, "LiminalBackrooms")
            else:
                base_dir = os.path.join(os.getcwd(), "exports")
        
        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Generate a timestamped folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_folder_name = f"session_{timestamp}"
        
        # Let user select the parent directory (where timestamped folder will be created)
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Parent Folder for Export",
            base_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        
        # Use selected dir or default base_dir
        if selected_dir:
            parent_dir = selected_dir
        else:
            # User cancelled - ask if they want to use default
            reply = QMessageBox.question(
                self,
                "Use Default Location?",
                f"Export to default location?\n\n{os.path.join(base_dir, export_folder_name)}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                parent_dir = base_dir
            else:
                return
        
        # Always create a timestamped subfolder
        folder_name = os.path.join(parent_dir, export_folder_name)
        
        try:
            # Create the export folder
            os.makedirs(folder_name, exist_ok=True)
            
            # Get main window for accessing session data
            main_window = self.window()
            
            # Export conversation as multiple formats
            # Plain text - build from conversation data
            text_path = os.path.join(folder_name, "conversation.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(self._get_conversation_as_text())
            
            # HTML - build from conversation data
            html_path = os.path.join(folder_name, "conversation.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(self._build_html_content_for_export())
            
            # Full HTML document - copy the current session's HTML file
            # Check for session-specific file first, then fall back to generic
            current_html_file = getattr(main_window, 'current_html_file', None)
            if current_html_file and os.path.exists(current_html_file):
                shutil.copy2(current_html_file, os.path.join(folder_name, "conversation_full.html"))
            else:
                # Fallback to old location
                full_html_path = os.path.join(OUTPUTS_DIR, "conversation_full.html")
                if os.path.exists(full_html_path):
                    shutil.copy2(full_html_path, os.path.join(folder_name, "conversation_full.html"))
            
            # Copy session images
            images_copied = 0
            if hasattr(main_window, 'right_sidebar') and hasattr(main_window.right_sidebar, 'image_preview_pane'):
                session_images = main_window.right_sidebar.image_preview_pane.session_images
                if session_images:
                    images_dir = os.path.join(folder_name, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    for img_path in session_images:
                        if os.path.exists(img_path):
                            shutil.copy2(img_path, images_dir)
                            images_copied += 1
            
            # Copy session videos
            videos_copied = 0
            if hasattr(main_window, 'session_videos'):
                session_videos = main_window.session_videos
                if session_videos:
                    videos_dir = os.path.join(folder_name, "videos")
                    os.makedirs(videos_dir, exist_ok=True)
                    for vid_path in session_videos:
                        if os.path.exists(vid_path):
                            shutil.copy2(vid_path, videos_dir)
                            videos_copied += 1
            
            # Create a manifest/summary file
            manifest_path = os.path.join(folder_name, "manifest.txt")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                f.write(f"Liminal Backrooms Session Export\n")
                f.write(f"================================\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Contents:\n")
                f.write(f"- conversation.txt (plain text)\n")
                f.write(f"- conversation.html (HTML format)\n")
                if os.path.exists(os.path.join(folder_name, "conversation_full.html")):
                    f.write(f"- conversation_full.html (styled document)\n")
                f.write(f"- images/ ({images_copied} files)\n")
                f.write(f"- videos/ ({videos_copied} files)\n")
            
            # Status message
            status_msg = f"Exported to {folder_name} ({images_copied} images, {videos_copied} videos)"
            main_window.statusBar().showMessage(status_msg)
            print(f"Session exported to {folder_name}")
            print(f"  - {images_copied} images")
            print(f"  - {videos_copied} videos")
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Session exported successfully!\n\n"
                f"Location: {folder_name}\n\n"
                f"• Conversation (txt, html)\n"
                f"• {images_copied} images\n"
                f"• {videos_copied} videos"
            )
            
        except Exception as e:
            error_msg = f"Error exporting session: {str(e)}"
            QMessageBox.critical(self, "Export Error", error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()


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
        
        # More visible atmospheric colors with green tint
        pulse = 0.5 + 0.5 * math.sin(math.radians(self.bg_offset * 2))
        center_r = int(0 + 5 * pulse)
        center_g = int(15 + 20 * pulse)
        center_b = int(0 + 5 * pulse)
        
        gradient.setColorAt(0, QColor(center_r, center_g, center_b))
        gradient.setColorAt(0.4, QColor(2, 12, 4))
        gradient.setColorAt(1, QColor(1, 6, 2))
        
        painter.fillRect(self.rect(), gradient)
        
        # Add subtle glow lines at edges
        glow_alpha = int(15 + 10 * pulse)
        glow_color = QColor(0, 255, 65, glow_alpha)  # Green glow
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


class ScanlineOverlayWidget(QWidget):
    """Transparent overlay widget for CRT scanline effect"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.scanline_offset = 0
        self.intensity = 0.25  # More visible scanlines
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
    
    def start_animation(self):
        self.anim_timer.start(100)
    
    def stop_animation(self):
        self.anim_timer.stop()
    
    def _animate(self):
        self.scanline_offset = (self.scanline_offset + 1) % 4
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw horizontal scanlines - more visible
        line_alpha = int(255 * self.intensity)
        line_color = QColor(0, 0, 0, line_alpha)
        painter.setPen(QPen(line_color, 1))
        
        # Draw every 2nd line for more visible effect
        for y in range(self.scanline_offset, self.height(), 2):
            painter.drawLine(0, y, self.width(), y)
        
        # Subtle vignette effect at edges
        gradient = QRadialGradient(self.width() / 2, self.height() / 2, 
                                   max(self.width(), self.height()) * 0.7)
        gradient.setColorAt(0, QColor(0, 0, 0, 0))
        gradient.setColorAt(0.7, QColor(0, 0, 0, 0))
        gradient.setColorAt(1, QColor(0, 0, 0, int(255 * self.intensity * 1.5)))
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRect(self.rect())


class LiminalBackroomsApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Session tracking - timestamp for this session's files
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_html_file = None  # Will be set when conversation starts
        
        # Main app state
        self.conversation = []
        self.turn_count = 0
        self.images = []
        self.image_paths = []
        self.session_videos = []  # Track videos generated this session
        self.branch_conversations = {}  # Store branch conversations by ID
        self.active_branch = None      # Currently displayed branch
        self.effective_turn_delay = TURN_DELAY  # Speed-adjusted delay between turns
        
        # Set up the UI
        self.setup_ui()
        
        # Connect signals and slots
        self.connect_signals()

        # Keyboard shortcuts
        self._stop_requested = False
        self.setup_shortcuts()

        # Dark theme
        self.apply_dark_theme()
        
        # Restore splitter state if available
        self.restore_splitter_state()
        
        # Start maximized
        self.showMaximized()
    
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("╔═ LIMINAL BACKROOMS v0.7 ═╗")
        self.setGeometry(100, 100, 1600, 900)  # Initial size before maximizing
        self.setMinimumSize(1200, 800)
        
        # Create central widget - this will be a custom widget that paints the background
        self.central_container = CentralContainer()
        self.setCentralWidget(self.central_container)
        
        # Main layout for content
        main_layout = QHBoxLayout(self.central_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Create horizontal splitter for left and right panes
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)  # Slim handle
        self.splitter.setChildrenCollapsible(False)  # Prevent panes from being collapsed
        self.splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['border_highlight']};
                border: none;
                margin: 2px 0px;
            }}
            QSplitter::handle:hover {{
                background-color: {COLORS['border_glow']};
            }}
            QSplitter::handle:pressed {{
                background-color: {COLORS['accent_cyan']};
            }}
        """)
        main_layout.addWidget(self.splitter)
        
        # Create left pane (conversation) and right sidebar (tabbed: setup + network)
        self.left_pane = ConversationPane()
        self.right_sidebar = RightSidebar()
        
        # Set minimum widths to prevent UI from being cut off when resizing
        self.left_pane.setMinimumWidth(780)  # Chat panel needs space for message boxes
        self.right_sidebar.setMinimumWidth(350)  # Control panel needs space for controls
        
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.right_sidebar)
        
        # Set initial splitter sizes (70:30 ratio - more space for conversation)
        total_width = 1600  # Based on default window width
        self.splitter.setSizes([int(total_width * 0.70), int(total_width * 0.30)])
        
        # Initialize main conversation as root node
        self.right_sidebar.add_node('main', 'Seed', 'main')
        
        # ═══ SIGNAL INDICATOR ═══
        self.signal_indicator = SignalIndicator()
        
        # Status bar with modern styling
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
        
        # Add notification label for agent actions (shows latest notification)
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
        
        # ═══ ITERATION COUNTER ═══
        self.iteration_label = QLabel("")
        self.iteration_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 11px;
                padding: 2px 10px;
                background-color: transparent;
            }}
        """)
        self.statusBar().addPermanentWidget(self.iteration_label)
        
        # Add signal indicator to status bar
        self.statusBar().addPermanentWidget(self.signal_indicator)

        # ═══ SPEED CONTROLS ═══
        speed_container = QWidget()
        speed_container.setStyleSheet("background-color: transparent;")
        speed_layout = QHBoxLayout(speed_container)
        speed_layout.setContentsMargins(4, 0, 4, 0)
        speed_layout.setSpacing(2)

        speed_label = QLabel("SPEED:")
        speed_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: 10px;
                padding: 0px 2px;
                background-color: transparent;
            }}
        """)
        speed_layout.addWidget(speed_label)

        speed_btn_unchecked_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                border: 1px solid {COLORS['border']};
                padding: 1px 6px;
                font-size: 10px;
                border-radius: 0px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
            }}
        """
        speed_btn_checked_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_bright']};
                border: 1px solid {COLORS['accent_cyan']};
                padding: 1px 6px;
                font-size: 10px;
                border-radius: 0px;
            }}
        """

        self.speed_buttons = []
        for speed_text in ["0.5x", "1x", "2x", "5x"]:
            btn = QPushButton(speed_text)
            btn.setCheckable(True)
            btn.setStyleSheet(speed_btn_unchecked_style)
            btn.setFixedHeight(18)
            btn.clicked.connect(self._on_speed_changed)
            speed_layout.addWidget(btn)
            self.speed_buttons.append(btn)

        # Store styles for toggling
        self._speed_btn_unchecked_style = speed_btn_unchecked_style
        self._speed_btn_checked_style = speed_btn_checked_style

        # Default: 1x is checked
        self.speed_buttons[1].setChecked(True)
        self.speed_buttons[1].setStyleSheet(speed_btn_checked_style)

        self.statusBar().addPermanentWidget(speed_container)

        # ═══ ZOOM INDICATOR ═══
        self.zoom_label = QLabel("ZOOM:100%")
        self.zoom_label.setStyleSheet(f"""
            color: {COLORS['text_dim']};
            font-family: {FONTS['family_mono']};
            font-size: 9px;
            padding: 0 6px;
            background: transparent;
        """)
        self.statusBar().addPermanentWidget(self.zoom_label)

        # ═══ CRT TOGGLE CHECKBOX ═══
        self.crt_checkbox = QCheckBox("CRT")
        self.crt_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_dim']};
                font-size: 10px;
                spacing: 6px;
                margin-left: 12px;
                padding-left: 2px;
            }}
            QCheckBox::indicator {{
                width: 12px;
                height: 12px;
                border: 1px solid {COLORS['border_glow']};
                background: {COLORS['bg_dark']};
            }}
            QCheckBox::indicator:checked {{
                background: {COLORS['accent_cyan']};
            }}
        """)
        self.crt_checkbox.setToolTip("Toggle CRT scanline effect")
        self.crt_checkbox.toggled.connect(self.toggle_crt_effect)
        self.statusBar().addPermanentWidget(self.crt_checkbox)
        
        # Set up input callback
        self.left_pane.set_input_callback(self.handle_user_input)
    
    def _on_speed_changed(self):
        """Handle speed button toggle - enforce exclusive selection and update delay"""
        sender = self.sender()
        speed_map = {"0.5x": 2.0, "1x": 1.0, "2x": 0.5, "5x": 0.2}

        # Enforce radio-button behavior: uncheck all others, ensure sender is checked
        for btn in self.speed_buttons:
            if btn is sender:
                btn.setChecked(True)
                btn.setStyleSheet(self._speed_btn_checked_style)
            else:
                btn.setChecked(False)
                btn.setStyleSheet(self._speed_btn_unchecked_style)

        # Calculate effective delay
        multiplier = speed_map.get(sender.text(), 1.0)
        self.effective_turn_delay = TURN_DELAY * multiplier

        # Brief status bar message
        self.statusBar().showMessage(
            f"Speed set to {sender.text()} (delay: {self.effective_turn_delay:.1f}s)", 3000
        )

    def toggle_crt_effect(self, enabled):
        """Toggle the CRT scanline effect"""
        if hasattr(self, 'central_container'):
            self.central_container.set_scanlines_enabled(enabled)

    def _update_zoom_label(self, level):
        """Update zoom indicator in status bar."""
        self.zoom_label.setText(f"ZOOM:{level}%")

    def update_iteration(self, current: int, total: int, ai_name: str = ""):
        """
        Update the iteration counter in status bar.
        
        Args:
            current: Current turn number (1-based)
            total: Total number of turns
            ai_name: Optional - currently responding AI name
        """
        if ai_name:
            self.iteration_label.setText(f"Turn {current}/{total} — {ai_name}")
        else:
            self.iteration_label.setText(f"Turn {current}/{total}")
    
    def clear_iteration(self):
        """Clear the iteration counter (when conversation completes)"""
        self.iteration_label.setText("")
    
    def set_signal_active(self, active):
        """Set signal indicator to active (waiting for response)"""
        self.signal_indicator.set_active(active)
    
    def update_signal_latency(self, latency_ms):
        """Update signal indicator with response latency"""
        self.signal_indicator.set_latency(latency_ms)
    
    def connect_signals(self):
        """Connect all signals and slots"""
        # Node selection in network view
        self.right_sidebar.nodeSelected.connect(self.on_branch_select)
        
        # Node hover in network view
        if hasattr(self.right_sidebar.network_pane.network_view, 'nodeHovered'):
            self.right_sidebar.network_pane.network_view.nodeHovered.connect(self.on_node_hover)
        
        # Export button
        self.right_sidebar.control_panel.export_button.clicked.connect(self.export_conversation)

        # BackroomsBench evaluation button
        self.right_sidebar.control_panel.backroomsbench_button.clicked.connect(self.run_backroomsbench_evaluation)

        # Connect context menu actions to the main app methods
        self.left_pane.set_rabbithole_callback(self.branch_from_selection)
        self.left_pane.set_fork_callback(self.fork_from_selection)
        
        # Save splitter state when it moves
        self.splitter.splitterMoved.connect(self.save_splitter_state)
        
        # Connect mode selector to update info label
        self.right_sidebar.control_panel.mode_selector.currentTextChanged.connect(self.on_mode_changed)
    
    def setup_shortcuts(self):
        """Set up keyboard shortcuts for the application"""
        # Ctrl+Enter - Propagate (same as clicking PROPAGATE button)
        shortcut_propagate = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut_propagate.activated.connect(self.left_pane.handle_propagate_click)

        # Ctrl+E - Export conversation
        shortcut_export = QShortcut(QKeySequence("Ctrl+E"), self)
        shortcut_export.activated.connect(self.export_conversation)

        # Ctrl+Shift+N - Reset conversation
        shortcut_reset = QShortcut(QKeySequence("Ctrl+Shift+N"), self)
        shortcut_reset.activated.connect(self.left_pane.handle_reset_click)

        # Ctrl+F - Toggle conversation search
        shortcut_focus = QShortcut(QKeySequence("Ctrl+F"), self)
        shortcut_focus.activated.connect(self.left_pane.toggle_search)

        # Escape - Stop/cancel current operation
        shortcut_stop = QShortcut(QKeySequence("Escape"), self)
        shortcut_stop.activated.connect(self.request_stop)

        # F11 - Toggle fullscreen
        shortcut_fullscreen = QShortcut(QKeySequence("F11"), self)
        shortcut_fullscreen.activated.connect(self.toggle_fullscreen)

        # Ctrl+T - Toggle CRT effect
        shortcut_crt = QShortcut(QKeySequence("Ctrl+T"), self)
        shortcut_crt.activated.connect(lambda: self.crt_checkbox.setChecked(not self.crt_checkbox.isChecked()))

        # Ctrl+= / Ctrl+- for zoom
        shortcut_zoom_in = QShortcut(QKeySequence("Ctrl+="), self)
        shortcut_zoom_in.activated.connect(self.left_pane.conversation_display.zoom_in)

        shortcut_zoom_out = QShortcut(QKeySequence("Ctrl+-"), self)
        shortcut_zoom_out.activated.connect(self.left_pane.conversation_display.zoom_out)

        shortcut_zoom_reset = QShortcut(QKeySequence("Ctrl+0"), self)
        shortcut_zoom_reset.activated.connect(self.left_pane.conversation_display.zoom_reset)

    def request_stop(self):
        """Request stop of the current operation"""
        self._stop_requested = True
        self.notification_label.setText("Stop requested...")

    def toggle_fullscreen(self):
        """Toggle between fullscreen and normal window mode"""
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def on_mode_changed(self, mode):
        """Update the info label when conversation mode changes"""
        if mode == "Human-AI":
            self.left_pane.info_label.setText("[ HUMAN-TO-AI CONVERSATION ]")
        else:
            self.left_pane.info_label.setText("[ AI-TO-AI CONVERSATION ]")
    
    def handle_user_input(self, text):
        """Handle user input from the conversation pane"""
        # Add user message to conversation
        if text:
            user_message = {
                "role": "user",
                "content": text
            }
            self.conversation.append(user_message)
            
            # Update conversation display
            self.left_pane.update_conversation(self.conversation)
        
        # Process the conversation (this will be implemented in main.py)
        if hasattr(self, 'process_conversation'):
            self.process_conversation()
    
    def append_text(self, text, format_type="normal", ai_name=None):
        """Append text to the conversation display with the specified format"""
        self.left_pane.append_text(text, format_type, ai_name=ai_name)
    
    def clear_conversation(self):
        """Clear the conversation display and reset images"""
        self.left_pane.clear_conversation()
        self.conversation = []
        self.images = []
        self.image_paths = []
    
    def display_conversation(self, conversation, branch_data=None):
        """Display the conversation in the text edit widget"""
        self.left_pane.display_conversation(conversation, branch_data)
    
    def display_image(self, image_path):
        """Display an image in the conversation"""
        self.left_pane.display_image(image_path)
    
    def export_conversation(self):
        """Export the current conversation"""
        self.left_pane.export_conversation()

    def run_backroomsbench_evaluation(self):
        """Run BackroomsBench multi-judge evaluation on current session."""
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt, QTimer
        import threading

        # Get current conversation from ConversationPane (left pane)
        conversation = self.left_pane.conversation

        # Filter out special messages (branch indicators, etc.) - only count actual dialogue
        dialogue_messages = [
            msg for msg in conversation
            if isinstance(msg, dict) and msg.get('role') in ('user', 'assistant') and msg.get('_type') != 'branch_indicator'
        ]

        if len(dialogue_messages) < 5:
            QMessageBox.warning(
                self,
                "Not Enough Content",
                f"Need at least 5 dialogue messages for evaluation.\nYou have {len(dialogue_messages)}. Let the dialogue deepen. 🌀"
            )
            return

        # Get scenario name from UI
        scenario_name = self.right_sidebar.control_panel.prompt_pair_selector.currentText()

        # Get participant models based on number of active AIs
        num_ais = int(self.right_sidebar.control_panel.num_ais_selector.currentText())
        participant_models = []
        model_selectors = [
            self.right_sidebar.control_panel.ai1_model_selector,
            self.right_sidebar.control_panel.ai2_model_selector,
            self.right_sidebar.control_panel.ai3_model_selector,
            self.right_sidebar.control_panel.ai4_model_selector,
            self.right_sidebar.control_panel.ai5_model_selector,
        ]

        for i in range(num_ais):
            model_text = model_selectors[i].currentText()
            participant_models.append(model_text)

        # Show progress dialog
        progress = QProgressDialog(
            "🌀 Running BackroomsBench...\n\nSending to 3 judges (Opus, Gemini, GPT)",
            None, 0, 0, self
        )
        progress.setWindowTitle("BackroomsBench Evaluation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Store result for callback
        self._backroomsbench_result = None
        self._backroomsbench_error = None
        self._backroomsbench_progress = progress

        def run_eval():
            try:
                from backroomsbench import run_backroomsbench
                self._backroomsbench_result = run_backroomsbench(
                    conversation=dialogue_messages,
                    scenario_name=scenario_name,
                    participant_models=participant_models
                )
            except Exception as e:
                print(f"[BackroomsBench] Error: {e}")
                self._backroomsbench_error = str(e)

        def check_complete():
            if self._backroomsbench_result is not None:
                progress.close()
                result = self._backroomsbench_result
                self.statusBar().showMessage(
                    f"🌀 BackroomsBench complete! {result['summary']['successful_evaluations']}/3 judges filed reports"
                )
                import subprocess
                import sys as _sys
                try:
                    output_dir = result["output_dir"]
                    if _sys.platform == 'win32':
                        subprocess.Popen(['explorer', output_dir])
                    elif _sys.platform == 'darwin':
                        subprocess.Popen(['open', output_dir])
                    else:
                        subprocess.Popen(['xdg-open', output_dir])
                except Exception:
                    pass
                self._backroomsbench_result = None
                self._backrooms_check_timer.stop()
            elif self._backroomsbench_error is not None:
                progress.close()
                QMessageBox.critical(
                    self,
                    "BackroomsBench Error",
                    f"Evaluation failed:\n{self._backroomsbench_error}"
                )
                self._backroomsbench_error = None
                self._backrooms_check_timer.stop()

        # Start background thread
        threading.Thread(target=run_eval, daemon=True).start()

        # Poll for completion
        self._backrooms_check_timer = QTimer()
        self._backrooms_check_timer.timeout.connect(check_complete)
        self._backrooms_check_timer.start(500)

    def on_node_hover(self, node_id):
        """Handle node hover in the network view"""
        if node_id == 'main':
            self.statusBar().showMessage("Main conversation")
        elif node_id in self.branch_conversations:
            branch_data = self.branch_conversations[node_id]
            branch_type = branch_data.get('type', 'branch')
            selected_text = branch_data.get('selected_text', '')
            self.statusBar().showMessage(f"{branch_type.capitalize()}: {selected_text[:50]}...")
    
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
        
        # Add specific styling for branch messages
        branch_header_format = QTextCharFormat()
        branch_header_format.setForeground(QColor(COLORS['ai_header']))
        branch_header_format.setFontWeight(QFont.Weight.Bold)
        branch_header_format.setFontPointSize(11)
        
        branch_inline_format = QTextCharFormat()
        branch_inline_format.setForeground(QColor(COLORS['ai_header']))
        branch_inline_format.setFontItalic(True)
        branch_inline_format.setFontPointSize(10)
        
        # Add formats to the left pane
        self.left_pane.text_formats["branch_header"] = branch_header_format
        self.left_pane.text_formats["branch_inline"] = branch_inline_format
    
    def on_branch_select(self, branch_id):
        """Handle branch selection in the network view"""
        try:
            # Check if branch exists
            if branch_id == 'main':
                # Switch to main conversation
                self.active_branch = None
                # Make sure we have a main_conversation attribute
                if not hasattr(self, 'main_conversation'):
                    self.main_conversation = []
                self.conversation = self.main_conversation
                self.left_pane.update_conversation(self.conversation)
                self.statusBar().showMessage("Switched to main conversation")
                return
            
            if branch_id not in self.branch_conversations:
                self.statusBar().showMessage(f"Branch {branch_id} not found")
                return
            
            # Get branch data
            branch_data = self.branch_conversations[branch_id]
            
            # Set active branch
            self.active_branch = branch_id
            
            # Update conversation
            self.conversation = branch_data['conversation']
            
            # Display the conversation with branch metadata
            self.left_pane.display_conversation(self.conversation, branch_data)
            
            # Update status bar
            self.statusBar().showMessage(f"Switched to {branch_data['type']} branch: {branch_id}")
            
        except Exception as e:
            print(f"Error selecting branch: {e}")
            self.statusBar().showMessage(f"Error selecting branch: {e}")
    
    def branch_from_selection(self, selected_text):
        """Create a rabbithole branch from selected text"""
        if not selected_text:
            return
        
        # Create branch
        branch_id = self.create_branch(selected_text, 'rabbithole')
        
        # Switch to branch
        self.on_branch_select(branch_id)
    
    def fork_from_selection(self, selected_text):
        """Create a fork branch from selected text"""
        if not selected_text:
            return
        
        # Create branch
        branch_id = self.create_branch(selected_text, 'fork')
        
        # Switch to branch
        self.on_branch_select(branch_id)
    
    def create_branch(self, selected_text, branch_type="rabbithole", parent_branch=None):
        """Create a new branch in the conversation"""
        try:
            # Generate a unique ID for the branch
            branch_id = str(uuid.uuid4())
            
            # Get parent branch ID
            parent_id = parent_branch if parent_branch else (self.active_branch if self.active_branch else 'main')
            
            # Get current conversation
            if parent_id == 'main':
                # If parent is main, use main conversation
                if not hasattr(self, 'main_conversation'):
                    self.main_conversation = []
                current_conversation = self.main_conversation.copy()
            else:
                # Otherwise, use parent branch conversation
                parent_data = self.branch_conversations.get(parent_id)
                if parent_data:
                    current_conversation = parent_data['conversation'].copy()
                else:
                    current_conversation = []
            
            # Create initial message based on branch type
            if branch_type == 'fork':
                initial_message = {
                    "role": "user",
                    "content": f"Complete this thought or sentence naturally, continuing forward from exactly this point: '{selected_text}'"
                }
            else:  # rabbithole
                initial_message = {
                    "role": "user",
                    "content": f"Let's explore and expand upon the concept of '{selected_text}' from our previous discussion."
                }
            
            # Create branch conversation with initial message
            branch_conversation = current_conversation.copy()
            branch_conversation.append(initial_message)
            
            # Create branch data
            branch_data = {
                'id': branch_id,
                'parent': parent_id,
                'type': branch_type,
                'selected_text': selected_text,
                'conversation': branch_conversation,
                'turn_count': 0,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'history': current_conversation
            }
            
            # Store branch data
            self.branch_conversations[branch_id] = branch_data
            
            # Add node to network graph - make sure parameters are in the correct order
            node_label = f"{branch_type.capitalize()}: {selected_text[:20]}{'...' if len(selected_text) > 20 else ''}"
            self.right_sidebar.add_node(branch_id, node_label, branch_type)
            self.right_sidebar.add_edge(parent_id, branch_id)
            
            # Set active branch to this new branch
            self.active_branch = branch_id
            self.conversation = branch_conversation
            
            # Display the conversation
            self.left_pane.display_conversation(branch_conversation, branch_data)
            
            # Trigger AI response processing for this branch
            if hasattr(self, 'process_branch_conversation'):
                # Add a small delay to ensure UI updates first
                QTimer.singleShot(100, lambda: self.process_branch_conversation(branch_id))
            
            # Return branch ID
            return branch_id
            
        except Exception as e:
            print(f"Error creating branch: {e}")
            self.statusBar().showMessage(f"Error creating branch: {e}")
            return None
    
    def get_branch_path(self, branch_id):
        """Get the full path of branch names from root to the given branch"""
        try:
            path = []
            current_id = branch_id
            
            # Prevent potential infinite loops by tracking visited branches
            visited = set()
            
            while current_id != 'main' and current_id not in visited:
                visited.add(current_id)
                branch_data = self.branch_conversations.get(current_id)
                if not branch_data:
                    break
                    
                # Get a readable version of the selected text (truncated if needed)
                selected_text = branch_data.get('selected_text', '')
                if selected_text:
                    display_text = f"{selected_text[:20]}{'...' if len(selected_text) > 20 else ''}"
                    path.append(display_text)
                else:
                    path.append(f"{branch_data.get('type', 'Branch').capitalize()}")
                
                # Check for valid parent attribute
                current_id = branch_data.get('parent')
                if not current_id:
                    break
            
            path.append('Seed')
            return ' → '.join(reversed(path))
        except Exception as e:
            print(f"Error building branch path: {e}")
            return f"Branch {branch_id}"
    
    def save_splitter_state(self):
        """Save the current splitter state to a file"""
        try:
            # Create settings directory if it doesn't exist
            if not os.path.exists('settings'):
                os.makedirs('settings')
                
            # Save splitter state to file
            with open('settings/splitter_state.json', 'w') as f:
                json.dump({
                    'sizes': self.splitter.sizes()
                }, f)
        except Exception as e:
            print(f"Error saving splitter state: {e}")
    
    def restore_splitter_state(self):
        """Restore the splitter state from a file if available"""
        try:
            if os.path.exists('settings/splitter_state.json'):
                with open('settings/splitter_state.json', 'r') as f:
                    state = json.load(f)
                    if 'sizes' in state:
                        self.splitter.setSizes(state['sizes'])
        except Exception as e:
            print(f"Error restoring splitter state: {e}")
            # Fall back to default sizes
            total_width = self.width()
            self.splitter.setSizes([int(total_width * 0.7), int(total_width * 0.3)])

    def process_branch_conversation(self, branch_id):
        """Process the branch conversation using the selected models"""
        # This method will be implemented in main.py to avoid circular imports
        pass

    def node_clicked(self, node_id):
        """Handle node click in the network view"""
        print(f"Node clicked: {node_id}")
        
        # Check if this is the main conversation or a branch
        if node_id == 'main':
            # Switch to main conversation
            self.active_branch = None
            self.left_pane.display_conversation(self.main_conversation)
        elif node_id in self.branch_conversations:
            # Switch to branch conversation
            self.active_branch = node_id
            branch_data = self.branch_conversations[node_id]
            conversation = branch_data['conversation']
            
            # Filter hidden messages for display
            visible_conversation = get_visible_messages(conversation)
            self.left_pane.display_conversation(visible_conversation, branch_data)

    def initialize_selectors(self):
        """Initialize the AI model selectors and prompt pair selector"""
        pass

    # Removed: create_new_living_document