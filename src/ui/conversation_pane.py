from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QMenu, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt6.QtGui import QAction, QTextCursor, QTextCharFormat, QFont, QColor, QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QTimer, QPropertyAnimation
import os
import base64
from datetime import datetime
import shutil
from src.ui.colors import COLORS
from src.ui.widgets.custom_widgets import GlowButton

class ConversationContextMenu(QMenu):
    """Context menu for the conversation display"""
    rabbitholeSelected = pyqtSignal()
    forkSelected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
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

class ConversationPane(QWidget):
    """Left pane containing the conversation and input area"""
    def __init__(self):
        super().__init__()

        # Set up the UI
        self.setup_ui()

        # Connect signals and slots
        self.connect_signals()

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

        self.info_label = QLabel("[ AI-TO-AI PROPAGATION ]")
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

        # Conversation display (read-only text edit in a scroll area)
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conversation_display.customContextMenuRequested.connect(self.show_context_menu)

        # Set font for conversation display - use Iosevka Term for better ASCII art rendering
        font = QFont("Iosevka Term", 10)
        # Set fallbacks in case Iosevka Term isn't loaded
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.conversation_display.setFont(font)

        # Apply cyberpunk styling
        self.conversation_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 15px;
                selection-background-color: {COLORS['accent_cyan']};
                selection-color: {COLORS['bg_dark']};
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

        # Input area with label
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)  # Reduced spacing

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
                padding: 6px 10px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
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

        # Clear button with subtle glow
        self.clear_button = GlowButton("CLEAR", COLORS['accent_pink'])
        self.clear_button.shadow.setBlurRadius(5)  # Subtler glow
        self.clear_button.base_blur = 5
        self.clear_button.hover_blur = 12
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 3px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border: 2px solid {COLORS['accent_pink']};
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
                border: 2px solid {COLORS['accent_cyan']};
                border-radius: 3px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border: 2px solid {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_cyan_active']};
                color: {COLORS['text_bright']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_dim']};
                border: 2px solid {COLORS['border']};
            }}
        """)

        # Add buttons to layout
        button_layout.addWidget(self.upload_image_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.submit_button)

        # Add input container to main layout
        input_layout.addWidget(button_container)

        # Add widgets to layout with adjusted stretch factors
        layout.addWidget(self.conversation_display, 1)  # Main conversation area gets most space
        layout.addWidget(input_container, 0)  # Input area gets minimal space

    def connect_signals(self):
        """Connect signals and slots"""
        # Submit button
        self.submit_button.clicked.connect(self.handle_propagate_click)

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
        self.upload_image_button.setText("📎 Image")
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
        self.upload_image_button.setText("📎 Image")
        self.input_field.setPlaceholderText("Seed the conversation or just click propagate...")

        # Always call the input callback, even with empty input
        if hasattr(self, 'input_callback') and self.input_callback:
            self.input_callback(message_data)

        # Start loading animation
        self.start_loading()

    def set_input_callback(self, callback):
        """Set callback function for input submission"""
        self.input_callback = callback

    def set_rabbithole_callback(self, callback):
        """Set callback function for rabbithole creation"""
        self.rabbithole_callback = callback

    def set_fork_callback(self, callback):
        """Set callback function for fork creation"""
        self.fork_callback = callback

    def update_conversation(self, conversation):
        """Update conversation display"""
        self.conversation = conversation
        self.render_conversation()

    def render_conversation(self):
        """Render conversation in the display"""
        # Save scroll position before re-rendering
        scrollbar = self.conversation_display.verticalScrollBar()
        old_scroll_value = scrollbar.value()
        old_scroll_max = scrollbar.maximum()
        was_at_bottom = old_scroll_value >= old_scroll_max - 20

        # Clear display
        self.conversation_display.clear()

        # Create HTML for conversation with modern styling
        html = "<style>"
        html += f"body {{ font-family: 'Iosevka Term', 'Consolas', 'Monaco', monospace; font-size: 10pt; line-height: 1.4; }}"
        html += f".message {{ margin-bottom: 10px; padding: 8px; border-radius: 4px; }}"
        html += f".user {{ background-color: {COLORS['bg_medium']}; }}"
        html += f".assistant {{ background-color: {COLORS['bg_medium']}; }}"
        html += f".system {{ background-color: {COLORS['bg_medium']}; font-style: italic; }}"
        html += f".header {{ font-weight: bold; margin: 10px 0; color: {COLORS['accent_blue']}; }}"
        html += f".content {{ white-space: pre-wrap; color: {COLORS['text_normal']}; }}"
        html += f".branch-indicator {{ color: {COLORS['text_dim']}; font-style: italic; text-align: center; margin: 8px 0; }}"
        html += f".rabbithole {{ color: {COLORS['accent_green']}; }}"
        html += f".fork {{ color: {COLORS['accent_yellow']}; }}"
        html += f".agent-notification {{ background-color: #1a2a2a; border-left: 3px solid {COLORS['accent_cyan']}; padding: 8px 12px; margin: 8px 0; color: {COLORS['accent_cyan']}; font-style: normal; }}"
        html += f"pre {{ background-color: {COLORS['bg_dark']}; border: 1px solid {COLORS['border']}; border-radius: 3px; padding: 8px; overflow-x: auto; margin: 8px 0; }}"
        html += f"code {{ font-family: 'Iosevka Term', 'Consolas', 'Monaco', monospace; color: {COLORS['text_bright']}; }}"
        html += "</style>"

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

            # Skip empty messages (no text and no image)
            if not text_content and not has_image:
                continue

            # Handle branch indicators with special styling
            if role == 'system' and message.get('_type') == 'branch_indicator':
                if "Rabbitholing down:" in content:
                    html += f'<div class="branch-indicator rabbithole">{content}</div>'
                elif "Forking off:" in content:
                    html += f'<div class="branch-indicator fork">{content}</div>'
                continue

            # Handle agent notifications with special styling
            if role == 'system' and message.get('_type') == 'agent_notification':
                html += f'<div class="agent-notification">{text_content}</div>'
                continue

            # Handle generated images with special styling
            if message.get('_type') == 'generated_image':
                creator = message.get('ai_name', 'AI')
                model = message.get('model', '')
                creator_display = f"{creator} ({model})" if model else creator
                if generated_image_path and os.path.exists(generated_image_path):
                    file_url = f"file:///{generated_image_path.replace(os.sep, '/')}"
                    html += f'<div class="message" style="background-color: #1a1a2e; border: 1px solid {COLORS["accent_purple"]}; text-align: center; padding: 12px;">'
                    html += f'<div style="color: {COLORS["accent_purple"]}; margin-bottom: 8px;">🎨 {creator_display} created an image</div>'
                    html += f'<img src="{file_url}" style="max-width: 100%; border-radius: 8px;" />'
                    if text_content:
                        # Extract just the prompt part
                        html += f'<div style="color: {COLORS["text_dim"]}; font-size: 9pt; margin-top: 8px; font-style: italic;">{text_content}</div>'
                    html += f'</div>'
                continue

            # Process content to handle code blocks
            processed_content = self.process_content_with_code_blocks(text_content) if text_content else ""

            # Add image display if present
            image_html = ""
            if has_image:
                if image_base64:
                    image_html = f'<div style="margin: 10px 0;"><img src="data:image/jpeg;base64,{image_base64}" style="max-width: 100%; border-radius: 8px;" /></div>'
                elif generated_image_path and os.path.exists(generated_image_path):
                    # Use file:// URL for local generated images
                    file_url = f"file:///{generated_image_path.replace(os.sep, '/')}"
                    image_html = f'<div style="margin: 10px 0; text-align: center;"><img src="{file_url}" style="max-width: 400px; border-radius: 8px; border: 1px solid {COLORS["border"]};" /><div style="font-size: 9pt; color: {COLORS["text_dim"]}; margin-top: 4px;">🎨 Generated image</div></div>'

            # Format based on role
            if role == 'user':
                # User message
                html += f'<div class="message user">'
                if image_html:
                    html += image_html
                if processed_content:
                    html += f'<div class="content">{processed_content}</div>'
                html += f'</div>'
            elif role == 'assistant':
                # AI message
                display_name = ai_name
                if model:
                    display_name += f" ({model})"
                html += f'<div class="message assistant">'
                html += f'<div class="header">\n{display_name}\n</div>'
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

        # Set HTML in display
        self.conversation_display.setHtml(html)

        # Restore scroll position
        if was_at_bottom:
            self.conversation_display.verticalScrollBar().setValue(
                self.conversation_display.verticalScrollBar().maximum()
            )
        else:
            new_max = self.conversation_display.verticalScrollBar().maximum()
            if old_scroll_max > 0 and new_max > 0:
                self.conversation_display.verticalScrollBar().setValue(
                    min(old_scroll_value, new_max)
                )
            else:
                self.conversation_display.verticalScrollBar().setValue(old_scroll_value)

    def process_content_with_code_blocks(self, content):
        """Process content to properly format code blocks"""
        import re
        from html import escape

        # First, escape HTML in the content
        escaped_content = escape(content)

        # Check if there are any code blocks in the content
        if "```" not in escaped_content:
            return escaped_content

        # Split the content by code block markers
        parts = re.split(r'(```(?:[a-zA-Z0-9_]*)\n.*?```)', escaped_content, flags=re.DOTALL)

        result = []
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                # This is a code block
                try:
                    # Extract language if specified
                    language_match = re.match(r'```([a-zA-Z0-9_]*)\n', part)
                    language = language_match.group(1) if language_match else ""

                    # Extract code content
                    code_content = part[part.find('\n')+1:part.rfind('```')]

                    # Format as HTML
                    formatted_code = f'<pre><code class="language-{language}">{code_content}</code></pre>'
                    result.append(formatted_code)
                except Exception as e:
                    # If there's an error, just add the original escaped content
                    print(f"Error processing code block: {e}")
                    result.append(part)
            else:
                # Process inline code in non-code-block parts
                inline_parts = re.split(r'(`[^`]+`)', part)
                processed_part = []

                for inline_part in inline_parts:
                    if inline_part.startswith("`") and inline_part.endswith("`") and len(inline_part) > 2:
                        # This is inline code
                        code = inline_part[1:-1]
                        processed_part.append(f'<code>{code}</code>')
                    else:
                        processed_part.append(inline_part)

                result.append(''.join(processed_part))

        return ''.join(result)

    def start_loading(self):
        """Start loading animation"""
        self.loading = True
        self.loading_dots = 0
        self.input_field.setEnabled(False)
        self.submit_button.setEnabled(False)
        self.submit_button.setText("Processing")
        self.loading_timer.start()

        # Add subtle pulsing animation to the button
        self.pulse_animation = QPropertyAnimation(self.submit_button, b"styleSheet")
        self.pulse_animation.setDuration(1000)
        self.pulse_animation.setLoopCount(-1)  # Infinite loop

        # Define keyframes for the animation
        normal_style = f"""
            QPushButton {{
                background-color: {COLORS['border']};
                color: {COLORS['text_dim']};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
        """

        pulse_style = f"""
            QPushButton {{
                background-color: {COLORS['border_highlight']};
                color: {COLORS['text_dim']};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
        """

        self.pulse_animation.setStartValue(normal_style)
        self.pulse_animation.setEndValue(pulse_style)
        self.pulse_animation.start()

    def stop_loading(self):
        """Stop loading animation"""
        self.loading = False
        self.loading_timer.stop()
        self.input_field.setEnabled(True)
        self.submit_button.setEnabled(True)
        self.submit_button.setText("Propagate")

        # Stop the pulsing animation
        if hasattr(self, 'pulse_animation'):
            self.pulse_animation.stop()

        # Reset button style
        self.submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['accent_cyan']};
                border-radius: 0px;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_cyan_active']};
                color: {COLORS['text_bright']};
            }}
        """)

    def update_loading_animation(self):
        """Update loading animation dots"""
        self.loading_dots = (self.loading_dots + 1) % 4
        dots = "." * self.loading_dots
        self.submit_button.setText(f"Processing{dots}")

    def show_context_menu(self, position):
        """Show context menu at the given position"""
        # Get selected text
        cursor = self.conversation_display.textCursor()
        selected_text = cursor.selectedText()

        # Only show context menu if text is selected
        if selected_text:
            # Show the context menu at cursor position
            self.context_menu.exec(self.conversation_display.mapToGlobal(position))

    def rabbithole_from_selection(self):
        """Create a rabbithole branch from selected text"""
        cursor = self.conversation_display.textCursor()
        selected_text = cursor.selectedText()

        if selected_text and hasattr(self, 'rabbithole_callback'):
            self.rabbithole_callback(selected_text)

    def fork_from_selection(self):
        """Create a fork branch from selected text"""
        cursor = self.conversation_display.textCursor()
        selected_text = cursor.selectedText()

        if selected_text and hasattr(self, 'fork_callback'):
            self.fork_callback(selected_text)

    def append_text(self, text, format_type="normal"):
        """Append text to the conversation display with the specified format"""
        # Check if user is at the bottom before appending
        scrollbar = self.conversation_display.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 20

        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Apply the format if specified
        if format_type in self.text_formats:
            self.conversation_display.setCurrentCharFormat(self.text_formats[format_type])

        # Insert the text
        cursor.insertText(text)

        # Reset to normal format after insertion
        if format_type != "normal":
            self.conversation_display.setCurrentCharFormat(self.text_formats["normal"])

        # Only auto-scroll if user was already at the bottom
        if was_at_bottom:
            self.conversation_display.setTextCursor(cursor)
            self.conversation_display.ensureCursorVisible()

    def clear_conversation(self):
        """Clear the conversation display"""
        self.conversation_display.clear()
        self.images = []

    def display_conversation(self, conversation, branch_data=None):
        """Display the conversation in the text edit widget"""
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
            self.info_label.setText("AI-to-AI conversation")

        # Render conversation
        self.render_conversation()

    def display_image(self, image_path):
        """Display an image in the conversation"""
        try:
            # Check if the image path is valid
            if not image_path or not os.path.exists(image_path):
                self.append_text("[Image not found]\n", "error")
                return

            # Load the image
            image = QImage(image_path)
            if image.isNull():
                self.append_text("[Invalid image format]\n", "error")
                return

            # Create a pixmap from the image
            pixmap = QPixmap.fromImage(image)

            # Scale the image to fit the conversation display
            max_width = self.conversation_display.width() - 50
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)

            # Insert the image into the conversation display
            cursor = self.conversation_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertImage(pixmap.toImage())
            cursor.insertText("\n\n")

            # Store the image to prevent garbage collection
            self.images.append(pixmap)
            self.image_paths.append(image_path)

        except Exception as e:
            self.append_text(f"[Error displaying image: {str(e)}]\n", "error")

    def export_conversation(self):
        """Export the conversation and all session media to a folder"""
        base_dir = os.path.join(os.getcwd(), "exports")

        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Generate a session folder name based on date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_folder = os.path.join(base_dir, f"session_{timestamp}")

        # Get the folder from a dialog
        folder_name = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder (or create new)",
            base_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if not folder_name:
            reply = QMessageBox.question(
                self,
                "Create Export Folder?",
                f"Create new export folder?\n\n{default_folder}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                folder_name = default_folder
            else:
                return

        try:
            # Create the export folder
            os.makedirs(folder_name, exist_ok=True)

            # Get main window for accessing session data
            main_window = self.window()

            # Export conversation as multiple formats
            # Plain text
            text_path = os.path.join(folder_name, "conversation.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(self.conversation_display.toPlainText())

            # HTML
            html_path = os.path.join(folder_name, "conversation.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(self.conversation_display.toHtml())

            # Full HTML document if it exists
            full_html_path = os.path.join(os.getcwd(), "conversation_full.html")
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
            if main_window.statusBar():
                main_window.statusBar().showMessage(status_msg)
            print(f"Session exported to {folder_name}")

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
