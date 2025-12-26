import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QSize
from src.ui.colors import COLORS

class ImagePreviewPane(QWidget):
    """Pane to display generated images with navigation"""
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.session_images = []  # List of all images generated this session
        self.current_index = -1   # Current image index
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title label
        self.title = QLabel("🎨 GENERATED IMAGES")
        self.title.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_purple']};
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
            }}
        """)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        # Image display label
        self.image_label = QLabel("No images generated yet")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                border: 2px dashed {COLORS['border']};
                border-radius: 8px;
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
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_purple']};
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
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_purple']};
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
                border-radius: 4px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_purple']};
            }}
        """)
        self.open_button.clicked.connect(self.open_images_folder)
        layout.addWidget(self.open_button)

    def add_image(self, image_path):
        """Add a new image to the session gallery and display it"""
        if image_path and os.path.exists(image_path):
            if image_path not in self.session_images:
                self.session_images.append(image_path)
            self.current_index = len(self.session_images) - 1
            self._display_current()

    def set_image(self, image_path):
        """Display an image - also adds to gallery if new"""
        self.add_image(image_path)

    def _display_current(self):
        """Display the image at current_index"""
        if not self.session_images or self.current_index < 0:
            self.image_label.setText("No images generated yet")
            self.info_label.setText("")
            self.position_label.setText("")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        image_path = self.session_images[self.current_index]
        self.current_image_path = image_path

        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.image_label.size() - QSize(20, 20),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
                self.image_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {COLORS['bg_medium']};
                        border: 2px solid {COLORS['accent_purple']};
                        border-radius: 8px;
                        padding: 10px;
                    }}
                """)

                filename = os.path.basename(image_path)
                self.info_label.setText(f"📁 {filename}")
            else:
                self.image_label.setText("Failed to load image")
                self.info_label.setText("")
        else:
            self.image_label.setText("Image not found")
            self.info_label.setText("")

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
        """Clear all session images"""
        self.session_images = []
        self.current_index = -1
        self.current_image_path = None
        self.image_label.setText("No images generated yet")
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                border: 2px dashed {COLORS['border']};
                border-radius: 8px;
                color: {COLORS['text_dim']};
                padding: 20px;
                min-height: 200px;
            }}
        """)
        self.info_label.setText("")
        self.position_label.setText("")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def open_images_folder(self):
        """Open the images folder in file explorer"""
        import subprocess
        images_dir = os.path.abspath("images")
        if os.path.exists(images_dir):
            if os.name == 'nt':
                os.startfile(images_dir)
            elif os.name == 'posix':
                subprocess.Popen(['open', images_dir] if os.uname().sysname == 'Darwin' else ['xdg-open', images_dir])
        else:
            os.makedirs(images_dir, exist_ok=True)
            if os.name == 'nt':
                os.startfile(images_dir)
            elif os.name == 'posix':
                subprocess.Popen(['open', images_dir] if os.uname().sysname == 'Darwin' else ['xdg-open', images_dir])

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
        self.session_videos = []
        self.current_index = -1
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title label
        self.title = QLabel("🎬 GENERATED VIDEOS")
        self.title.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_cyan']};
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
            }}
        """)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        # Video display area
        self.video_label = QLabel("No videos generated yet")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                border: 2px dashed {COLORS['border']};
                border-radius: 8px;
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
                border-radius: 4px;
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
                border-radius: 4px;
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
                border-radius: 4px;
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
                border-radius: 4px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['accent_cyan']};
            }}
        """)
        self.open_button.clicked.connect(self.open_videos_folder)
        layout.addWidget(self.open_button)

    def add_video(self, video_path):
        """Add a new video to the session gallery and display it"""
        if video_path and os.path.exists(video_path):
            if video_path not in self.session_videos:
                self.session_videos.append(video_path)
            self.current_index = len(self.session_videos) - 1
            self._display_current()

    def set_video(self, video_path):
        """Display a video - also adds to gallery if new"""
        self.add_video(video_path)

    def _display_current(self):
        """Display the video at current_index"""
        if not self.session_videos or self.current_index < 0:
            self.video_label.setText("No videos generated yet")
            self.info_label.setText("")
            self.position_label.setText("")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.play_button.setEnabled(False)
            return

        video_path = self.session_videos[self.current_index]
        self.current_video_path = video_path

        if os.path.exists(video_path):
            filename = os.path.basename(video_path)
            self.video_label.setText(f"🎬 {filename}\n\n(Click Play to view)")
            self.video_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS['bg_medium']};
                    border: 2px solid {COLORS['accent_cyan']};
                    border-radius: 8px;
                    color: {COLORS['text_bright']};
                    padding: 20px;
                    min-height: 150px;
                }}
            """)
            self.info_label.setText(f"📁 {filename}")
            self.play_button.setEnabled(True)
        else:
            self.video_label.setText("Video not found")
            self.info_label.setText("")
            self.play_button.setEnabled(False)

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
        """Clear all session videos"""
        self.session_videos = []
        self.current_index = -1
        self.current_video_path = None
        self.video_label.setText("No videos generated yet")
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_medium']};
                border: 2px dashed {COLORS['border']};
                border-radius: 8px;
                color: {COLORS['text_dim']};
                padding: 20px;
                min-height: 150px;
            }}
        """)
        self.info_label.setText("")
        self.position_label.setText("")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.play_button.setEnabled(False)

    def open_videos_folder(self):
        """Open the videos folder in file explorer"""
        import subprocess
        videos_dir = os.path.abspath("videos")
        if os.path.exists(videos_dir):
            if os.name == 'nt':
                os.startfile(videos_dir)
            elif os.name == 'posix':
                subprocess.Popen(['open', videos_dir] if os.uname().sysname == 'Darwin' else ['xdg-open', videos_dir])
        else:
            os.makedirs(videos_dir, exist_ok=True)
            if os.name == 'nt':
                os.startfile(videos_dir)
            elif os.name == 'posix':
                subprocess.Popen(['open', videos_dir] if os.uname().sysname == 'Darwin' else ['xdg-open', videos_dir])
